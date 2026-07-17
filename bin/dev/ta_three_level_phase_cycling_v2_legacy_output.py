#!/usr/bin/env python3
"""Run TA recipe v2 workflow and write legacy-demo shaped outputs.

这个脚本用于迁移检查：

    v2 generic pulse-sequence / phase-cycling 计算路径
    + legacy phase-cycling demo 的 output/checkpoint 目录结构

它不修改 solver、run_case、TA recipe v2 或旧 demo。checkpoint 仍保存
`DynamicsResult`，路径与旧 demo 对齐：

    <output_dir>/checkpoints/carrier_envelope_v2/*.ckp
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

LEGACY_DEMO_PATH = (
    REPO_ROOT
    / "bin"
    / "examples"
    / "ta"
    / "ta_three_level_intrinsic_response_phase_cycling_demo.py"
)
SMOKE_V2_PATH = REPO_ROOT / "bin" / "dev" / "smoke_ta_three_level_phase_cycling_v2.py"
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "bin"
    / "examples"
    / "ta"
    / "outputs"
    / "ta_three_level_intrinsic_response_phase_cycling_demo_v2_legacy_output"
)

from qudpy_sjh.experiments.pulse_sequence import AxisMetadataSpec, PhaseGrid, SingleRunCheckpointSettings  # noqa: E402
from qudpy_sjh.experiments.ta import (  # noqa: E402
    TADelayCenters,
    TAPhaseCyclingSpec,
    TASingleDelayPlan,
    build_ta_pump_probe_phase_cycling_plan,
    extract_ta_absorption_bundle,
)
from qudpy_sjh.utils.core import ParaNormalizer  # noqa: E402


def _load_module(path: Path, module_name: str):
    if not path.exists():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _checkpoint_path(output_dir: Path, case_key: str) -> Path:
    return output_dir / "checkpoints" / "carrier_envelope_v2" / f"{case_key}.ckp"


def _with_checkpoint(plan, *, output_dir: Path, case_key: str, config):
    if not bool(config.use_checkpoints):
        return plan
    path = _checkpoint_path(output_dir, case_key)
    path.parent.mkdir(parents=True, exist_ok=True)
    return replace(
        plan,
        checkpoint=SingleRunCheckpointSettings(
            enabled=True,
            checkpoint_path=path,
            force_run=bool(config.force_run),
        ),
    )


def _physical_params_compatible(expected, actual) -> tuple[bool, str]:
    if actual is None:
        return False, "checkpoint result has no physical_params"
    scalar_fields = ("t_start_fs", "t_end_fs", "dt_fs")
    for name in scalar_fields:
        expected_value = float(getattr(expected, name))
        actual_value = float(getattr(actual, name))
        if not np.isclose(expected_value, actual_value, rtol=0.0, atol=1.0e-12):
            return False, f"{name} mismatch: expected {expected_value:g}, got {actual_value:g}"

    if str(expected.solver_mode) != str(actual.solver_mode):
        return False, f"solver_mode mismatch: expected {expected.solver_mode!r}, got {actual.solver_mode!r}"
    if tuple(expected.basis or ()) != tuple(actual.basis or ()):
        return False, f"basis mismatch: expected {expected.basis!r}, got {actual.basis!r}"

    expected_energy = np.asarray(expected.energies_eV, dtype=float)
    actual_energy = np.asarray(actual.energies_eV, dtype=float)
    if expected_energy.shape != actual_energy.shape or not np.allclose(expected_energy, actual_energy, rtol=0.0, atol=1.0e-12):
        return False, "energies_eV mismatch"

    expected_dipole = np.asarray(expected.dipole_matrix_D, dtype=np.complex128)
    actual_dipole = np.asarray(actual.dipole_matrix_D, dtype=np.complex128)
    if expected_dipole.shape != actual_dipole.shape or not np.allclose(expected_dipole, actual_dipole, rtol=0.0, atol=1.0e-12):
        return False, "dipole_matrix_D mismatch"

    return True, "ok"


def _execute_with_checkpoint(plan, *, output_dir: Path, case_key: str, config):
    local_plan = _with_checkpoint(plan, output_dir=output_dir, case_key=case_key, config=config)
    result = local_plan.execute()
    if not bool(config.use_checkpoints) or bool(config.force_run):
        return result

    expected_params = plan.make_params()
    ok, reason = _physical_params_compatible(expected_params, result.dynamics_result.physical_params)
    if ok:
        return result

    checkpoint_path = _checkpoint_path(output_dir, case_key)
    print(
        "Incompatible checkpoint loaded; rerunning and overwriting: "
        f"{checkpoint_path} ({reason})"
    )
    rerun_plan = replace(
        local_plan,
        checkpoint=replace(local_plan.checkpoint, force_run=True),
    )
    return rerun_plan.execute()


def _assert_reference_axis(name: str, reference: np.ndarray, current: np.ndarray, *, case_key: str) -> None:
    try:
        _assert_same_axis(name, reference, current)
    except ValueError as exc:
        raise ValueError(
            f"{name} axis mismatch after checkpoint validation for case_key={case_key!r}. "
            "If this persists, remove the output checkpoint directory or rerun with --force-run."
        ) from exc


def _assert_same_axis(name: str, reference: np.ndarray, current: np.ndarray) -> None:
    if reference.shape != current.shape:
        raise ValueError(f"{name} axis shape mismatch: {reference.shape} vs {current.shape}")
    diff = float(np.max(np.abs(reference - current))) if reference.size else 0.0
    if diff > 1.0e-12:
        raise ValueError(f"{name} axis mismatch. max_abs_diff={diff:.6e}")


def _read_spectrum(result) -> dict[str, np.ndarray]:
    if result.readout is None or result.readout.spectrum is None:
        raise ValueError("SingleRunResult.readout.spectrum is required.")
    spectrum = result.readout.spectrum
    required = ("absorption", "energy_eV", "omega_fs_inv")
    missing = [key for key in required if key not in spectrum]
    if missing:
        raise KeyError(f"readout.spectrum missing required keys: {missing}")
    return {
        "absorption": np.asarray(spectrum["absorption"], dtype=float),
        "energy_eV": np.asarray(spectrum["energy_eV"], dtype=float),
        "omega_fs_inv": np.asarray(spectrum["omega_fs_inv"], dtype=float),
    }


def _phase_case_key(legacy, *, phase: float, delay_fs: float) -> str:
    return f"phase_{legacy.phase_label(phase)}_delay_{legacy.safe_delay_label(delay_fs)}_fs"


def _trace_payload_from_result(legacy, result, *, number_density_m3: float) -> dict[str, Any]:
    physical = result.dynamics_result.physical_params
    if physical is None:
        raise ValueError("DynamicsResult.physical_params is required for preview trace.")
    t_fs = np.asarray(result.dynamics_result.times_fs, dtype=float)
    density = result.dynamics_result.density_array()
    polarization_t = legacy.polarization_C_per_m2(
        density,
        physical.dipole_matrix_D,
        float(number_density_m3),
    )
    field_t = np.asarray(physical.field(t_fs), dtype=float)
    return {
        "result": result.dynamics_result,
        "times_fs": t_fs,
        "density": density,
        "polarization_t": polarization_t,
        "field_t": field_t,
    }


def _make_config(legacy, smoke_v2, args: argparse.Namespace):
    config = legacy.DemoConfig()
    if bool(args.quick):
        config = smoke_v2._quick_config(config)
    config = replace(
        config,
        use_checkpoints=not bool(args.no_checkpoints),
        force_run=bool(args.force_run),
        plot_use_wavelength=bool(args.wavelength),
    )
    return config


def _select_delays(config, *, quick: bool, max_delays: int | None) -> tuple[float, ...]:
    delays = tuple(float(x) for x in (config.quick_probe_delays_fs if quick else config.probe_delays_fs))
    if max_delays is None:
        return delays
    if int(max_delays) < 1:
        raise ValueError("--max-delays must be >= 1.")
    return delays[: int(max_delays)]


BaseParamsBuilder = Callable[[Any, Any, Any, Any], tuple[Any, dict[str, Any]]]


def run_v2_legacy_output(
    args: argparse.Namespace,
    *,
    base_params_builder: BaseParamsBuilder | None = None,
    example_name: str = "ta_three_level_intrinsic_response_phase_cycling_demo_v2_legacy_output",
    workflow_extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    legacy = _load_module(LEGACY_DEMO_PATH, "ta_phase_cycling_legacy_output_reference")
    smoke_v2 = _load_module(SMOKE_V2_PATH, "ta_phase_cycling_v2_smoke_reference")

    config = _make_config(legacy, smoke_v2, args)
    output_dir = args.output_dir.resolve() if args.output_dir is not None else DEFAULT_OUTPUT_DIR
    data_dir = output_dir / "data"
    plot_dir = output_dir / "figures" / "plot"
    legacy_dir = output_dir / "figures" / "legacy"
    preview_dir = output_dir / "figures" / "preview"
    delays = _select_delays(config, quick=bool(args.quick), max_delays=args.max_delays)
    delays_array = np.asarray(delays, dtype=float)

    pump, probe = smoke_v2._make_pulses(config)
    builder_metadata: dict[str, Any] = {}
    if base_params_builder is None:
        base_params = smoke_v2._make_base_params(config, probe.field_template)
    else:
        base_params, builder_metadata = base_params_builder(legacy, smoke_v2, config, probe)
    readout = smoke_v2._make_readout(config)
    normalizer = ParaNormalizer()

    phase_grid = PhaseGrid({"pump": tuple(float(x) for x in config.pump_phase_cases_rad)})
    phase_names = [legacy.phase_label(float(x)) for x in config.pump_phase_cases_rad]
    phase_cycling = TAPhaseCyclingSpec(
        phase_grid=phase_grid,
        target_phase_vector={"pump": 0},
        projection_quantity="readout.spectrum.absorption",
        signal_name="pump_phase_avg_absorption",
        axis_specs=(
            AxisMetadataSpec(name="energy_eV", quantity="readout.spectrum.energy_eV", source="validate_all_cases"),
            AxisMetadataSpec(name="omega_fs_inv", quantity="readout.spectrum.omega_fs_inv", source="validate_all_cases"),
        ),
        normalize=True,
        sign=-1,
        metadata={
            "meaning": "legacy pump-phase average",
            "not_a_universal_ta_phase_convention": True,
        },
    )

    first_plan = TASingleDelayPlan(
        base_params=base_params,
        pump=pump,
        probe=probe,
        delay=TADelayCenters(delay_fs=float(delays[0]), probe_center_fs=float(config.probe_center_fs)),
        normalizer=normalizer,
        readout=readout,
        case_name="v2_legacy_output_delay_000",
        metadata={"legacy_reference": str(LEGACY_DEMO_PATH)},
    )
    probe_result = _execute_with_checkpoint(
        first_plan.make_probe_only_plan(),
        output_dir=output_dir,
        case_key="probe_only",
        config=config,
    )
    probe_bundle = extract_ta_absorption_bundle(probe_result, case_name="probe_only")
    probe_spectrum = _read_spectrum(probe_result)
    energy_eV = np.asarray(probe_spectrum["energy_eV"], dtype=float)
    omega_fs_inv = np.asarray(probe_spectrum["omega_fs_inv"], dtype=float)
    s_probe = np.asarray(probe_bundle.absorption, dtype=float)

    phase_rows: list[list[np.ndarray]] = [[] for _ in phase_names]
    max_trace_error = 0.0
    max_hermiticity_error = 0.0

    for delay_index, delay_fs in enumerate(delays):
        print(f"[v2-legacy-output] delay={delay_fs:g} fs, pump phases={len(phase_grid)}")
        ta_plan = TASingleDelayPlan(
            base_params=base_params,
            pump=pump,
            probe=probe,
            delay=TADelayCenters(delay_fs=float(delay_fs), probe_center_fs=float(config.probe_center_fs)),
            normalizer=normalizer,
            readout=readout,
            case_name=f"v2_legacy_output_delay_{delay_index:03d}",
            metadata={"legacy_reference": str(LEGACY_DEMO_PATH)},
        )
        phase_plan = build_ta_pump_probe_phase_cycling_plan(
            ta_plan,
            phase_cycling=phase_cycling,
            case_name=f"{ta_plan.case_name}_pump_phase_avg",
        )

        def execute_phase_case(case_plan):
            phase = float(case_plan.field_plan.phase_vector["pump"])
            case_key = _phase_case_key(legacy, phase=phase, delay_fs=float(delay_fs))
            return _execute_with_checkpoint(case_plan, output_dir=output_dir, case_key=case_key, config=config)

        phase_result = phase_plan.execute(executor=execute_phase_case)
        for phase_index, record in enumerate(phase_result.case_records):
            if record.single_run_result is None:
                raise ValueError("PhaseCyclingResult.case_records must store SingleRunResult.")
            spectrum = _read_spectrum(record.single_run_result)
            phase = float(record.phase_vector["pump"])
            case_key = _phase_case_key(legacy, phase=phase, delay_fs=float(delay_fs))
            _assert_reference_axis("energy_eV", energy_eV, spectrum["energy_eV"], case_key=case_key)
            _assert_reference_axis("omega_fs_inv", omega_fs_inv, spectrum["omega_fs_inv"], case_key=case_key)
            phase_rows[phase_index].append(np.asarray(spectrum["absorption"], dtype=float) - s_probe)
            dyn = record.single_run_result.dynamics_result
            max_trace_error = max(max_trace_error, float(dyn.max_trace_error()))
            max_hermiticity_error = max(max_hermiticity_error, float(dyn.max_hermiticity_error()))

    phase_maps = [np.vstack(rows) for rows in phase_rows]
    phase_stack = np.stack(phase_maps, axis=0)
    ta_phase_avg = np.mean(phase_stack, axis=0)
    ta_phase_rms = np.sqrt(np.mean(phase_stack ** 2, axis=0))
    ta_phase_avg_unitnorm_diagnostic = np.mean(
        np.stack([legacy.normalize_map_for_diagnostic(item, scale="p99abs") for item in phase_maps], axis=0),
        axis=0,
    )
    shared_vlim = legacy.robust_vlim(phase_maps + [ta_phase_avg], percentile=99.0)
    phase_mean_maxabs = float(
        np.mean(
            [
                legacy.map_stats(f"phase_{label}", ta_map)["maxabs"]
                for label, ta_map in zip(phase_names, phase_maps)
            ]
        )
    )

    stats_rows = []
    for label, ta_map in zip(phase_names, phase_maps):
        stats_rows.append(legacy.map_stats(f"TA_phase_{label}_full_energy", ta_map, reference_maxabs=phase_mean_maxabs))
        stats_rows.append(
            legacy.map_stats(
                f"TA_phase_{label}_displayed_energy",
                legacy.displayed_energy_map_values(energy_eV, ta_map, config),
                reference_maxabs=phase_mean_maxabs,
            )
        )
    stats_rows.append(legacy.map_stats("TA_phase_avg_raw_full_energy", ta_phase_avg, reference_maxabs=phase_mean_maxabs))
    stats_rows.append(
        legacy.map_stats(
            "TA_phase_avg_raw_displayed_energy",
            legacy.displayed_energy_map_values(energy_eV, ta_phase_avg, config),
            reference_maxabs=phase_mean_maxabs,
        )
    )
    stats_rows.append(
        legacy.map_stats(
            "TA_phase_rms_displayed_energy",
            legacy.displayed_energy_map_values(energy_eV, ta_phase_rms, config),
            reference_maxabs=phase_mean_maxabs,
        )
    )
    stats_rows.append(
        legacy.map_stats(
            "TA_phase_avg_unitnorm_diagnostic_displayed_energy",
            legacy.displayed_energy_map_values(energy_eV, ta_phase_avg_unitnorm_diagnostic, config),
        )
    )

    data_dir.mkdir(parents=True, exist_ok=True)
    stats_csv = legacy.write_csv_rows(data_dir / "map_stats.csv", stats_rows)
    stats_json = legacy.write_json(data_dir / "map_stats.json", {"map_stats": stats_rows})

    print("\nMap statistics:")
    for row in stats_rows:
        print(
            f"{row['name']:48s} "
            f"maxabs={row['maxabs']:.3e} "
            f"p99abs={row['p99abs']:.3e} "
            f"rms={row['rms']:.3e} "
            f"ratio={row['ratio_to_reference_maxabs']:.3e}"
        )
    print()

    figure_paths: dict[str, str] = {}
    phase_titles = {
        "0": "pump phase 0",
        "pi2": "pump phase π/2",
        "pi": "pump phase π",
        "3pi2": "pump phase 3π/2",
    }
    phase_filenames = {
        "0": "ta_phase_case_0.png",
        "pi2": "ta_phase_case_pi2.png",
        "pi": "ta_phase_case_pi.png",
        "3pi2": "ta_phase_case_3pi2.png",
    }

    figure_paths["phase_avg_autoscale"] = str(
        legacy.plot_one_map(
            path=plot_dir / "ta_phase_avg_autoscale.png",
            title="TA map: 4-step pump-phase average",
            energy_eV=energy_eV,
            delays_fs=delays_array,
            values=ta_phase_avg,
            config=config,
            vlim=None,
        )
    )
    compare_maps_norm = [
        ("phase 0", phase_maps[0]),
        ("phase π/2", phase_maps[1]),
        ("phase π", phase_maps[2]),
        ("phase 3π/2", phase_maps[3]),
        ("phase average", ta_phase_avg),
        ("phase-case RMS", ta_phase_rms),
    ]
    figure_paths["compare_autoscale"] = str(
        legacy.plot_compare(
            path=plot_dir / "ta_phase_cycling_compare_autoscale.png",
            energy_eV=energy_eV,
            delays_fs=delays_array,
            maps=compare_maps_norm,
            config=config,
            shared_vlim=None,
        )
    )

    for label, ta_map in zip(phase_names, phase_maps):
        figure_paths[f"legacy_phase_case_{label}"] = str(
            legacy.plot_one_map(
                path=legacy_dir / phase_filenames.get(label, f"ta_phase_case_{label}.png"),
                title=f"TA map: {phase_titles.get(label, label)}",
                energy_eV=energy_eV,
                delays_fs=delays_array,
                values=ta_map,
                config=config,
                vlim=shared_vlim,
            )
        )
    figure_paths["legacy_phase_avg_shared"] = str(
        legacy.plot_one_map(
            path=legacy_dir / "ta_phase_avg.png",
            title="TA map: 4-step pump-phase average, shared raw scale",
            energy_eV=energy_eV,
            delays_fs=delays_array,
            values=ta_phase_avg,
            config=config,
            vlim=shared_vlim,
        )
    )
    figure_paths["legacy_phase_avg_unitnorm_diagnostic"] = str(
        legacy.plot_one_map(
            path=legacy_dir / "ta_phase_avg_unitnorm_diagnostic.png",
            title="Diagnostic only: mean of p99-normalized phase maps",
            energy_eV=energy_eV,
            delays_fs=delays_array,
            values=ta_phase_avg_unitnorm_diagnostic,
            config=config,
            vlim=None,
        )
    )
    figure_paths["legacy_compare_shared"] = str(
        legacy.plot_compare(
            path=legacy_dir / "ta_phase_cycling_compare.png",
            energy_eV=energy_eV,
            delays_fs=delays_array,
            maps=compare_maps_norm,
            config=config,
            shared_vlim=shared_vlim,
        )
    )

    figure_paths["preview_figure_2_suppression_stats"] = str(
        legacy.plot_phase_cycling_suppression_stats(
            path=preview_dir / "figure_2_phase_cycling_suppression_stats.png",
            phase_names=phase_names,
            phase_maps=phase_maps,
            ta_phase_avg=ta_phase_avg,
            config=config,
            energy_eV=energy_eV,
        )
    )

    def nearest_delay_index(target_fs: float) -> int:
        return int(np.argmin(np.abs(delays_array - float(target_fs))))

    delay_to_trace: dict[float, dict[str, Any]] = {}
    for target_delay in config.preview_delays_fs:
        idx = nearest_delay_index(float(target_delay))
        actual_delay = float(delays_array[idx])
        trace_plan = TASingleDelayPlan(
            base_params=base_params,
            pump=pump,
            probe=probe,
            delay=TADelayCenters(delay_fs=actual_delay, probe_center_fs=float(config.probe_center_fs)),
            normalizer=normalizer,
            readout=readout,
            case_name=f"v2_legacy_output_trace_delay_{legacy.safe_delay_label(actual_delay)}",
            metadata={"legacy_reference": str(LEGACY_DEMO_PATH), "preview_trace": True},
        ).make_pump_probe_plan()
        trace_plan = replace(
            trace_plan,
            field_plan=replace(
                trace_plan.field_plan,
                phase_vector={"pump": 0.0},
                case_name=f"trace_delay_{legacy.safe_delay_label(actual_delay)}_phase_0",
            ),
            case_name=f"trace_delay_{legacy.safe_delay_label(actual_delay)}_phase_0",
        )
        trace_result = _execute_with_checkpoint(
            trace_plan,
            output_dir=output_dir,
            case_key=f"trace_delay_{legacy.safe_delay_label(actual_delay)}_phase_0",
            config=config,
        )
        trace_payload = _trace_payload_from_result(
            legacy,
            trace_result,
            number_density_m3=float(config.number_density_m3),
        )
        delay_to_trace[actual_delay] = trace_payload
        figure_paths[f"preview_rho_delay_{legacy.safe_delay_label(actual_delay)}"] = str(
            legacy.plot_rho_preview(
                path=preview_dir / f"rho_preview_delay_{legacy.safe_delay_label(actual_delay)}.png",
                delay_fs=actual_delay,
                payload=trace_payload,
            )
        )

        figure_paths[f"preview_diff_spectra_delay_{legacy.safe_delay_label(actual_delay)}"] = str(
            legacy.plot_selected_delay_phase_spectra(
                path=preview_dir / f"ta_diff_spectra_delay_{legacy.safe_delay_label(actual_delay)}.png",
                delay_fs=actual_delay,
                delay_index=idx,
                energy_eV=energy_eV,
                phase_maps=phase_maps,
                phase_names=phase_names,
                config=config,
            )
        )
        figure_paths[f"preview_diff_spectra_overlay_delay_{legacy.safe_delay_label(actual_delay)}"] = str(
            legacy.plot_selected_delay_phase_spectra_overlay(
                path=preview_dir / f"ta_diff_spectra_overlay_delay_{legacy.safe_delay_label(actual_delay)}.png",
                delay_fs=actual_delay,
                delay_index=idx,
                energy_eV=energy_eV,
                phase_maps=phase_maps,
                phase_names=phase_names,
                config=config,
            )
        )
        figure_paths[f"preview_mean_spectrum_delay_{legacy.safe_delay_label(actual_delay)}"] = str(
            legacy.plot_selected_delay_mean_spectrum(
                path=preview_dir / f"ta_phase_avg_spectrum_delay_{legacy.safe_delay_label(actual_delay)}.png",
                delay_fs=actual_delay,
                delay_index=idx,
                energy_eV=energy_eV,
                phase_maps=phase_maps,
                config=config,
            )
        )

    figure_paths["preview_figure_1_field_polarization"] = str(
        legacy.plot_field_polarization_selected_delays(
            path=preview_dir / "figure_1_field_polarization_selected_delays.png",
            delay_to_trace=delay_to_trace,
        )
    )

    all_delay_spectra_csv = legacy.write_all_delay_spectra_csv(
        data_dir / "ta_all_delay_spectra.csv",
        delays_fs=delays_array,
        energy_eV=energy_eV,
        phase_stack=phase_stack,
        phase_avg=ta_phase_avg,
        phase_rms=ta_phase_rms,
        phase_avg_unitnorm_diagnostic=ta_phase_avg_unitnorm_diagnostic,
        phase_labels=phase_names,
    )
    npz_path = data_dir / "ta_phase_cycling_comparison.npz"
    np.savez_compressed(
        npz_path,
        delays_fs=delays_array,
        energy_eV=energy_eV,
        omega_fs_inv=omega_fs_inv,
        wavelength_nm=legacy.HC_EV_NM / energy_eV,
        TA_phase_cases=phase_stack,
        TA_phase_avg=ta_phase_avg,
        TA_phase_rms=ta_phase_rms,
        TA_phase_avg_unitnorm_diagnostic=ta_phase_avg_unitnorm_diagnostic,
        phase_values_rad=np.asarray(config.pump_phase_cases_rad, dtype=float),
        phase_labels=np.asarray(phase_names, dtype=str),
    )

    meta = {
        "example_name": example_name,
        "quick": bool(args.quick),
        "output_dir": output_dir,
        "data_npz": npz_path,
        "all_delay_spectra_csv": all_delay_spectra_csv,
        "stats_csv": stats_csv,
        "stats_json": stats_json,
        "figures": figure_paths,
        "workflow": {
            "compute_path": "TA recipe v2 generic pulse-sequence + PhaseCyclingPlan",
            "output_shape": "legacy TA phase-cycling demo output layout",
            "legacy_reference": LEGACY_DEMO_PATH,
            "smoke_reference": SMOKE_V2_PATH,
            **dict(workflow_extra or {}),
        },
        "base_params_builder": builder_metadata,
        "checkpoint": {
            "enabled": bool(config.use_checkpoints),
            "force_run": bool(config.force_run),
            "checkpoint_dir": output_dir / "checkpoints" / "carrier_envelope_v2",
            "probe_only_case_key": "probe_only",
            "phase_case_pattern": "phase_<label>_delay_<delay>_fs",
            "trace_case_pattern": "trace_delay_<delay>_phase_0",
        },
        "phase_cases": {
            "pump_phase_rad": list(config.pump_phase_cases_rad),
            "probe_phase_rad": config.probe_phase_rad,
            "physical_phase_average": "unweighted arithmetic mean across four pump phases",
            "diagnostic_unitnorm_average": (
                "Each phase map is divided by its own p99 abs value before averaging. "
                "This is not a physical phase-cycling result."
            ),
            "phase_case_rms": "sqrt(mean(TA_phase_case^2, axis=phase)); diagnostic map.",
        },
        "field_convention": "E(t)=2E0 envelope(t-center) cos[omega*(t-center)+phase]",
        "model_parameters": {
            "basis": config.basis,
            "energies_eV": config.energies_eV,
            "dipole_matrix_D": config.dipole_matrix_D,
        },
        "field_parameters": {
            "pump_E0_MV_per_cm": config.pump_E0_MV_per_cm,
            "probe_E0_MV_per_cm": config.probe_E0_MV_per_cm,
            "pump_laser_energy_eV": config.pump_laser_energy_eV,
            "probe_laser_energy_eV": config.probe_laser_energy_eV,
            "pump_sigma_fs": config.pump_sigma_fs,
            "probe_sigma_fs": config.probe_sigma_fs,
            "probe_center_fs": config.probe_center_fs,
        },
        "time_grid": {
            "t_start_fs": config.t_start_fs,
            "t_end_fs": config.t_end_fs,
            "dt_fs": config.dt_fs,
            "n_time_points": int(np.asarray(probe_result.dynamics_result.times_fs).size),
        },
        "spectroscopy": {
            "definition": "absorption = omega * Im[P(omega)/E_probe(omega)]",
            "TA_definition": "S_TA = S_pump_probe - S_probe_only",
            "number_density_m3": config.number_density_m3,
            "window": config.window,
            "subtract_mean": config.subtract_mean,
            "rel_threshold": config.rel_threshold,
            "zero_padding_factor": config.zero_padding_factor,
        },
        "diagnostics": {
            "max_trace_error": max_trace_error,
            "max_hermiticity_error": max_hermiticity_error,
            "elapsed_s": time.perf_counter() - started,
        },
    }
    meta_path = legacy.write_json(output_dir / "meta.json", meta)

    print("TA v2 legacy-output run finished.")
    print(f"n delays          : {len(delays)}")
    print(f"energy points     : {energy_eV.size}")
    print(f"output directory  : {output_dir}")
    print(f"checkpoint dir    : {output_dir / 'checkpoints' / 'carrier_envelope_v2'}")
    print(f"data npz          : {npz_path}")
    print(f"all-delay spectra : {all_delay_spectra_csv}")
    print(f"stats csv         : {stats_csv}")
    print(f"metadata          : {meta_path}")
    print(f"final phase avg   : {figure_paths['phase_avg_autoscale']}")
    print(f"compare autoscale : {figure_paths['compare_autoscale']}")

    return {
        "output_dir": str(output_dir),
        "data_npz": str(npz_path),
        "all_delay_spectra_csv": str(all_delay_spectra_csv),
        "stats_csv": str(stats_csv),
        "stats_json": str(stats_json),
        "meta_json": str(meta_path),
        "figures": figure_paths,
        "base_params_builder": builder_metadata,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for checkpoints, data, metadata, and figures.",
    )
    parser.add_argument("--force-run", action="store_true", help="Ignore existing checkpoints and rerun all simulations.")
    parser.add_argument("--no-checkpoints", action="store_true", help="Run without checkpoint load/save.")
    parser.add_argument("--quick", action="store_true", help="Use the v2 smoke quick grid for fast validation.")
    parser.add_argument("--wavelength", action="store_true", help="Plot wavelength instead of photon energy on the x-axis.")
    parser.add_argument(
        "--max-delays",
        type=int,
        default=None,
        help="Diagnostic option: run only the first N selected delays.",
    )
    return parser.parse_args()


def main() -> None:
    run_v2_legacy_output(parse_args())


if __name__ == "__main__":
    main()
