#!/usr/bin/env python3
"""TA recipe v2 three-level pump-phase-average migration smoke.

这是开发 smoke / migration script，不是正式替代旧 demo。它只用当前已经
完成的 generic pulse-sequence、PhaseCycler 和 TA recipe v2 接口，最小复现
旧 phase-cycling TA demo 的核心 workflow：

    three-level system
    -> shared probe-only reference
    -> pump-probe phase cycling over pump phase
    -> target_phase_vector={"pump": 0} 做 pump-phase average
    -> TA contrast = phase-averaged pump-probe - probe-only
    -> delay × energy map

这里的 ``{"pump": 0}`` 只是复现旧 demo 的 pump-phase average 行为，不是
通用 TA phase-cycling 物理约定。脚本不修改 solver / run_case /
normalization / DynamicsResult，不修改旧 demo，不保存大型输出，不画图。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

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
TA_MAP_XLIM_EV = (1.40, 1.80)

from qudpy_sjh.experiments.pulse_sequence import AxisMetadataSpec, PhaseGrid, PulseSpec, ReadoutSpec  # noqa: E402
from qudpy_sjh.experiments.ta import (  # noqa: E402
    TADelayCenters,
    TAPhaseCyclingSpec,
    TAReadoutBundle,
    TASingleDelayPlan,
    build_ta_delay_scan_map,
    build_ta_phase_cycled_pump_probe_bundle,
    build_ta_pump_probe_phase_cycling_plan,
    compute_ta_contrast,
    extract_ta_absorption_bundle,
)
from qudpy_sjh.utils.core import NLevelPhysicalParams, ParaNormalizer, PureDephasingChannel  # noqa: E402
from qudpy_sjh.utils.fields.carrier_envelope import make_gaussian_carrier_envelope_field  # noqa: E402


def _load_legacy_demo_module():
    """加载旧 demo 模块以复用 DemoConfig；不调用旧 demo 的 main/run_demo。"""

    if not LEGACY_DEMO_PATH.exists():
        raise FileNotFoundError(f"Cannot find legacy demo: {LEGACY_DEMO_PATH}")
    spec = importlib.util.spec_from_file_location("ta_phase_cycling_legacy_reference", LEGACY_DEMO_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load legacy demo module: {LEGACY_DEMO_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _json_safe(value: Any) -> Any:
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _json_safe(value.to_dict())
    if hasattr(value, "__dataclass_fields__"):
        return _json_safe(asdict(value))
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, complex):
        return {"real": float(value.real), "imag": float(value.imag)}
    if isinstance(value, Path):
        return str(value)
    return value


def _hash_float_array(values: np.ndarray) -> str:
    array = np.asarray(values, dtype=np.float64)
    return hashlib.sha256(array.tobytes()).hexdigest()[:16]


def _range(values: np.ndarray) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        raise ValueError("range requires a non-empty array.")
    return float(np.min(array)), float(np.max(array))


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _write_csv_rows(path: Path, rows: list[dict[str, Any]]) -> Path:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


def _finite_stats(name: str, values: np.ndarray) -> dict[str, Any]:
    array = np.asarray(values)
    finite = np.asarray(array[np.isfinite(array)], dtype=float)
    if finite.size == 0:
        return {
            "name": name,
            "shape": tuple(array.shape),
            "min": None,
            "max": None,
            "mean": None,
            "rms": None,
            "maxabs": None,
            "p99abs": None,
        }
    return {
        "name": name,
        "shape": tuple(array.shape),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "mean": float(np.mean(finite)),
        "rms": float(np.sqrt(np.mean(finite ** 2))),
        "maxabs": float(np.max(np.abs(finite))),
        "p99abs": float(np.percentile(np.abs(finite), 99.0)),
    }


def _selected_delay_indices(delays_fs: np.ndarray) -> np.ndarray:
    delays = np.asarray(delays_fs, dtype=float)
    if delays.ndim != 1 or delays.size == 0:
        raise ValueError("delays_fs must be a non-empty 1D array.")
    indices = {0, int(np.argmin(np.abs(delays))), int(delays.size - 1)}
    if delays.size >= 5:
        indices.add(int(delays.size // 2))
    if delays.size >= 9:
        indices.add(int(delays.size // 4))
        indices.add(int((3 * delays.size) // 4))
    return np.asarray(sorted(indices), dtype=int)


def _quick_config(config):
    """基于旧 DemoConfig 做 quick smoke 缩小，不作为数值一致性验证。"""

    return replace(
        config,
        probe_delays_fs=(-20.0, 0.0, 20.0),
        quick_probe_delays_fs=(-20.0, 0.0, 20.0),
        t_start_fs=-120.0,
        t_end_fs=160.0,
        dt_fs=1.0,
        use_checkpoints=False,
        force_run=True,
    )


def _select_delays(config, *, quick: bool, max_delays: int | None) -> tuple[float, ...]:
    delays = tuple(float(x) for x in (config.quick_probe_delays_fs if quick else config.probe_delays_fs))
    if max_delays is None:
        return delays
    limit = int(max_delays)
    if limit < 1:
        raise ValueError("--max-delays must be >= 1.")
    return delays[:limit]


def _time_grid_points(config) -> int:
    return int(round((float(config.t_end_fs) - float(config.t_start_fs)) / float(config.dt_fs))) + 1


def _print_profile_config(
    *,
    mode: str,
    config,
    old_config,
    delays_fs: tuple[float, ...],
    phase_grid: PhaseGrid,
    phase_cycling: TAPhaseCyclingSpec,
    readout: ReadoutSpec,
    estimated_solver_runs: int,
    max_delays: int | None,
) -> None:
    old_delays = tuple(float(x) for x in old_config.probe_delays_fs)
    old_quick_delays = tuple(float(x) for x in old_config.quick_probe_delays_fs)
    print("[profile] config summary")
    print(f"[profile] mode: {mode}")
    print(f"[profile] n_delays: {len(delays_fs)}")
    print(f"[profile] delay_range_fs: {min(delays_fs):.6g} -> {max(delays_fs):.6g}")
    print(f"[profile] max_delays: {max_delays}")
    print(f"[profile] n_phase_cases: {len(phase_grid)}")
    print(f"[profile] phase_grid_tags: {list(phase_grid.tags)}")
    print(f"[profile] target_phase_vector: {phase_cycling.target_phase_vector}")
    print(f"[profile] n_time_points: {_time_grid_points(config)}")
    print(
        "[profile] time_grid: "
        f"t_start_fs={float(config.t_start_fs):.6g}, "
        f"t_end_fs={float(config.t_end_fs):.6g}, "
        f"dt_fs={float(config.dt_fs):.6g}"
    )
    print("[profile] solver_mode: lab_exact")
    print(f"[profile] readout_mode: {readout.mode}")
    print(f"[profile] readout_field_name: {readout.readout_field_name}")
    print(f"[profile] readout_zero_padding_factor: {readout.zero_padding_factor}")
    print(f"[profile] readout_rel_threshold: {readout.rel_threshold:.6g}")
    print(f"[profile] estimated_solver_runs: {estimated_solver_runs}")
    print("[profile] probe_only_shared: true")
    print("[profile] checkpoint_enabled: false")
    print(f"[profile] old DemoConfig full delays count/range: {len(old_delays)} / {min(old_delays):.6g} -> {max(old_delays):.6g}")
    print(f"[profile] old DemoConfig quick delays count/range: {len(old_quick_delays)} / {min(old_quick_delays):.6g} -> {max(old_quick_delays):.6g}")
    print(
        "[profile] old DemoConfig time_grid: "
        f"t_start_fs={float(old_config.t_start_fs):.6g}, "
        f"t_end_fs={float(old_config.t_end_fs):.6g}, "
        f"dt_fs={float(old_config.dt_fs):.6g}, "
        f"n_time_points={_time_grid_points(old_config)}"
    )
    print(f"[profile] old phase cases: {list(old_config.pump_phase_cases_rad)}")
    print(f"[profile] old checkpoint_enabled_default: {bool(old_config.use_checkpoints)}")
    print("[profile] old checkpoint path pattern: <output_dir>/checkpoints/carrier_envelope_v2/<case_key>.ckp")
    print("[profile] old output inventory:")
    for item in _legacy_output_inventory():
        print(f"[profile]   - {item}")
    print("[profile] v2 smoke output inventory: stdout summary only by default; optional CSV/JSON/NPZ/figures with --save/--save-figures")


def _new_timing() -> dict[str, Any]:
    return {
        "config_s": 0.0,
        "probe_only_execute_s": 0.0,
        "delay_total_s": 0.0,
        "scan_map_s": 0.0,
        "total_s": 0.0,
        "per_delay": [],
    }


def _legacy_output_inventory() -> tuple[str, ...]:
    return (
        "checkpoints/carrier_envelope_v2/probe_only.ckp",
        "checkpoints/carrier_envelope_v2/phase_<label>_delay_<delay>_fs.ckp",
        "checkpoints/carrier_envelope_v2/trace_delay_<delay>_phase_<label>.ckp",
        "data/map_stats.csv",
        "data/map_stats.json",
        "data/ta_all_delay_spectra.csv",
        "data/ta_phase_cycling_comparison.npz",
        "metadata summary json with figure/data paths",
        "figures/plot/ta_phase_avg_autoscale.png",
        "figures/plot/ta_phase_cycling_compare_autoscale.png",
        "figures/legacy/phase-case and shared-scale TA maps",
        "figures/preview/field, polarization, rho, selected-delay spectra diagnostics",
    )


def _make_pulses(config) -> tuple[PulseSpec, PulseSpec]:
    """用旧 demo 数值构造 v2 PulseSpec templates。

    数值来自 legacy DemoConfig；template center 固定为 0，由
    TASingleDelayPlan/TADelayCenters 在每个 delay 上移动到实际 center。
    """

    pump_template = make_gaussian_carrier_envelope_field(
        E0_MV_per_cm=float(config.pump_E0_MV_per_cm),
        laser_energy_eV=float(config.pump_laser_energy_eV),
        center_fs=0.0,
        sigma_fs=float(config.pump_sigma_fs),
        phase_rad=0.0,
        name="pump_template",
        metadata={"copied_from": "legacy DemoConfig pump field parameters"},
    )
    probe_template = make_gaussian_carrier_envelope_field(
        E0_MV_per_cm=float(config.probe_E0_MV_per_cm),
        laser_energy_eV=float(config.probe_laser_energy_eV),
        center_fs=0.0,
        sigma_fs=float(config.probe_sigma_fs),
        phase_rad=float(config.probe_phase_rad),
        name="probe_template",
        metadata={"copied_from": "legacy DemoConfig probe field parameters"},
    )
    pump = PulseSpec(
        name="pump",
        field_template=pump_template,
        template_center_fs=0.0,
        phase_tag="pump",
        independent_phase=True,
        metadata={"source": "legacy DemoConfig pump parameters"},
    )
    probe = PulseSpec(
        name="probe",
        field_template=probe_template,
        template_center_fs=0.0,
        phase_tag="probe",
        independent_phase=True,
        metadata={"source": "legacy DemoConfig probe parameters"},
    )
    return pump, probe


def _make_base_params(config, field_template) -> NLevelPhysicalParams:
    return NLevelPhysicalParams(
        energies_eV=tuple(float(x) for x in config.energies_eV),
        dipole_matrix_D=tuple(tuple(float(v) for v in row) for row in config.dipole_matrix_D),
        t_start_fs=float(config.t_start_fs),
        t_end_fs=float(config.t_end_fs),
        dt_fs=float(config.dt_fs),
        field=field_template,
        basis=tuple(config.basis),
        pure_dephasing_channels=(
            PureDephasingChannel(
                name="pure_dephasing_level_1",
                level=1,
                Tphi_fs=float(config.Tphi_1_fs),
            ),
            PureDephasingChannel(
                name="pure_dephasing_level_2",
                level=2,
                Tphi_fs=float(config.Tphi_2_fs),
            ),
        ),
        solver_mode="lab_exact",
        input_description="TA recipe v2 three-level pump-phase-average migration smoke.",
        input_metadata={
            "script": "bin/dev/smoke_ta_three_level_phase_cycling_v2.py",
            "legacy_reference": str(LEGACY_DEMO_PATH),
            "phase_average_note": "target_phase_vector={'pump': 0} reproduces legacy pump-phase average only.",
        },
    )


def _make_readout(config) -> ReadoutSpec:
    return ReadoutSpec(
        mode="absorption",
        number_density_m3=float(config.number_density_m3),
        readout_field_name="probe",
        window=config.window,
        subtract_mean=bool(config.subtract_mean),
        rel_threshold=float(config.rel_threshold),
        zero_padding_factor=int(config.zero_padding_factor),
        return_intermediates=True,
        metadata={
            "legacy_readout_reference": "legacy demo response_from_result/lab_frame_absorption_response settings",
        },
    )


def _projected_bundle_to_ta_readout_bundle(
    projected_bundle,
    *,
    case_name: str,
) -> TAReadoutBundle:
    absorption = np.real_if_close(np.asarray(projected_bundle.projected_signal), tol=1000)
    energy_eV = np.asarray(projected_bundle.axes["energy_eV"], dtype=float)
    omega_fs_inv = None
    if "omega_fs_inv" in projected_bundle.axes:
        omega_fs_inv = np.asarray(projected_bundle.axes["omega_fs_inv"], dtype=float)
    return TAReadoutBundle(
        case_name=case_name,
        absorption=absorption,
        energy_eV=energy_eV,
        omega_fs_inv=omega_fs_inv,
        metadata={
            "source": "ProjectedReadoutBundle",
            "signal_name": projected_bundle.signal_name,
            "scope": "pump_phase_averaged_pump_probe_absorption",
        },
    )


def _trace_summary(result) -> dict[str, float]:
    if result.dynamics_result is None:
        raise ValueError("SingleRunResult.dynamics_result is required for trace summary.")
    return {
        "max_trace_error": float(result.dynamics_result.max_trace_error()),
        "max_hermiticity_error": float(result.dynamics_result.max_hermiticity_error()),
    }


def run_v2_workflow(
    *,
    quick: bool,
    profile: bool = False,
    profile_limit: int = 30,
    max_delays: int | None = None,
) -> dict[str, Any]:
    total_t0 = time.perf_counter()
    timing = _new_timing()
    config_t0 = time.perf_counter()
    legacy = _load_legacy_demo_module()
    old_config = legacy.DemoConfig()
    config = old_config
    if quick:
        config = _quick_config(config)

    delays_fs = _select_delays(config, quick=quick, max_delays=max_delays)
    pump, probe = _make_pulses(config)
    base_params = _make_base_params(config, probe.field_template)
    readout = _make_readout(config)
    normalizer = ParaNormalizer()

    phase_grid = PhaseGrid({"pump": tuple(float(x) for x in config.pump_phase_cases_rad)})
    phase_cycling = TAPhaseCyclingSpec(
        phase_grid=phase_grid,
        target_phase_vector={"pump": 0},
        projection_quantity="readout.spectrum.absorption",
        signal_name="pump_phase_avg_absorption",
        axis_specs=(
            AxisMetadataSpec(
                name="energy_eV",
                quantity="readout.spectrum.energy_eV",
                source="validate_all_cases",
            ),
            AxisMetadataSpec(
                name="omega_fs_inv",
                quantity="readout.spectrum.omega_fs_inv",
                source="validate_all_cases",
            ),
        ),
        normalize=True,
        sign=-1,
        metadata={
            "meaning": "legacy pump-phase average",
            "not_a_universal_ta_phase_convention": True,
        },
    )
    estimated_solver_runs = 1 + len(delays_fs) * len(phase_grid)
    timing["config_s"] = time.perf_counter() - config_t0
    if profile:
        _print_profile_config(
            mode="quick" if quick else "full",
            config=config,
            old_config=old_config,
            delays_fs=delays_fs,
            phase_grid=phase_grid,
            phase_cycling=phase_cycling,
            readout=readout,
            estimated_solver_runs=estimated_solver_runs,
            max_delays=max_delays,
        )

    first_plan = TASingleDelayPlan(
        base_params=base_params,
        pump=pump,
        probe=probe,
        delay=TADelayCenters(delay_fs=delays_fs[0], probe_center_fs=float(config.probe_center_fs)),
        normalizer=normalizer,
        readout=readout,
        case_name="v2_three_level_delay_000",
        metadata={"legacy_reference": str(LEGACY_DEMO_PATH)},
    )
    probe_t0 = time.perf_counter()
    probe_only_result = first_plan.execute_probe_only()
    probe_only_bundle = extract_ta_absorption_bundle(
        probe_only_result,
        case_name="shared_probe_only_reference",
    )
    probe_trace = _trace_summary(probe_only_result)
    timing["probe_only_execute_s"] = time.perf_counter() - probe_t0
    if profile:
        print(
            "[profile] probe_only_execute: "
            f"{timing['probe_only_execute_s']:.6f} s, "
            f"max_trace_error={probe_trace['max_trace_error']:.6e}, "
            f"max_hermiticity_error={probe_trace['max_hermiticity_error']:.6e}"
        )

    contrasts = []
    delay_summaries = []
    projected_pump_probe_spectra = []

    for index, delay_fs in enumerate(delays_fs):
        print(f"[v2] delay={delay_fs:g} fs, pump phases={len(phase_grid)}")
        ta_plan = TASingleDelayPlan(
            base_params=base_params,
            pump=pump,
            probe=probe,
            delay=TADelayCenters(delay_fs=delay_fs, probe_center_fs=float(config.probe_center_fs)),
            normalizer=normalizer,
            readout=readout,
            case_name=f"v2_three_level_delay_{index:03d}",
            metadata={"legacy_reference": str(LEGACY_DEMO_PATH)},
        )
        phase_plan = build_ta_pump_probe_phase_cycling_plan(
            ta_plan,
            phase_cycling=phase_cycling,
            case_name=f"{ta_plan.case_name}_pump_phase_avg",
        )
        delay_t0 = time.perf_counter()
        phase_t0 = time.perf_counter()
        phase_result = phase_plan.execute()
        phase_execute_s = time.perf_counter() - phase_t0
        bundle_t0 = time.perf_counter()
        projected_bundle = build_ta_phase_cycled_pump_probe_bundle(
            phase_result,
            phase_cycling=phase_cycling,
            metadata={"delay_fs": float(delay_fs)},
        )
        projected_readout = _projected_bundle_to_ta_readout_bundle(
            projected_bundle,
            case_name=f"{ta_plan.case_name}_pump_phase_avg",
        )
        bundle_s = time.perf_counter() - bundle_t0
        contrast_t0 = time.perf_counter()
        contrast = compute_ta_contrast(
            projected_readout,
            probe_only_bundle,
            delay_fs=delay_fs,
            case_name=f"{ta_plan.case_name}_contrast",
            metadata={
                "phase_target_vector": dict(phase_cycling.target_phase_vector),
                "phase_average_note": "pump-phase average only; not universal TA target convention",
            },
        )
        contrast_s = time.perf_counter() - contrast_t0
        contrasts.append(contrast)
        projected_pump_probe_spectra.append(np.asarray(projected_readout.absorption))

        phase_trace_errors = [
            _trace_summary(record.single_run_result)
            for record in phase_result.case_records
            if record.single_run_result is not None
        ]
        delay_summaries.append(
            {
                "delay_fs": float(delay_fs),
                "phase_cases": len(phase_result.phase_vectors),
                "pump_probe_phase_cases": phase_trace_errors,
                "pump_probe_max_trace_error": float(
                    max(item["max_trace_error"] for item in phase_trace_errors)
                ),
                "pump_probe_max_hermiticity_error": float(
                    max(item["max_hermiticity_error"] for item in phase_trace_errors)
                ),
            }
        )
        delay_total_s = time.perf_counter() - delay_t0
        timing["delay_total_s"] += delay_total_s
        timing["per_delay"].append(
            {
                "index": int(index),
                "delay_fs": float(delay_fs),
                "n_phase_cases": len(phase_result.phase_vectors),
                "phase_execute_s": float(phase_execute_s),
                "bundle_s": float(bundle_s),
                "contrast_s": float(contrast_s),
                "total_delay_s": float(delay_total_s),
                "max_trace_error": delay_summaries[-1]["pump_probe_max_trace_error"],
                "max_hermiticity_error": delay_summaries[-1]["pump_probe_max_hermiticity_error"],
            }
        )
        if profile and (profile_limit <= 0 or len(timing["per_delay"]) <= int(profile_limit)):
            print(
                "[profile] delay "
                f"index={index}, delay_fs={float(delay_fs):.6g}, "
                f"n_phase_cases={len(phase_result.phase_vectors)}, "
                f"phase_execute_s={phase_execute_s:.6f}, "
                f"bundle_s={bundle_s:.6f}, "
                f"contrast_s={contrast_s:.6f}, "
                f"total_delay_s={delay_total_s:.6f}, "
                f"max_trace_error={delay_summaries[-1]['pump_probe_max_trace_error']:.6e}, "
                f"max_hermiticity_error={delay_summaries[-1]['pump_probe_max_hermiticity_error']:.6e}"
            )

    scan_t0 = time.perf_counter()
    scan_map = build_ta_delay_scan_map(
        contrasts,
        case_name="v2_three_level_pump_phase_avg_ta_map",
        validate_omega_axis=True,
        metadata={
            "legacy_reference": str(LEGACY_DEMO_PATH),
            "phase_target_vector": dict(phase_cycling.target_phase_vector),
            "phase_average_note": "target {'pump': 0} reproduces legacy pump-phase average behavior only",
        },
    )
    timing["scan_map_s"] = time.perf_counter() - scan_t0
    delta = np.asarray(scan_map.delta_absorption)
    energy = np.asarray(scan_map.energy_eV, dtype=float)
    delays = np.asarray(scan_map.delays_fs, dtype=float)
    projected_stack = np.stack(projected_pump_probe_spectra, axis=0)
    probe_absorption = np.asarray(probe_only_bundle.absorption)
    selected_indices = _selected_delay_indices(delays)

    summary = {
        "script": "bin/dev/smoke_ta_three_level_phase_cycling_v2.py",
        "mode": "quick" if quick else "full",
        "legacy_reference": str(LEGACY_DEMO_PATH),
        "n_delays": int(delays.size),
        "n_energy": int(energy.size),
        "delay_range_fs": _range(delays),
        "energy_range_eV": _range(energy),
        "map_shape": tuple(delta.shape),
        "max_abs_delta_absorption": float(np.max(np.abs(delta))),
        "phase_target_vector": dict(phase_cycling.target_phase_vector),
        "phase_grid_tags": list(phase_grid.tags),
        "n_phase_cases": int(len(phase_grid)),
        "phase_cases_rad": list(phase_grid.phases_by_tag["pump"]),
        "estimated_solver_runs": int(estimated_solver_runs),
        "probe_only_shared": True,
        "checkpoint_enabled": False,
        "n_time_points": _time_grid_points(config),
        "time_grid": {
            "t_start_fs": float(config.t_start_fs),
            "t_end_fs": float(config.t_end_fs),
            "dt_fs": float(config.dt_fs),
        },
        "solver_mode": "lab_exact",
        "readout": readout.to_dict(),
        "delay_fs_hash": _hash_float_array(delays),
        "energy_eV_hash": _hash_float_array(energy),
        "probe_only": {
            **probe_trace,
            "n_points": int(probe_absorption.size),
            "max_abs_absorption": float(np.max(np.abs(probe_absorption))),
        },
        "pump_probe_phase_trace_summary": {
            "max_trace_error": float(max(item["pump_probe_max_trace_error"] for item in delay_summaries)),
            "max_hermiticity_error": float(
                max(item["pump_probe_max_hermiticity_error"] for item in delay_summaries)
            ),
        },
        "per_delay": delay_summaries,
        "selected_delay_spectra_summary": [
            {
                "delay_fs": float(delays[index]),
                "max_abs_projected_pump_probe": float(np.max(np.abs(projected_stack[index]))),
                "max_abs_ta": float(np.max(np.abs(delta[index]))),
            }
            for index in selected_indices
        ],
        "selected_delay_indices": selected_indices.tolist(),
        "selected_delay_fs": delays[selected_indices].tolist(),
        "differences_from_legacy_demo": [
            "v2 script uses generic ReadoutSpec instead of legacy response_from_result wrapper",
            "v2 script does not use legacy checkpoints or output directory structure",
            "quick mode intentionally uses reduced delay/time grid and is not a numerical equivalence check",
        ],
        "future_ta_result_io_v2_migration_targets": list(_legacy_output_inventory()),
    }
    timing["total_s"] = time.perf_counter() - total_t0
    summary["timing"] = timing
    if profile:
        print(f"[profile] scan_map_build: {timing['scan_map_s']:.6f} s")
        print(
            "[profile] total timing: "
            f"config={timing['config_s']:.6f} s, "
            f"probe_only={timing['probe_only_execute_s']:.6f} s, "
            f"delay_total={timing['delay_total_s']:.6f} s, "
            f"scan_map={timing['scan_map_s']:.6f} s, "
            f"total={timing['total_s']:.6f} s"
        )
    return {
        "summary": summary,
        "scan_map": scan_map,
        "projected_pump_probe_absorption": projected_stack,
        "probe_only_absorption": probe_absorption,
    }


def _v2_arrays(payload: dict[str, Any]) -> dict[str, np.ndarray]:
    scan_map = payload["scan_map"]
    delays = np.asarray(scan_map.delays_fs, dtype=float)
    energy = np.asarray(scan_map.energy_eV, dtype=float)
    delta = np.asarray(scan_map.delta_absorption)
    projected = np.asarray(payload["projected_pump_probe_absorption"])
    probe = np.asarray(payload["probe_only_absorption"])
    selected = np.asarray(payload["summary"]["selected_delay_indices"], dtype=int)
    return {
        "delays_fs": delays,
        "energy_eV": energy,
        "delta_absorption": delta,
        "probe_only_absorption": probe,
        "phase_averaged_pump_probe_absorption": projected,
        "selected_delay_indices": selected,
        "selected_delay_fs": delays[selected],
        "selected_delta_absorption": delta[selected],
    }


def _save_v2_outputs(payload: dict[str, Any], out_dir: Path) -> dict[str, Path]:
    data_dir = out_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    arrays = _v2_arrays(payload)
    summary = dict(payload["summary"])

    npz_path = data_dir / "v2_ta_phase_cycling_output.npz"
    np.savez_compressed(npz_path, **arrays)

    stats_rows = [
        _finite_stats("delta_absorption", arrays["delta_absorption"]),
        _finite_stats("probe_only_absorption", arrays["probe_only_absorption"]),
        _finite_stats("phase_averaged_pump_probe_absorption", arrays["phase_averaged_pump_probe_absorption"]),
    ]
    stats_csv = _write_csv_rows(data_dir / "v2_map_stats.csv", stats_rows)
    stats_json = _write_json(data_dir / "v2_map_stats.json", {"map_stats": stats_rows})

    selected_rows: list[dict[str, Any]] = []
    selected = arrays["selected_delay_indices"]
    for delay_index in selected:
        for energy_index, energy_eV in enumerate(arrays["energy_eV"]):
            selected_rows.append(
                {
                    "delay_index": int(delay_index),
                    "delay_fs": float(arrays["delays_fs"][delay_index]),
                    "energy_index": int(energy_index),
                    "energy_eV": float(energy_eV),
                    "delta_absorption": _json_safe(arrays["delta_absorption"][delay_index, energy_index]),
                    "phase_averaged_pump_probe_absorption": _json_safe(
                        arrays["phase_averaged_pump_probe_absorption"][delay_index, energy_index]
                    ),
                }
            )
    selected_csv = _write_csv_rows(data_dir / "v2_selected_delay_spectra.csv", selected_rows)

    probe_rows = [
        {
            "energy_index": int(index),
            "energy_eV": float(energy_eV),
            "probe_only_absorption": _json_safe(arrays["probe_only_absorption"][index]),
        }
        for index, energy_eV in enumerate(arrays["energy_eV"])
    ]
    probe_csv = _write_csv_rows(data_dir / "v2_probe_only_spectrum.csv", probe_rows)

    summary_json = _write_json(data_dir / "v2_summary.json", summary)
    written = {
        "npz": npz_path,
        "map_stats_csv": stats_csv,
        "map_stats_json": stats_json,
        "selected_delay_spectra_csv": selected_csv,
        "probe_only_spectrum_csv": probe_csv,
        "summary_json": summary_json,
    }
    for label, path in written.items():
        print(f"saved_{label}: {path}")
    return written


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError("matplotlib is required for --save-figures.") from exc
    return plt


def _plot_values(values: np.ndarray) -> np.ndarray:
    return np.asarray(np.real_if_close(values, tol=1000), dtype=float)


def _centered_norm(vlim: float):
    """TA map 使用以 0 为中心的发散色标，保证零信号落在白色中心。"""

    from matplotlib.colors import TwoSlopeNorm

    return TwoSlopeNorm(vmin=-float(vlim), vcenter=0.0, vmax=float(vlim))


def _set_ta_map_xlim(ax) -> None:
    """TA map 横轴固定展示 1.4--1.8 eV。"""

    ax.set_xlim(*TA_MAP_XLIM_EV)


def _plot_ta_map(
    plt,
    *,
    path: Path,
    delays_fs: np.ndarray,
    energy_eV: np.ndarray,
    values: np.ndarray,
    title: str,
    vlim: float | None = None,
) -> Path:
    data = _plot_values(values)
    if vlim is None:
        finite = data[np.isfinite(data)]
        vlim = float(np.percentile(np.abs(finite), 99.0)) if finite.size else 1.0
        if not np.isfinite(vlim) or vlim <= 0.0:
            vlim = 1.0
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    mesh = ax.pcolormesh(
        np.asarray(energy_eV, dtype=float),
        np.asarray(delays_fs, dtype=float),
        data,
        shading="auto",
        cmap="RdBu_r",
        norm=_centered_norm(vlim),
    )
    ax.set_title(title)
    ax.set_xlabel("energy_eV")
    ax.set_ylabel("delay_fs")
    _set_ta_map_xlim(ax)
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label("delta_absorption")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def _save_v2_figures(payload: dict[str, Any], out_dir: Path, *, figure_format: str) -> dict[str, Path]:
    plt = _require_matplotlib()
    arrays = _v2_arrays(payload)
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    suffix = figure_format.lstrip(".")
    written: dict[str, Path] = {}

    written["v2_ta_map"] = _plot_ta_map(
        plt,
        path=figures_dir / f"v2_ta_map.{suffix}",
        delays_fs=arrays["delays_fs"],
        energy_eV=arrays["energy_eV"],
        values=arrays["delta_absorption"],
        title="v2 TA map",
    )

    selected = arrays["selected_delay_indices"]
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    for delay_index in selected:
        ax.plot(
            arrays["energy_eV"],
            _plot_values(arrays["delta_absorption"][delay_index]),
            label=f"{arrays['delays_fs'][delay_index]:g} fs",
        )
    ax.set_title("v2 selected delay spectra")
    ax.set_xlabel("energy_eV")
    ax.set_ylabel("delta_absorption")
    _set_ta_map_xlim(ax)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = figures_dir / f"v2_selected_delay_spectra.{suffix}"
    fig.savefig(path)
    plt.close(fig)
    written["v2_selected_delay_spectra"] = path

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.plot(arrays["energy_eV"], _plot_values(arrays["probe_only_absorption"]))
    ax.set_title("v2 probe-only spectrum")
    ax.set_xlabel("energy_eV")
    ax.set_ylabel("probe_only_absorption")
    _set_ta_map_xlim(ax)
    fig.tight_layout()
    path = figures_dir / f"v2_probe_only_spectrum.{suffix}"
    fig.savefig(path)
    plt.close(fig)
    written["v2_probe_only_spectrum"] = path

    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    for delay_index in selected:
        ax.plot(
            arrays["energy_eV"],
            _plot_values(arrays["phase_averaged_pump_probe_absorption"][delay_index]),
            label=f"{arrays['delays_fs'][delay_index]:g} fs",
        )
    ax.set_title("v2 phase-averaged pump-probe spectra")
    ax.set_xlabel("energy_eV")
    ax.set_ylabel("phase_averaged_pump_probe_absorption")
    _set_ta_map_xlim(ax)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = figures_dir / f"v2_phase_averaged_pump_probe_spectra.{suffix}"
    fig.savefig(path)
    plt.close(fig)
    written["v2_phase_averaged_pump_probe_spectra"] = path

    for label, figure_path in written.items():
        print(f"saved_figure_{label}: {figure_path}")
    return written


def _load_legacy_outputs(legacy_output_dir: Path) -> dict[str, Any]:
    data_dir = legacy_output_dir / "data"
    npz_path = data_dir / "ta_phase_cycling_comparison.npz"
    output: dict[str, Any] = {
        "legacy_output_dir": str(legacy_output_dir),
        "available_npz_keys": [],
        "missing_fields": [],
        "map_stats_rows": None,
    }
    if not npz_path.exists():
        output["missing_fields"].append(f"missing npz: {npz_path}")
        return output

    with np.load(npz_path, allow_pickle=False) as data:
        keys = sorted(data.files)
        output["available_npz_keys"] = keys
        if "delays_fs" in data:
            output["delays_fs"] = np.asarray(data["delays_fs"], dtype=float)
        if "energy_eV" in data:
            output["energy_eV"] = np.asarray(data["energy_eV"], dtype=float)
        for candidate in ("delta_absorption", "ta_map", "delta_spectrum", "TA_phase_avg"):
            if candidate in data:
                output["delta_absorption"] = np.asarray(data[candidate])
                output["delta_absorption_key"] = candidate
                break
        for candidate in ("probe_only_absorption", "probe_absorption", "S_probe_only"):
            if candidate in data:
                output["probe_only_absorption"] = np.asarray(data[candidate])
                output["probe_only_absorption_key"] = candidate
                break
        for candidate in ("phase_averaged_pump_probe_absorption", "pump_probe_phase_avg", "S_pump_probe_phase_avg"):
            if candidate in data:
                output["phase_averaged_pump_probe_absorption"] = np.asarray(data[candidate])
                output["phase_averaged_pump_probe_absorption_key"] = candidate
                break

    for field in ("delays_fs", "energy_eV", "delta_absorption"):
        if field not in output:
            output["missing_fields"].append(field)
    for optional in ("probe_only_absorption", "phase_averaged_pump_probe_absorption"):
        if optional not in output:
            output["missing_fields"].append(optional)

    stats_csv = data_dir / "map_stats.csv"
    if stats_csv.exists():
        with stats_csv.open(newline="", encoding="utf-8") as handle:
            output["map_stats_rows"] = list(csv.DictReader(handle))
    return output


def _compare_arrays(name: str, old: np.ndarray | None, new: np.ndarray | None, *, rtol: float, atol: float) -> dict[str, Any]:
    row: dict[str, Any] = {
        "name": name,
        "available": old is not None and new is not None,
        "rtol": float(rtol),
        "atol": float(atol),
    }
    if old is None or new is None:
        row["status"] = "missing"
        return row
    old_arr = np.asarray(old)
    new_arr = np.asarray(new)
    row["old_shape"] = tuple(old_arr.shape)
    row["v2_shape"] = tuple(new_arr.shape)
    if old_arr.shape != new_arr.shape:
        row["status"] = "shape_mismatch"
        row["allclose"] = False
        return row
    diff = np.asarray(new_arr - old_arr)
    row["max_abs_diff"] = float(np.max(np.abs(diff))) if diff.size else 0.0
    row["mean_abs_diff"] = float(np.mean(np.abs(diff))) if diff.size else 0.0
    old_max = float(np.max(np.abs(old_arr))) if old_arr.size else 0.0
    row["relative_max_diff"] = None if old_max == 0.0 else row["max_abs_diff"] / old_max
    row["allclose"] = bool(np.allclose(new_arr, old_arr, rtol=rtol, atol=atol))
    row["status"] = "ok" if row["allclose"] else "value_mismatch"
    return row


def _save_legacy_comparison(
    payload: dict[str, Any],
    *,
    legacy_output_dir: Path,
    out_dir: Path,
    figure_format: str,
    save_figures: bool,
) -> tuple[dict[str, Any], dict[str, Path]]:
    data_dir = out_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    arrays = _v2_arrays(payload)
    legacy = _load_legacy_outputs(legacy_output_dir)
    rtol = 1.0e-7
    atol = 1.0e-10
    rows = [
        _compare_arrays("delays_fs", legacy.get("delays_fs"), arrays["delays_fs"], rtol=rtol, atol=atol),
        _compare_arrays("energy_eV", legacy.get("energy_eV"), arrays["energy_eV"], rtol=rtol, atol=atol),
        _compare_arrays("delta_absorption", legacy.get("delta_absorption"), arrays["delta_absorption"], rtol=rtol, atol=atol),
        _compare_arrays("probe_only_absorption", legacy.get("probe_only_absorption"), arrays["probe_only_absorption"], rtol=rtol, atol=atol),
        _compare_arrays(
            "phase_averaged_pump_probe_absorption",
            legacy.get("phase_averaged_pump_probe_absorption"),
            arrays["phase_averaged_pump_probe_absorption"],
            rtol=rtol,
            atol=atol,
        ),
    ]
    comparison = {
        "legacy_output_dir": str(legacy_output_dir),
        "available_npz_keys": legacy.get("available_npz_keys", []),
        "missing_fields": legacy.get("missing_fields", []),
        "tolerances": {"rtol": rtol, "atol": atol},
        "comparisons": rows,
    }
    json_path = _write_json(data_dir / "v2_vs_legacy_comparison.json", comparison)
    csv_path = _write_csv_rows(data_dir / "v2_vs_legacy_comparison.csv", rows)
    written = {
        "comparison_json": json_path,
        "comparison_csv": csv_path,
    }
    print(f"saved_comparison_json: {json_path}")
    print(f"saved_comparison_csv: {csv_path}")
    if legacy.get("missing_fields"):
        print(f"legacy_missing_fields: {legacy['missing_fields']}")
    if legacy.get("available_npz_keys"):
        print(f"legacy_available_npz_keys: {legacy['available_npz_keys']}")
    if save_figures:
        written.update(_save_comparison_figures(arrays, legacy, out_dir, figure_format=figure_format))
    return comparison, written


def _save_comparison_figures(
    arrays: dict[str, np.ndarray],
    legacy: dict[str, Any],
    out_dir: Path,
    *,
    figure_format: str,
) -> dict[str, Path]:
    plt = _require_matplotlib()
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    suffix = figure_format.lstrip(".")
    written: dict[str, Path] = {}

    old_delta = legacy.get("delta_absorption")
    old_delays = legacy.get("delays_fs")
    old_energy = legacy.get("energy_eV")
    if old_delta is not None and old_delays is not None and old_energy is not None:
        old_values = _plot_values(old_delta)
        v2_values = _plot_values(arrays["delta_absorption"])
        if old_values.shape == v2_values.shape:
            diff = v2_values - old_values
            shared = max(
                float(np.percentile(np.abs(old_values[np.isfinite(old_values)]), 99.0)),
                float(np.percentile(np.abs(v2_values[np.isfinite(v2_values)]), 99.0)),
                1.0e-30,
            )
            diff_vlim = float(np.percentile(np.abs(diff[np.isfinite(diff)]), 99.0)) if np.any(np.isfinite(diff)) else 1.0
            fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.6), sharey=True)
            for ax, title, values, vlim in (
                (axes[0], "old TA map", old_values, shared),
                (axes[1], "v2 TA map", v2_values, shared),
                (axes[2], f"v2 - old, max_abs_diff={np.max(np.abs(diff)):.3e}", diff, diff_vlim),
            ):
                mesh = ax.pcolormesh(
                    arrays["energy_eV"],
                    arrays["delays_fs"],
                    values,
                    shading="auto",
                    cmap="RdBu_r",
                    norm=_centered_norm(vlim),
                )
                ax.set_title(title)
                ax.set_xlabel("energy_eV")
                _set_ta_map_xlim(ax)
                fig.colorbar(mesh, ax=ax)
            axes[0].set_ylabel("delay_fs")
            fig.tight_layout()
            path = figures_dir / f"compare_ta_maps_old_v2_diff.{suffix}"
            fig.savefig(path)
            plt.close(fig)
            written["compare_ta_maps_old_v2_diff"] = path

            selected = arrays["selected_delay_indices"]
            ncols = len(selected)
            fig, axes = plt.subplots(1, ncols, figsize=(5.0 * ncols, 4.0), squeeze=False)
            for ax, delay_index in zip(axes.ravel(), selected):
                ax.plot(arrays["energy_eV"], old_values[delay_index], label="old", linewidth=1.2)
                ax.plot(arrays["energy_eV"], v2_values[delay_index], "--", label="v2", linewidth=1.2)
                local_diff = float(np.max(np.abs(v2_values[delay_index] - old_values[delay_index])))
                ax.set_title(f"delay={arrays['delays_fs'][delay_index]:g} fs\nmax_abs_diff={local_diff:.3e}")
                ax.set_xlabel("energy_eV")
                ax.set_ylabel("delta_absorption")
                _set_ta_map_xlim(ax)
                ax.legend(fontsize=8)
            fig.tight_layout()
            path = figures_dir / f"compare_selected_delay_spectra_overlay.{suffix}"
            fig.savefig(path)
            plt.close(fig)
            written["compare_selected_delay_spectra_overlay"] = path

    if "probe_only_absorption" in legacy and "energy_eV" in legacy:
        old_probe = _plot_values(legacy["probe_only_absorption"])
        v2_probe = _plot_values(arrays["probe_only_absorption"])
        if old_probe.shape == v2_probe.shape:
            fig, ax = plt.subplots(figsize=(7.0, 4.2))
            ax.plot(arrays["energy_eV"], old_probe, label="old")
            ax.plot(arrays["energy_eV"], v2_probe, "--", label="v2")
            ax.set_title("probe-only spectrum overlay")
            ax.set_xlabel("energy_eV")
            ax.set_ylabel("probe_only_absorption")
            _set_ta_map_xlim(ax)
            ax.legend()
            fig.tight_layout()
            path = figures_dir / f"compare_probe_only_spectrum_overlay.{suffix}"
            fig.savefig(path)
            plt.close(fig)
            written["compare_probe_only_spectrum_overlay"] = path

    for label, path in written.items():
        print(f"saved_comparison_figure_{label}: {path}")
    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Use a small delay/time grid for fast smoke validation.")
    parser.add_argument("--save", action="store_true", help="Save minimal comparable v2 numeric outputs under --out-dir.")
    parser.add_argument("--out-dir", type=Path, default=None, help="Output directory for --save / --save-figures / --compare-legacy.")
    parser.add_argument("--no-save", action="store_true", help="Do not save files even when --out-dir is provided.")
    parser.add_argument("--save-figures", action="store_true", help="Save v2 diagnostic figures under --out-dir/figures.")
    parser.add_argument("--legacy-output-dir", type=Path, default=None, help="Existing legacy demo output directory for comparison.")
    parser.add_argument("--compare-legacy", action="store_true", help="Compare v2 outputs against --legacy-output-dir.")
    parser.add_argument("--figure-format", default="png", help="Figure file extension, default: png.")
    parser.add_argument("--profile", action="store_true", help="Print timing and static configuration diagnostics.")
    parser.add_argument(
        "--profile-limit",
        type=int,
        default=30,
        help="Maximum number of per-delay profile rows to print; use 0 for all rows.",
    )
    parser.add_argument(
        "--max-delays",
        type=int,
        default=None,
        help="Diagnostic option: run only the first N selected delays. Default keeps full/quick behavior unchanged.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    will_save = (bool(args.save) or bool(args.save_figures) or bool(args.compare_legacy)) and not bool(args.no_save)
    if will_save and args.out_dir is None:
        raise ValueError("--out-dir is required when saving outputs or comparing legacy outputs.")
    if args.compare_legacy and args.legacy_output_dir is None:
        raise ValueError("--compare-legacy requires --legacy-output-dir.")
    if args.compare_legacy and args.no_save:
        raise ValueError("--compare-legacy writes comparison files; remove --no-save.")

    payload = run_v2_workflow(
        quick=bool(args.quick),
        profile=bool(args.profile),
        profile_limit=int(args.profile_limit),
        max_delays=args.max_delays,
    )
    summary = payload["summary"]

    print("smoke_ta_three_level_phase_cycling_v2_ok")
    print(f"mode: {summary['mode']}")
    print(f"n_delays: {summary['n_delays']}")
    print(f"n_energy: {summary['n_energy']}")
    print(f"delay_range_fs: {summary['delay_range_fs'][0]:.6g} -> {summary['delay_range_fs'][1]:.6g}")
    print(f"energy_range_eV: {summary['energy_range_eV'][0]:.6f} -> {summary['energy_range_eV'][1]:.6f}")
    print(f"map_shape: {summary['map_shape']}")
    print(f"max_abs_delta_absorption: {summary['max_abs_delta_absorption']:.6e}")
    print(f"phase_target_vector: {summary['phase_target_vector']}")
    print(f"phase_cases_rad: {summary['phase_cases_rad']}")
    print(f"delay_fs_hash: {summary['delay_fs_hash']}")
    print(f"energy_eV_hash: {summary['energy_eV_hash']}")
    print(f"probe_only max_trace_error: {summary['probe_only']['max_trace_error']:.6e}")
    print(f"probe_only max_hermiticity_error: {summary['probe_only']['max_hermiticity_error']:.6e}")
    print(
        "pump_probe phase cases max_trace_error: "
        f"{summary['pump_probe_phase_trace_summary']['max_trace_error']:.6e}"
    )
    print(
        "pump_probe phase cases max_hermiticity_error: "
        f"{summary['pump_probe_phase_trace_summary']['max_hermiticity_error']:.6e}"
    )
    for item in summary["selected_delay_spectra_summary"]:
        print(
            "selected_delay_summary: "
            f"delay_fs={item['delay_fs']:.6g}, "
            f"max_abs_projected_pump_probe={item['max_abs_projected_pump_probe']:.6e}, "
            f"max_abs_ta={item['max_abs_ta']:.6e}"
        )

    if (args.save or args.compare_legacy) and not args.no_save:
        assert args.out_dir is not None
        _save_v2_outputs(payload, args.out_dir)
    if args.save_figures and not args.no_save:
        assert args.out_dir is not None
        _save_v2_figures(payload, args.out_dir, figure_format=str(args.figure_format))
    if args.compare_legacy:
        assert args.out_dir is not None
        assert args.legacy_output_dir is not None
        _save_legacy_comparison(
            payload,
            legacy_output_dir=args.legacy_output_dir,
            out_dir=args.out_dir,
            figure_format=str(args.figure_format),
            save_figures=bool(args.save_figures),
        )
    if not will_save:
        print("saved_files: none")


if __name__ == "__main__":
    main()
