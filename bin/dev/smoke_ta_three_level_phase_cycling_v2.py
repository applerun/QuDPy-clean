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
import hashlib
import importlib.util
import json
import sys
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


def run_v2_workflow(*, quick: bool) -> dict[str, Any]:
    legacy = _load_legacy_demo_module()
    config = legacy.DemoConfig()
    if quick:
        config = _quick_config(config)

    delays_fs = tuple(float(x) for x in (config.quick_probe_delays_fs if quick else config.probe_delays_fs))
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
    probe_only_result = first_plan.execute_probe_only()
    probe_only_bundle = extract_ta_absorption_bundle(
        probe_only_result,
        case_name="shared_probe_only_reference",
    )
    probe_trace = _trace_summary(probe_only_result)

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
        phase_result = phase_plan.execute()
        projected_bundle = build_ta_phase_cycled_pump_probe_bundle(
            phase_result,
            phase_cycling=phase_cycling,
            metadata={"delay_fs": float(delay_fs)},
        )
        projected_readout = _projected_bundle_to_ta_readout_bundle(
            projected_bundle,
            case_name=f"{ta_plan.case_name}_pump_phase_avg",
        )
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
    delta = np.asarray(scan_map.delta_absorption)
    energy = np.asarray(scan_map.energy_eV, dtype=float)
    delays = np.asarray(scan_map.delays_fs, dtype=float)
    projected_stack = np.stack(projected_pump_probe_spectra, axis=0)
    probe_absorption = np.asarray(probe_only_bundle.absorption)

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
        "phase_cases_rad": list(phase_grid.phases_by_tag["pump"]),
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
            for index in sorted({0, len(delays) // 2, len(delays) - 1})
        ],
        "differences_from_legacy_demo": [
            "v2 script uses generic ReadoutSpec instead of legacy response_from_result wrapper",
            "v2 script does not use legacy checkpoints or output directory structure",
            "quick mode intentionally uses reduced delay/time grid and is not a numerical equivalence check",
        ],
    }
    return {
        "summary": summary,
        "scan_map": scan_map,
        "projected_pump_probe_absorption": projected_stack,
        "probe_only_absorption": probe_absorption,
    }


def _save_small_outputs(payload: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.json"
    summary_path.write_text(
        json.dumps(_json_safe(payload["summary"]), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    scan_map = payload["scan_map"]
    np.savez_compressed(
        out_dir / "ta_phase_cycling_v2_summary_arrays.npz",
        delays_fs=np.asarray(scan_map.delays_fs),
        energy_eV=np.asarray(scan_map.energy_eV),
        delta_absorption=np.asarray(scan_map.delta_absorption),
        projected_pump_probe_absorption=np.asarray(payload["projected_pump_probe_absorption"]),
        probe_only_absorption=np.asarray(payload["probe_only_absorption"]),
    )
    print(f"saved_summary_json: {summary_path}")
    print(f"saved_summary_npz: {out_dir / 'ta_phase_cycling_v2_summary_arrays.npz'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Use a small delay/time grid for fast smoke validation.")
    parser.add_argument("--out-dir", type=Path, default=None, help="Optional directory for small summary json/npz outputs.")
    parser.add_argument("--no-save", action="store_true", help="Do not save files even when --out-dir is provided.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run_v2_workflow(quick=bool(args.quick))
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

    if args.out_dir is not None and not args.no_save:
        _save_small_outputs(payload, args.out_dir)
    else:
        print("saved_files: none")


if __name__ == "__main__":
    main()
