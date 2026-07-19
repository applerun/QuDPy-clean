#!/usr/bin/env python3
"""Harmonic exciton ladder TA v2 factorial sanity-check demo.

本脚本是 systems maker + TA v2 phase-cycling pipeline 的 harmonic
sanity-check demo，不是正式 TAResultIO，也不是旧 TA demo 的替代品。
它不修改旧 demo、不修改 solver/run_case/normalization/DynamicsResult。

物理预期：

1. ``harmonic_control`` 是 sanity check：
   EIS=0, PB=1, EID=1。理想 harmonic oscillator ladder 下，GSB/SE/ESA
   可能强烈抵消，因此 TA-like nonlinear response 可能接近消失或明显减弱。

2. EIS 改变 X->XX transition energy，破坏 0-X 与 X-XX 的能量重合，使
   ESA 相对主 transition 移位。

3. PB 改变 X->XX transition dipole，破坏 harmonic oscillator 的 sqrt(q)
   dipole scaling，使 GSB/SE 与 ESA 不再严格抵消。

4. EID 改变 X->XX transition dephasing，破坏不同 transition 的 linewidth
   匹配；即使能量和 oscillator strength harmonic，也可能改变 cancellation。

本脚本使用当前 TA phase convention：

    target_phase_vector = {"pump": 0}

因此 phase grid 包含四个 pump phase cases。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_OUTPUT_DIR = REPO_ROOT / "bin" / "optical_bloch_plots" / "ta_harmonic_exciton_ladder_factorial_v2"
PHASE_VALUES_RAD = (0.0, 0.5 * math.pi, math.pi, 1.5 * math.pi)
TARGET_PHASE_VECTOR = {"pump": 0}
TA_MAP_XLIM_EV = (1.40, 1.80)

from qudpy_sjh.experiments.pulse_sequence import AxisMetadataSpec, PhaseGrid, PulseSpec, ReadoutSpec  # noqa: E402
from qudpy_sjh.experiments.ta import (  # noqa: E402
    TADelayCenters,
    TAPhaseCyclingSpec,
    TASingleDelayPlan,
    build_ta_pump_probe_phase_cycling_plan,
)
from qudpy_sjh.systems import make_base_physical_params_from_system, make_single_exciton_ladder_system  # noqa: E402
from qudpy_sjh.utils.constants import HC_EV_NM  # noqa: E402
from qudpy_sjh.utils.core import ParaNormalizer, PureDephasingChannel  # noqa: E402
from qudpy_sjh.utils.fields.carrier_envelope import make_gaussian_carrier_envelope_field  # noqa: E402


@dataclass(frozen=True)
class FactorialSettings:
    energy_1q_eV: float = 1.55
    mu_1q_D: float = 5.0
    gamma_1q_fs_inv: float = 1.0 / 120.0
    case_mode: str = "one_dim_scan"
    eis_on_eV: float = -0.02
    pb_on: float = 0.99
    eid_on: float = 1.1
    pb_scan: tuple[float, ...] = (0.7, 0.8, 0.9, 0.95, 0.99)
    eid_scan: tuple[float, ...] = (1.1, 1.3, 1.5)
    eis_scan_eV: tuple[float, ...] = (0.01, 0.02, 0.05, 0.1)
    pump_E0_MV_per_cm: float = 0.30
    probe_E0_MV_per_cm: float = 0.008
    pump_laser_energy_eV: float = 1.55
    probe_laser_energy_eV: float = 1.62
    pump_sigma_fs: float = 12.0
    probe_sigma_fs: float = 7.0
    number_density_m3: float = 1.0e24
    zero_padding_factor: int = 4
    rel_threshold: float = 1.0e-6
    probe_delays_fs: tuple[float, ...] = (
        -300.0, -220.0, -160.0, -110.0, -100.0, -80.0, -60.0, -45.0, -30.0,
        -20.0, -10.0, 0.0, 10.0, 20.0, 30.0, 45.0, 60.0, 80.0, 100.0, 110.0,
        150.0, 220.0, 320.0,
    )
    quick_probe_delays_fs: tuple[float, ...] = (-100.0, 0.0, 100.0)
    selected_lineout_delays_fs: tuple[float, ...] = (-100.0, 0.0, 100.0)
    t_start_fs: float = -260.0
    t_end_fs: float = 320.0
    dt_fs: float = 1.0


@dataclass(frozen=True)
class FactorialCase:
    case_name: str
    EIS_on: bool
    PB_on: bool
    EID_on: bool
    eis_eV: float
    pb: float
    eid: float


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


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def build_case_config(settings: FactorialSettings) -> list[FactorialCase]:
    if settings.case_mode == "factorial":
        off = {"eis_eV": 0.0, "pb": 1.0, "eid": 1.0}
        on = {
            "eis_eV": float(settings.eis_on_eV),
            "pb": float(settings.pb_on),
            "eid": float(settings.eid_on),
        }
        return [
            FactorialCase("harmonic_control", False, False, False, off["eis_eV"], off["pb"], off["eid"]),
            FactorialCase("EIS_only", True, False, False, on["eis_eV"], off["pb"], off["eid"]),
            FactorialCase("PB_only", False, True, False, off["eis_eV"], on["pb"], off["eid"]),
            FactorialCase("EID_only", False, False, True, off["eis_eV"], off["pb"], on["eid"]),
            FactorialCase("EIS_PB", True, True, False, on["eis_eV"], on["pb"], off["eid"]),
            FactorialCase("EIS_EID", True, False, True, on["eis_eV"], off["pb"], on["eid"]),
            FactorialCase("PB_EID", False, True, True, off["eis_eV"], on["pb"], on["eid"]),
            FactorialCase("EIS_PB_EID", True, True, True, on["eis_eV"], on["pb"], on["eid"]),
        ]
    if settings.case_mode != "one_dim_scan":
        raise ValueError(f"Unsupported case_mode: {settings.case_mode!r}")
    cases: list[FactorialCase] = []
    for value in settings.pb_scan:
        pb = float(value)
        cases.append(FactorialCase(f"PB_{pb:.2f}", False, True, False, 0.0, pb, 1.0))
    for value in settings.eid_scan:
        eid = float(value)
        cases.append(FactorialCase(f"EID_{eid:.2f}", False, False, True, 0.0, 1.0, eid))
    for value in settings.eis_scan_eV:
        eis = float(value)
        cases.append(FactorialCase(f"EIS_{eis:.2f}", True, False, False, eis, 1.0, 1.0))
    return cases


def make_system_for_case(case: FactorialCase, settings: FactorialSettings):
    return make_single_exciton_ladder_system(
        n_quantum=2,
        energy_1q_eV=float(settings.energy_1q_eV),
        mu_1q_D=float(settings.mu_1q_D),
        gamma_1q_fs_inv=float(settings.gamma_1q_fs_inv),
        eis_eV=float(case.eis_eV),
        pb=float(case.pb),
        eid=float(case.eid),
        initial_state="ground",
        name=f"harmonic_factorial_{case.case_name}",
        metadata={
            "case_name": case.case_name,
            "factorial_flags": {
                "EIS_on": bool(case.EIS_on),
                "PB_on": bool(case.PB_on),
                "EID_on": bool(case.EID_on),
            },
        },
    )


def _transition_dephasing_values(system) -> tuple[float, float]:
    gamma_0x = float(system.transition_dephasing_fs_inv[("0", "X")])
    gamma_xx = float(system.transition_dephasing_fs_inv[("X", "XX")])
    return gamma_0x, gamma_xx


def _pure_dephasing_channels_from_transition_gamma(case_name: str, gamma_0x: float, gamma_xx: float):
    """脚本级映射：用 level projector dephasing 近似指定 transition linewidth。

    对 pure dephasing projector channels，coherence ij 的 dephasing 约为
    0.5 * (rate_i + rate_j)。设 ground level rate 为 0，则：

        rate_X = 2 * gamma_0X
        rate_XX = 2 * gamma_X_XX - rate_X

    若 rate_XX 为负则 fail-fast，避免隐藏不物理输入。
    """

    rate_x = 2.0 * float(gamma_0x)
    rate_xx = 2.0 * float(gamma_xx) - rate_x
    if rate_x < 0.0 or rate_xx < -1.0e-15:
        raise ValueError(
            f"Cannot map transition dephasing to non-negative level dephasing for {case_name}: "
            f"gamma_0X={gamma_0x}, gamma_X_XX={gamma_xx}."
        )
    rate_xx = max(0.0, rate_xx)
    channels = []
    if rate_x > 0.0:
        channels.append(PureDephasingChannel(name=f"{case_name}_pure_dephasing_X", level=1, rate_fs_inv=rate_x))
    if rate_xx > 0.0:
        channels.append(PureDephasingChannel(name=f"{case_name}_pure_dephasing_XX", level=2, rate_fs_inv=rate_xx))
    return tuple(channels), {"rate_X_fs_inv": rate_x, "rate_XX_fs_inv": rate_xx}


def _make_pulses(settings: FactorialSettings) -> tuple[PulseSpec, PulseSpec]:
    pump_template = make_gaussian_carrier_envelope_field(
        E0_MV_per_cm=float(settings.pump_E0_MV_per_cm),
        laser_energy_eV=float(settings.pump_laser_energy_eV),
        center_fs=0.0,
        sigma_fs=float(settings.pump_sigma_fs),
        phase_rad=0.0,
        name="pump_template",
    )
    probe_template = make_gaussian_carrier_envelope_field(
        E0_MV_per_cm=float(settings.probe_E0_MV_per_cm),
        laser_energy_eV=float(settings.probe_laser_energy_eV),
        center_fs=0.0,
        sigma_fs=float(settings.probe_sigma_fs),
        phase_rad=0.0,
        name="probe_template",
    )
    return (
        PulseSpec(
            name="pump",
            field_template=pump_template,
            template_center_fs=0.0,
            phase_tag="pump",
            independent_phase=True,
        ),
        PulseSpec(
            name="probe",
            field_template=probe_template,
            template_center_fs=0.0,
            phase_tag="probe",
            independent_phase=True,
        ),
    )


def _make_readout(settings: FactorialSettings) -> ReadoutSpec:
    return ReadoutSpec(
        mode="absorption",
        number_density_m3=float(settings.number_density_m3),
        readout_field_name="probe",
        window="hann",
        subtract_mean=True,
        rel_threshold=float(settings.rel_threshold),
        zero_padding_factor=int(settings.zero_padding_factor),
        return_intermediates=True,
    )


def save_case_spectrum_csv(path: Path, *, energy_eV: np.ndarray, ta_signal: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    values = np.asarray(ta_signal, dtype=float)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["energy_eV", "wavelength_nm", "ta_signal"],
        )
        writer.writeheader()
        for energy, value in zip(np.asarray(energy_eV, dtype=float), values):
            writer.writerow(
                {
                    "energy_eV": float(energy),
                    "wavelength_nm": float(HC_EV_NM / energy),
                    "ta_signal": float(value),
                }
            )
    return path


def save_factorial_summary_csv(path: Path, rows: list[dict[str, Any]]) -> Path:
    if not rows:
        raise ValueError("No factorial summary rows to write.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


def _integrated_abs(energy_eV: np.ndarray, values: np.ndarray) -> float:
    return float(np.trapezoid(np.abs(np.asarray(values, dtype=float)), np.asarray(energy_eV, dtype=float)))


def _read_absorption_spectrum(result) -> dict[str, np.ndarray]:
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


def _selected_delays(settings: FactorialSettings, *, quick: bool) -> tuple[float, ...]:
    return tuple(float(x) for x in (settings.quick_probe_delays_fs if quick else settings.probe_delays_fs))


def _phase_label(phase: float) -> str:
    phase = float(phase) % (2.0 * math.pi)
    if np.isclose(phase, 0.0):
        return "0"
    if np.isclose(phase, 0.5 * math.pi):
        return "pi2"
    if np.isclose(phase, math.pi):
        return "pi"
    if np.isclose(phase, 1.5 * math.pi):
        return "3pi2"
    return f"{phase:.4f}".rstrip("0").rstrip(".").replace(".", "p")


def _delay_label(delay_fs: float) -> str:
    value = 0.0 if abs(float(delay_fs)) < 1.0e-12 else float(delay_fs)
    if float(value).is_integer():
        return str(int(value))
    return f"{value:g}".replace(".", "p")


def _format_optional_sci(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.3e}"


def _case_output_dir_name(case: FactorialCase) -> str:
    return f"EIS_{case.eis_eV:.2f}_PB_{case.pb:.2f}_EID_{case.eid:.2f}"


def _zero_delay_index(delays_fs: np.ndarray) -> int:
    return int(np.argmin(np.abs(np.asarray(delays_fs, dtype=float))))


def _saved_npz_path(case_dir: Path) -> Path:
    return case_dir / "data" / "ta_phase_cycling_comparison.npz"


def _saved_case_data_is_compatible(case_dir: Path, *, expected_delays_fs: tuple[float, ...]) -> bool:
    npz_path = _saved_npz_path(case_dir)
    if not npz_path.exists():
        return False
    required = {"delays_fs", "energy_eV", "omega_fs_inv", "ta_map", "TA_phase_cases", "phase_labels"}
    try:
        with np.load(npz_path, allow_pickle=False) as payload:
            if not required.issubset(set(payload.files)):
                return False
            delays = np.asarray(payload["delays_fs"], dtype=float)
            expected = np.asarray(expected_delays_fs, dtype=float)
            if delays.shape != expected.shape or not np.allclose(delays, expected, rtol=0.0, atol=1.0e-12):
                return False
            ta_map = np.asarray(payload["ta_map"], dtype=float)
            phase_cases = np.asarray(payload["TA_phase_cases"], dtype=float)
            return ta_map.shape[0] == delays.size and phase_cases.shape[1] == delays.size
    except Exception:
        return False


def _load_saved_case_result(case: FactorialCase, case_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    npz_path = _saved_npz_path(case_dir)
    if not npz_path.exists():
        raise FileNotFoundError(npz_path)
    with np.load(npz_path, allow_pickle=False) as payload:
        delays = np.asarray(payload["delays_fs"], dtype=float)
        energy = np.asarray(payload["energy_eV"], dtype=float)
        omega = np.asarray(payload["omega_fs_inv"], dtype=float)
        ta_map = np.asarray(payload["ta_map"], dtype=float)
        phase_cases = np.asarray(payload["TA_phase_cases"], dtype=float)
        phase_labels = [str(item) for item in np.asarray(payload["phase_labels"], dtype=str)]
    zero_idx = _zero_delay_index(delays)
    summary = {
        "case_name": case.case_name,
        "EIS_on": bool(case.EIS_on),
        "PB_on": bool(case.PB_on),
        "EID_on": bool(case.EID_on),
        "eis_eV": float(case.eis_eV),
        "pb": float(case.pb),
        "eid": float(case.eid),
        "basis": None,
        "energies_eV": None,
        "transition_energies": None,
        "transition_dipoles": None,
        "transition_dephasing": None,
        "phase_grid_size": int(len(PHASE_VALUES_RAD)),
        "target_phase_vector": dict(TARGET_PHASE_VECTOR),
        "n_delays": int(delays.size),
        "delay_min_fs": float(np.min(delays)),
        "delay_max_fs": float(np.max(delays)),
        "n_spectrum_points": int(energy.size),
        "energy_min_eV": float(np.min(energy)),
        "energy_max_eV": float(np.max(energy)),
        "max_trace_error": None,
        "max_hermiticity_error": None,
        "max_abs_ta_signal": float(np.max(np.abs(ta_map))),
        "integrated_abs_ta_signal": _integrated_abs(energy, ta_map[zero_idx]),
        "spectrum_csv": str(case_dir / "data" / "spectra" / f"{case.case_name}_ta_spectrum.csv"),
    }
    result = {
        "summary": summary,
        "delays_fs": delays,
        "energy_eV": energy,
        "omega_fs_inv": omega,
        "ta_map": ta_map,
        "TA_phase_cases": phase_cases,
        "probe_absorption": None,
        "system": None,
        "base_params_adapter": None,
    }
    legacy_outputs = {
        "case_names": [case.case_name],
        "delays_fs": delays,
        "energy_eV": energy,
        "omega_fs_inv": omega,
        "ta_map": ta_map,
        "TA_phase_cases": phase_cases,
        "phase_labels": phase_labels,
        "map_csv": case_dir / "data" / "ta_map.csv",
        "all_delay_spectra_csv": case_dir / "data" / "ta_all_delay_spectra.csv",
        "stats_csv": case_dir / "data" / "map_stats.csv",
        "stats_json": case_dir / "data" / "map_stats.json",
        "data_npz": npz_path,
    }
    return result, legacy_outputs


def run_one_case(
    case: FactorialCase,
    *,
    settings: FactorialSettings,
    delays_fs: tuple[float, ...],
    output_dir: Path,
    normalizer: ParaNormalizer,
) -> dict[str, Any]:
    system = make_system_for_case(case, settings)
    gamma_0x, gamma_xx = _transition_dephasing_values(system)
    dephasing_channels, dephasing_mapping = _pure_dephasing_channels_from_transition_gamma(
        case.case_name,
        gamma_0x,
        gamma_xx,
    )
    pump, probe = _make_pulses(settings)
    base_params = make_base_physical_params_from_system(
        system,
        field=probe.field_template,
        t_start_fs=float(settings.t_start_fs),
        t_end_fs=float(settings.t_end_fs),
        dt_fs=float(settings.dt_fs),
        solver_mode="lab_exact",
        pure_dephasing_channels=dephasing_channels,
        input_description="Harmonic exciton ladder factorial TA v2 sanity-check.",
        input_metadata={
            "factorial_case": asdict(case),
            "transition_to_level_dephasing_mapping": dephasing_mapping,
            "mapping_note": "Script-level mapping; systems adapter remains metadata-only for transition dephasing.",
        },
    )
    readout = _make_readout(settings)
    first_plan = TASingleDelayPlan(
        base_params=base_params,
        pump=pump,
        probe=probe,
        delay=TADelayCenters(delay_fs=float(delays_fs[0]), probe_center_fs=0.0),
        normalizer=normalizer,
        readout=readout,
        case_name=f"ta_factorial_{case.case_name}",
        metadata={"factorial_case": asdict(case)},
    )
    phase_grid = PhaseGrid({"pump": PHASE_VALUES_RAD})
    phase_cycling = TAPhaseCyclingSpec(
        phase_grid=phase_grid,
        target_phase_vector=TARGET_PHASE_VECTOR,
        projection_quantity="readout.spectrum.absorption",
        signal_name="pump_phase_avg_absorption",
        axis_specs=(
            AxisMetadataSpec(name="energy_eV", quantity="readout.spectrum.energy_eV", source="validate_all_cases"),
            AxisMetadataSpec(name="omega_fs_inv", quantity="readout.spectrum.omega_fs_inv", source="validate_all_cases"),
        ),
        normalize=True,
        sign=-1,
        metadata={
            "meaning": "legacy pump-phase average; TA is pump-probe absorption minus probe-only absorption",
            "not_a_universal_ta_phase_convention": True,
        },
    )
    print(f"[factorial] running {case.case_name}: EIS={case.eis_eV:g}, PB={case.pb:g}, EID={case.eid:g}")
    probe_result = first_plan.make_probe_only_plan().execute()
    probe_spectrum = _read_absorption_spectrum(probe_result)
    energy = np.asarray(probe_spectrum["energy_eV"], dtype=float)
    omega = np.asarray(probe_spectrum["omega_fs_inv"], dtype=float)
    probe_absorption = np.asarray(probe_spectrum["absorption"], dtype=float)
    phase_ta_by_delay = []
    trace_errors = []
    hermiticity_errors = []

    for delay_index, delay_fs in enumerate(delays_fs):
        print(f"[factorial]   delay={delay_fs:g} fs, pump phases={len(phase_grid)}")
        ta_plan = TASingleDelayPlan(
            base_params=base_params,
            pump=pump,
            probe=probe,
            delay=TADelayCenters(delay_fs=float(delay_fs), probe_center_fs=0.0),
            normalizer=normalizer,
            readout=readout,
            case_name=f"ta_factorial_{case.case_name}_delay_{delay_index:03d}",
            metadata={"factorial_case": asdict(case)},
        )
        phase_plan = build_ta_pump_probe_phase_cycling_plan(
            ta_plan,
            phase_cycling=phase_cycling,
            case_name=f"ta_factorial_{case.case_name}_delay_{delay_index:03d}_phase_cycled",
        )
        phase_result = phase_plan.execute()
        phase_rows = []
        for record in phase_result.case_records:
            if record.single_run_result is None:
                raise ValueError("PhaseCyclingResult.case_records must store SingleRunResult.")
            spectrum = _read_absorption_spectrum(record.single_run_result)
            if spectrum["energy_eV"].shape != energy.shape or not np.allclose(spectrum["energy_eV"], energy):
                raise ValueError(f"energy_eV axis mismatch for {case.case_name} delay={delay_fs:g}.")
            if spectrum["omega_fs_inv"].shape != omega.shape or not np.allclose(spectrum["omega_fs_inv"], omega):
                raise ValueError(f"omega_fs_inv axis mismatch for {case.case_name} delay={delay_fs:g}.")
            phase_rows.append(np.asarray(spectrum["absorption"], dtype=float) - probe_absorption)
            dyn = record.single_run_result.dynamics_result
            trace_errors.append(float(dyn.max_trace_error()))
            hermiticity_errors.append(float(dyn.max_hermiticity_error()))
        phase_ta_by_delay.append(np.stack(phase_rows, axis=0))

    trace_errors.append(float(probe_result.dynamics_result.max_trace_error()))
    hermiticity_errors.append(float(probe_result.dynamics_result.max_hermiticity_error()))
    delay_phase_ta = np.stack(phase_ta_by_delay, axis=0)
    phase_ta = np.moveaxis(delay_phase_ta, 1, 0)
    ta_map = np.mean(delay_phase_ta, axis=1)
    spectrum_csv = save_case_spectrum_csv(
        output_dir / "data" / "spectra" / f"{case.case_name}_ta_spectrum.csv",
        energy_eV=energy,
        ta_signal=ta_map[int(np.argmin(np.abs(np.asarray(delays_fs, dtype=float))))],
    )

    transition_energies = system.metadata["transition_energies_eV"]
    transition_dipoles = system.metadata["transition_dipoles_D"]
    transition_dephasing = system.metadata["transition_dephasing_fs_inv"]
    summary = {
        "case_name": case.case_name,
        "EIS_on": bool(case.EIS_on),
        "PB_on": bool(case.PB_on),
        "EID_on": bool(case.EID_on),
        "eis_eV": float(case.eis_eV),
        "pb": float(case.pb),
        "eid": float(case.eid),
        "basis": tuple(system.basis),
        "energies_eV": tuple(float(x) for x in system.energies_eV),
        "transition_energies": transition_energies,
        "transition_dipoles": transition_dipoles,
        "transition_dephasing": transition_dephasing,
        "phase_grid_size": int(len(phase_grid)),
        "target_phase_vector": dict(TARGET_PHASE_VECTOR),
        "n_delays": int(len(delays_fs)),
        "delay_min_fs": float(np.min(delays_fs)),
        "delay_max_fs": float(np.max(delays_fs)),
        "n_spectrum_points": int(energy.size),
        "energy_min_eV": float(np.min(energy)),
        "energy_max_eV": float(np.max(energy)),
        "max_trace_error": None if not trace_errors else float(max(trace_errors)),
        "max_hermiticity_error": None if not hermiticity_errors else float(max(hermiticity_errors)),
        "max_abs_ta_signal": float(np.max(np.abs(ta_map))),
        "integrated_abs_ta_signal": _integrated_abs(energy, ta_map[int(np.argmin(np.abs(np.asarray(delays_fs, dtype=float))))]),
        "spectrum_csv": str(spectrum_csv),
    }
    return {
        "summary": summary,
        "delays_fs": np.asarray(delays_fs, dtype=float),
        "energy_eV": energy,
        "omega_fs_inv": omega,
        "ta_map": ta_map,
        "TA_phase_cases": phase_ta,
        "probe_absorption": probe_absorption,
        "system": system.to_dict(include_arrays=True),
        "base_params_adapter": base_params.input_metadata.get("system_adapter"),
    }


def _summary_csv_rows(case_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = [
        "case_name",
        "EIS_on",
        "PB_on",
        "EID_on",
        "eis_eV",
        "pb",
        "eid",
        "max_abs_ta_signal",
        "integrated_abs_ta_signal",
        "n_delays",
        "delay_min_fs",
        "delay_max_fs",
        "n_spectrum_points",
        "energy_min_eV",
        "energy_max_eV",
        "max_trace_error",
        "max_hermiticity_error",
        "output_dir",
        "meta_json",
        "data_npz",
        "ta_map_csv",
        "all_delay_spectra_csv",
    ]
    return [{key: item["summary"][key] for key in keys} for item in case_results]


def _finite_values(arr: np.ndarray) -> np.ndarray:
    values = np.asarray(arr, dtype=float).ravel()
    return values[np.isfinite(values)]


def _display_energy_mask(energy_eV: np.ndarray) -> np.ndarray:
    energy = np.asarray(energy_eV, dtype=float)
    return (energy >= TA_MAP_XLIM_EV[0]) & (energy <= TA_MAP_XLIM_EV[1])


def _normalize_by_display_p99(values: np.ndarray, energy_eV: np.ndarray) -> tuple[np.ndarray, float, float, float]:
    raw = np.asarray(values, dtype=float)
    mask = _display_energy_mask(energy_eV)
    display_values = raw[..., mask] if np.any(mask) else raw
    finite = display_values[np.isfinite(display_values)]
    if finite.size == 0:
        return raw, 1.0, np.nan, np.nan
    scale = float(np.percentile(np.abs(finite), 99.0))
    if scale <= 0.0 or not np.isfinite(scale):
        scale = 1.0
    return raw / scale, scale, float(np.min(finite)), float(np.max(finite))


def _map_stats(name: str, values: np.ndarray, *, reference_maxabs: float | None = None) -> dict[str, Any]:
    finite = _finite_values(values)
    if finite.size == 0:
        return {
            "name": name,
            "min": np.nan,
            "max": np.nan,
            "mean": np.nan,
            "rms": np.nan,
            "maxabs": np.nan,
            "p95abs": np.nan,
            "p99abs": np.nan,
            "ratio_to_reference_maxabs": np.nan,
        }
    maxabs = float(np.max(np.abs(finite)))
    ratio = np.nan if reference_maxabs is None or reference_maxabs <= 0.0 else maxabs / float(reference_maxabs)
    return {
        "name": name,
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "mean": float(np.mean(finite)),
        "rms": float(np.sqrt(np.mean(finite ** 2))),
        "maxabs": maxabs,
        "p95abs": float(np.percentile(np.abs(finite), 95.0)),
        "p99abs": float(np.percentile(np.abs(finite), 99.0)),
        "ratio_to_reference_maxabs": ratio,
    }


def save_ta_map_csv(
    path: Path,
    *,
    delays_fs: np.ndarray,
    energy_eV: np.ndarray,
    ta_map: np.ndarray,
) -> Path:
    values = np.asarray(ta_map, dtype=float)
    delays = np.asarray(delays_fs, dtype=float)
    energy = np.asarray(energy_eV, dtype=float)
    if values.shape != (delays.size, energy.size):
        raise ValueError(f"ta_map shape {values.shape} is incompatible with delay/energy axes.")
    rows: list[dict[str, Any]] = []
    for delay_index, delay_fs in enumerate(delays):
        for energy_index, photon_energy_eV in enumerate(energy):
            rows.append(
                {
                    "delay_index": int(delay_index),
                    "delay_fs": float(delay_fs),
                    "energy_index": int(energy_index),
                    "energy_eV": float(photon_energy_eV),
                    "wavelength_nm": float(HC_EV_NM / photon_energy_eV),
                    "ta_signal": float(values[delay_index, energy_index]),
                }
            )
    return save_factorial_summary_csv(path, rows)


def save_all_delay_spectra_csv(
    path: Path,
    *,
    delays_fs: np.ndarray,
    energy_eV: np.ndarray,
    phase_stack: np.ndarray,
    phase_avg: np.ndarray,
    phase_labels: list[str],
) -> Path:
    delays = np.asarray(delays_fs, dtype=float)
    energy = np.asarray(energy_eV, dtype=float)
    phase_cases = np.asarray(phase_stack, dtype=float)
    avg = np.asarray(phase_avg, dtype=float)
    if phase_cases.shape != (len(phase_labels), delays.size, energy.size):
        raise ValueError(f"TA_phase_cases shape {phase_cases.shape} is incompatible with phase/delay/energy axes.")
    if avg.shape != (delays.size, energy.size):
        raise ValueError(f"TA_phase_avg shape {avg.shape} is incompatible with delay/energy axes.")
    rows: list[dict[str, Any]] = []
    for delay_index, delay_fs in enumerate(delays):
        for energy_index, photon_energy_eV in enumerate(energy):
            row = {
                "delay_index": int(delay_index),
                "delay_fs": float(delay_fs),
                "energy_index": int(energy_index),
                "energy_eV": float(photon_energy_eV),
                "wavelength_nm": float(HC_EV_NM / photon_energy_eV),
                "TA_phase_avg": float(avg[delay_index, energy_index]),
            }
            for phase_index, label in enumerate(phase_labels):
                row[f"TA_phase_{label}"] = float(phase_cases[phase_index, delay_index, energy_index])
            rows.append(row)
    return save_factorial_summary_csv(path, rows)


def save_legacy_shaped_outputs(
    output_dir: Path,
    *,
    settings: FactorialSettings,
    case_results: list[dict[str, Any]],
) -> dict[str, Any]:
    data_dir = output_dir / "data"
    case_names = [item["summary"]["case_name"] for item in case_results]
    if not case_names:
        raise ValueError("At least one case result is required.")
    if len(case_names) != 1:
        raise ValueError("Per-case legacy-shaped output expects exactly one case result.")
    energy = np.asarray(case_results[0]["energy_eV"], dtype=float)
    omega = np.asarray(case_results[0]["omega_fs_inv"], dtype=float)
    delays = np.asarray(case_results[0]["delays_fs"], dtype=float)
    ta_map = np.asarray(case_results[0]["ta_map"], dtype=float)
    phase_cases = np.asarray(case_results[0]["TA_phase_cases"], dtype=float)
    phase_labels = [_phase_label(value) for value in PHASE_VALUES_RAD]
    map_csv = save_ta_map_csv(data_dir / "ta_map.csv", delays_fs=delays, energy_eV=energy, ta_map=ta_map)
    all_delay_spectra_csv = save_all_delay_spectra_csv(
        data_dir / "ta_all_delay_spectra.csv",
        delays_fs=delays,
        energy_eV=energy,
        phase_stack=phase_cases,
        phase_avg=ta_map,
        phase_labels=phase_labels,
    )
    stats_rows = [_map_stats("TA_phase_avg_raw_full_energy", ta_map)]
    reference = stats_rows[0]["maxabs"]
    stats_rows.extend(
        _map_stats(f"TA_phase_{phase_label}_full_energy", phase_cases[index], reference_maxabs=reference)
        for index, phase_label in enumerate(phase_labels)
    )
    stats_csv = save_factorial_summary_csv(data_dir / "map_stats.csv", stats_rows)
    stats_json = _write_json(data_dir / "map_stats.json", {"map_stats": stats_rows})
    npz_path = data_dir / "ta_phase_cycling_comparison.npz"
    np.savez_compressed(
        npz_path,
        case_name=np.asarray(case_names[0], dtype=str),
        delays_fs=delays,
        energy_eV=energy,
        omega_fs_inv=omega,
        wavelength_nm=HC_EV_NM / energy,
        phase_values_rad=np.asarray(PHASE_VALUES_RAD, dtype=float),
        phase_labels=np.asarray(phase_labels, dtype=str),
        TA_phase_cases=phase_cases,
        TA_phase_avg=ta_map,
        ta_map=ta_map,
    )
    return {
        "case_names": case_names,
        "delays_fs": delays,
        "energy_eV": energy,
        "omega_fs_inv": omega,
        "ta_map": ta_map,
        "TA_phase_cases": phase_cases,
        "phase_labels": phase_labels,
        "map_csv": map_csv,
        "all_delay_spectra_csv": all_delay_spectra_csv,
        "stats_csv": stats_csv,
        "stats_json": stats_json,
        "data_npz": npz_path,
    }


def maybe_plot_ta_map(path: Path, *, delays_fs: np.ndarray, energy_eV: np.ndarray, ta_map: np.ndarray) -> Path | None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Skipping TA map plot because matplotlib import failed: {exc}")
        return None

    values, scale_used, raw_min, raw_max = _normalize_by_display_p99(ta_map, energy_eV)
    delays = np.asarray(delays_fs, dtype=float)
    fig, ax = plt.subplots(figsize=(6.6, 4.7))
    image = ax.imshow(
        values,
        aspect="auto",
        origin="lower",
        extent=[float(np.min(energy_eV)), float(np.max(energy_eV)), float(np.min(delays)), float(np.max(delays))],
        cmap="RdBu_r",
        vmin=-1.0,
        vmax=1.0,
    )
    ax.set_xlabel("Probe photon energy (eV)")
    ax.set_ylabel("Pump-probe delay (fs)")
    ax.set_title("TA map: delay x energy\nnormalized display, colorbar fixed to [-1, 1]")
    ax.set_xlim(*TA_MAP_XLIM_EV)
    ax.text(
        0.02,
        0.98,
        f"min={raw_min:.2e}\nmax={raw_max:.2e}\nscale={scale_used:.2e}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        bbox={"facecolor": "white", "alpha": 0.72, "edgecolor": "none", "pad": 2},
    )
    cbar = fig.colorbar(image, ax=ax, label="Normalized S_TA (displayed map / p99abs)")
    cbar.set_ticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _nearest_delay_index(delays_fs: np.ndarray, target_fs: float) -> int:
    return int(np.argmin(np.abs(np.asarray(delays_fs, dtype=float) - float(target_fs))))


def maybe_plot_selected_delay_lineouts(
    path: Path,
    *,
    delays_fs: np.ndarray,
    energy_eV: np.ndarray,
    ta_map: np.ndarray,
    selected_delays_fs: tuple[float, ...],
) -> Path | None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Skipping selected-delay lineouts because matplotlib import failed: {exc}")
        return None

    delays = np.asarray(delays_fs, dtype=float)
    values = np.asarray(ta_map, dtype=float)
    fig, axes = plt.subplots(len(selected_delays_fs), 1, figsize=(7.4, 2.4 * len(selected_delays_fs)), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, target_delay in zip(axes, selected_delays_fs):
        idx = _nearest_delay_index(delays, target_delay)
        actual_delay = float(delays[idx])
        y_raw = values[idx]
        y, scale_used, raw_min, raw_max = _normalize_by_display_p99(y_raw, energy_eV)
        ax.plot(energy_eV, y, linewidth=1.6, color="black")
        ax.axhline(0.0, linewidth=0.8, linestyle="--", color="black", alpha=0.5)
        ax.set_xlim(*TA_MAP_XLIM_EV)
        ax.set_ylim(-1.08, 1.08)
        ax.set_ylabel("Normalized S_TA")
        ax.set_title(f"delay = {actual_delay:g} fs")
        ax.text(
            0.02,
            0.96,
            f"min={raw_min:.2e}\nmax={raw_max:.2e}\nscale={scale_used:.2e}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            bbox={"facecolor": "white", "alpha": 0.72, "edgecolor": "none", "pad": 2},
        )
    axes[-1].set_xlabel("Probe photon energy (eV)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def maybe_plot_biased_overlay_lineout(
    path: Path,
    *,
    delay_fs: float,
    delay_index: int,
    energy_eV: np.ndarray,
    phase_cases: np.ndarray,
    phase_labels: list[str],
) -> Path | None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Skipping biased overlay lineout because matplotlib import failed: {exc}")
        return None

    fig, ax_avg = plt.subplots(figsize=(8.6, 5.4))
    ax_phase = ax_avg.twinx()
    spectra = np.asarray(phase_cases, dtype=float)[:, int(delay_index), :]
    avg_spectrum = np.mean(spectra, axis=0)
    styles = {"0": "-", "pi2": "--", "pi": "-.", "3pi2": ":"}
    for phase_label, spectrum in zip(phase_labels, spectra):
        ax_phase.plot(
            energy_eV,
            spectrum,
            linestyle=styles.get(phase_label, "-"),
            linewidth=1.2,
            color="red",
            alpha=0.75,
            label=f"phase {phase_label}",
        )
    ax_avg.plot(energy_eV, avg_spectrum, linewidth=2.2, color="black", label="phase average")
    ax_avg.axhline(0.0, linewidth=0.8, linestyle="--", color="black", alpha=0.5)
    ax_phase.axhline(0.0, linewidth=0.8, linestyle="--", color="red", alpha=0.35)
    ax_avg.set_title(f"Biased overlay TA lineout at delay = {delay_fs:g} fs")
    ax_avg.set_xlabel("Probe photon energy (eV)")
    ax_avg.set_ylabel("Phase-averaged S_TA", color="black")
    ax_phase.set_ylabel("Single-phase S_TA", color="red")
    ax_avg.set_xlim(*TA_MAP_XLIM_EV)
    avg_finite = avg_spectrum[np.isfinite(avg_spectrum)]
    phase_finite = spectra[np.isfinite(spectra)]
    if avg_finite.size:
        avg_abs = float(np.max(np.abs(avg_finite)))
        if avg_abs > 0.0:
            ax_avg.set_ylim(-1.08 * avg_abs, 1.08 * avg_abs)
    if phase_finite.size:
        phase_abs = float(np.max(np.abs(phase_finite)))
        if phase_abs > 0.0:
            ax_phase.set_ylim(-1.08 * phase_abs, 1.08 * phase_abs)
    if avg_finite.size and phase_finite.size:
        ax_avg.text(
            0.02,
            0.98,
            (
                f"avg min={np.min(avg_finite):.2e}\n"
                f"avg max={np.max(avg_finite):.2e}\n"
                f"phase min={np.min(phase_finite):.2e}\n"
                f"phase max={np.max(phase_finite):.2e}"
            ),
            transform=ax_avg.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            bbox={"facecolor": "white", "alpha": 0.72, "edgecolor": "none", "pad": 2},
        )
    lines_avg, labels_avg = ax_avg.get_legend_handles_labels()
    lines_phase, labels_phase = ax_phase.get_legend_handles_labels()
    ax_avg.legend(lines_avg + lines_phase, labels_avg + labels_phase, fontsize=8, loc="lower right")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _settings_from_args(args: argparse.Namespace) -> FactorialSettings:
    if bool(args.quick):
        return FactorialSettings(
            case_mode=str(args.case_mode),
            eis_on_eV=float(args.eis_eV),
            pb_on=float(args.pb),
            eid_on=float(args.eid),
            pb_scan=tuple(float(value) for value in args.pb_scan),
            eid_scan=tuple(float(value) for value in args.eid_scan),
            eis_scan_eV=tuple(float(value) for value in args.eis_scan_eV),
            t_start_fs=-180.0,
            t_end_fs=220.0,
            dt_fs=1.0,
            zero_padding_factor=2,
        )
    return FactorialSettings(
        case_mode=str(args.case_mode),
        eis_on_eV=float(args.eis_eV),
        pb_on=float(args.pb),
        eid_on=float(args.eid),
        pb_scan=tuple(float(value) for value in args.pb_scan),
        eid_scan=tuple(float(value) for value in args.eid_scan),
        eis_scan_eV=tuple(float(value) for value in args.eis_scan_eV),
    )


def run_factorial(args: argparse.Namespace) -> dict[str, Any]:
    settings = _settings_from_args(args)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    normalizer = ParaNormalizer()
    delays_fs = _selected_delays(settings, quick=bool(args.quick))

    case_results = []
    case_entries = []
    force_rerun = bool(getattr(args, "force_rerun", False) or getattr(args, "force", False))
    for case in build_case_config(settings):
        case_dir = output_dir / _case_output_dir_name(case)
        saved_npz = _saved_npz_path(case_dir)
        loaded_from_saved_data = bool(
            _saved_case_data_is_compatible(case_dir, expected_delays_fs=delays_fs) and not force_rerun
        )
        if loaded_from_saved_data:
            print(f"[factorial] using saved data for {case.case_name}: {saved_npz}")
            case_result, legacy_outputs = _load_saved_case_result(case, case_dir)
        else:
            case_result = run_one_case(
                case,
                settings=settings,
                delays_fs=delays_fs,
                output_dir=case_dir,
                normalizer=normalizer,
            )
            legacy_outputs = save_legacy_shaped_outputs(case_dir, settings=settings, case_results=[case_result])
        figure_paths: dict[str, Path | None] = {}
        ta_map_path = None
        if not bool(args.no_plots):
            ta_map_path = maybe_plot_ta_map(
                case_dir / "figures" / "plot" / "ta_map.png",
                delays_fs=legacy_outputs["delays_fs"],
                energy_eV=legacy_outputs["energy_eV"],
                ta_map=legacy_outputs["ta_map"],
            )
            figure_paths["ta_map"] = ta_map_path
            figure_paths["selected_delay_lineouts"] = maybe_plot_selected_delay_lineouts(
                case_dir / "figures" / "preview" / "selected_delay_lineouts.png",
                delays_fs=legacy_outputs["delays_fs"],
                energy_eV=legacy_outputs["energy_eV"],
                ta_map=legacy_outputs["ta_map"],
                selected_delays_fs=settings.selected_lineout_delays_fs,
            )
            for target_delay in settings.selected_lineout_delays_fs:
                delay_index = _nearest_delay_index(legacy_outputs["delays_fs"], target_delay)
                actual_delay = float(legacy_outputs["delays_fs"][delay_index])
                figure_paths[f"biased_overlay_lineout_{_delay_label(actual_delay)}_fs"] = maybe_plot_biased_overlay_lineout(
                    case_dir / "figures" / "preview" / f"biased_overlay_lineout_{_delay_label(actual_delay)}_fs.png",
                    delay_fs=actual_delay,
                    delay_index=delay_index,
                    energy_eV=legacy_outputs["energy_eV"],
                    phase_cases=legacy_outputs["TA_phase_cases"],
                    phase_labels=legacy_outputs["phase_labels"],
                )
        case_meta = {
            "example_name": "ta_harmonic_exciton_ladder_factorial_v2",
            "script": str(Path(__file__).resolve()),
            "output_dir": case_dir,
            "quick": bool(args.quick),
            "settings": asdict(settings),
            "phase_values_rad": list(PHASE_VALUES_RAD),
            "target_phase_vector": dict(TARGET_PHASE_VECTOR),
            "workflow": {
                "compute_path": "TA recipe v2 generic pulse-sequence + pump phase cases",
                "output_shape": "legacy TA phase-cycling demo output layout for one factorial case",
                "loaded_from_saved_data": loaded_from_saved_data,
            },
            "spectroscopy": {
                "definition": "absorption = omega * Im[P(omega)/E_probe(omega)]",
                "TA_definition": "S_TA = S_pump_probe - S_probe_only",
                "ta_map_axes": ["delay_fs", "energy_eV"],
                "delays_fs": delays_fs,
                "number_density_m3": settings.number_density_m3,
                "zero_padding_factor": settings.zero_padding_factor,
                "rel_threshold": settings.rel_threshold,
                "ta_map_xlim_eV": TA_MAP_XLIM_EV,
            },
            "data_npz": legacy_outputs["data_npz"],
            "ta_map_csv": legacy_outputs["map_csv"],
            "all_delay_spectra_csv": legacy_outputs["all_delay_spectra_csv"],
            "stats_csv": legacy_outputs["stats_csv"],
            "stats_json": legacy_outputs["stats_json"],
            "figures": figure_paths,
            "case": {
                "summary": case_result["summary"],
                "system": case_result["system"],
                "base_params_adapter": case_result["base_params_adapter"],
            },
        }
        case_meta_path = _write_json(case_dir / "meta.json", case_meta)
        case_result["summary"]["output_dir"] = str(case_dir)
        case_result["summary"]["meta_json"] = str(case_meta_path)
        case_result["summary"]["data_npz"] = str(legacy_outputs["data_npz"])
        case_result["summary"]["ta_map_csv"] = str(legacy_outputs["map_csv"])
        case_result["summary"]["all_delay_spectra_csv"] = str(legacy_outputs["all_delay_spectra_csv"])
        case_results.append(case_result)
        case_entries.append(
            {
                "case_name": case.case_name,
                "output_dir": case_dir,
                "loaded_from_saved_data": loaded_from_saved_data,
                "meta_json": case_meta_path,
                "data_npz": legacy_outputs["data_npz"],
                "ta_map_csv": legacy_outputs["map_csv"],
                "all_delay_spectra_csv": legacy_outputs["all_delay_spectra_csv"],
                "stats_csv": legacy_outputs["stats_csv"],
                "figures": figure_paths,
            }
        )

    summary_rows = _summary_csv_rows(case_results)
    summary_csv = save_factorial_summary_csv(output_dir / "data" / "factorial_summary.csv", summary_rows)
    meta = {
        "example_name": "ta_harmonic_exciton_ladder_factorial_v2",
        "script": str(Path(__file__).resolve()),
        "output_dir": output_dir,
        "quick": bool(args.quick),
        "settings": asdict(settings),
        "delays_fs": delays_fs,
        "phase_values_rad": list(PHASE_VALUES_RAD),
        "target_phase_vector": dict(TARGET_PHASE_VECTOR),
        "workflow": {
            "compute_path": "TA recipe v2 generic pulse-sequence + pump phase cases",
            "output_shape": "one legacy-shaped output directory per factorial case",
        },
        "spectroscopy": {
            "definition": "absorption = omega * Im[P(omega)/E_probe(omega)]",
            "TA_definition": "S_TA = S_pump_probe - S_probe_only",
            "number_density_m3": settings.number_density_m3,
            "zero_padding_factor": settings.zero_padding_factor,
            "rel_threshold": settings.rel_threshold,
            "ta_map_xlim_eV": TA_MAP_XLIM_EV,
        },
        "summary_csv": summary_csv,
        "cases": case_entries,
    }
    meta_path = _write_json(output_dir / "meta.json", meta)

    print("harmonic exciton ladder factorial v2 finished")
    print(f"output_dir: {output_dir}")
    print(f"summary_csv: {summary_csv}")
    print(f"meta_json: {meta_path}")
    for row in summary_rows:
        print(
            f"{row['case_name']:16s} "
            f"max_abs={row['max_abs_ta_signal']:.6e} "
            f"integrated_abs={row['integrated_abs_ta_signal']:.6e} "
            f"trace={_format_optional_sci(row['max_trace_error'])} "
            f"herm={_format_optional_sci(row['max_hermiticity_error'])} "
            f"dir={row['output_dir']}"
        )
    return meta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Use short time grid and one selected delay.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory.")
    parser.add_argument("--force-rerun", action="store_true", help="Ignore saved per-case NPZ data and rerun simulations.")
    parser.add_argument("--force", action="store_true", help="Alias for --force-rerun.")
    parser.add_argument("--no-plots", action="store_true", help="Skip per-case PNG plots.")
    parser.add_argument("--case-mode", choices=("one_dim_scan", "factorial"), default="one_dim_scan")
    parser.add_argument("--eis-eV", type=float, default=-0.02, help="EIS on-value for --case-mode factorial.")
    parser.add_argument("--pb", type=float, default=0.99, help="PB on-value for --case-mode factorial.")
    parser.add_argument("--eid", type=float, default=1.1, help="EID on-value for --case-mode factorial.")
    parser.add_argument("--pb-scan", type=float, nargs="+", default=[0.7, 0.8, 0.9, 0.95, 0.99])
    parser.add_argument("--eid-scan", type=float, nargs="+", default=[1.1, 1.3, 1.5])
    parser.add_argument("--eis-scan-eV", type=float, nargs="+", default=[0.01, 0.02, 0.05, 0.1])
    return parser.parse_args()


def main() -> None:
    run_factorial(parse_args())


if __name__ == "__main__":
    main()
