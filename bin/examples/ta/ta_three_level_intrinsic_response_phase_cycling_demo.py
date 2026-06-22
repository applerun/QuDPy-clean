#!/usr/bin/env python3
"""Three-level TA intrinsic response: phase-cycling comparison demo.

This script is example/scratch-level only. It does not modify the core solver,
DynamicsResult, or IO layer, and it does not use piecewise/dark propagation.

It compares three phase / delay conventions:

1. delay changes carrier phase
   The pulse is shifted as a full lab-frame electric field. For a zero-centered
   Gaussian template this is equivalent to

       E(t) = envelope(t - center) * cos(omega * (t - center) + phase0)

   In a delay scan, the pump carrier phase relative to the fixed probe changes
   with delay. This can produce optical-phase ripples in a direct lab-frame
   P(omega)/E_probe(omega) TA map.

2. 4-step pump-phase average
   Four pump carrier phases are used with a fixed probe phase:

       pump_phase = 0, pi/2, pi, 3pi/2
       probe_phase = 0

   The physical phase average is an unweighted arithmetic mean. This is a simple
   phase-averaging control, not a weighted nonlinear-response phase projection.

3. fixed lab-frame carrier phase / v3-like envelope shift
   The Gaussian envelope center is moved while the carrier remains cos(omega*t+phase0):

       E(t) = envelope(t - center) * cos(omega * t + phase0)

   This reproduces the convention used by GaussianCarrierFieldPhysical and
   make_ta_gaussian_field(...), and usually resembles the smoother v3 TA map.

Important:
    Do not normalize phase cases before computing the physical phase average.
    This script additionally saves normalized-display figures only for visual inspection.

Outputs
-------
Under bin/examples/ta/outputs/ta_three_level_intrinsic_response_phase_cycling_demo/:

    data/ta_phase_cycling_comparison.npz
    data/map_stats.csv
    data/map_stats.json
    meta.json

    figures/ta_phase_case_0.png
    figures/ta_phase_case_pi2.png
    figures/ta_phase_case_pi.png
    figures/ta_phase_case_3pi2.png
    figures/ta_phase_avg.png
    figures/ta_phase_avg_autoscale.png
    figures/ta_phase_avg_unitnorm_diagnostic.png
    figures/ta_fixed_carrier.png
    figures/ta_fixed_carrier_autoscale.png
    figures/ta_phase_cycling_compare.png
    figures/ta_phase_cycling_compare_autoscale.png

Sign convention
---------------
The spectrum uses the current spectroscopy helper definition:

    absorption = omega * Im[P(omega) / E_probe(omega)]

If the plotted signs do not match the usual GSB/SE/ESA intuition, do not change
the sign here; interpret the figure using this convention.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import argparse
import csv
import json
import math
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

if __package__ is None or __package__ == "":
    # Intended path:
    #   QuDPy-clean/bin/examples/ta/ta_three_level_intrinsic_response_phase_cycling_demo.py
    # so parents[3] is the repository root.
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from qudpy_sjh.utils.core import (
    NLevelPhysicalParams,
    ParaNormalizer,
    PureDephasingChannel,
    RelaxationChannel,
    run_case,
)
from qudpy_sjh.utils.fields import FieldPhySeries
from qudpy_sjh.utils.fields.specific.basic_fields import (
    GaussianCarrierFieldPhysical,
    make_default_gaussian_carrier_field,
)
from qudpy_sjh.utils.spectroscopy import (
    lab_frame_absorption_response,
    polarization_C_per_m2,
)


EXAMPLE_NAME = "ta_three_level_intrinsic_response_phase_cycling_demo"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / EXAMPLE_NAME
HC_EV_NM = 1239.8419843320026


@dataclass(frozen=True)
class DemoConfig:
    example_name: str = EXAMPLE_NAME

    probe_delays_fs: tuple[float, ...] = (
        -300.0, -220.0, -160.0, -110.0, -80.0, -60.0, -45.0, -30.0,
        -20.0, -10.0, 0.0, 10.0, 20.0, 30.0, 45.0, 60.0, 80.0,
        110.0, 150.0, 220.0, 320.0, 460.0, 650.0, 900.0, 1200.0,
    )
    quick_probe_delays_fs: tuple[float, ...] = (
        -220.0, -80.0, -30.0, 0.0, 30.0, 80.0, 220.0, 650.0, 1200.0,
    )

    probe_center_fs: float = 0.0

    # Three-level ladder:
    # g<->e near 1.55 eV, e<->f near 1.70 eV.
    basis: tuple[str, ...] = ("g", "e", "f")
    energies_eV: tuple[float, ...] = (0.0, 1.55, 3.25)
    dipole_matrix_D: tuple[tuple[float, ...], ...] = (
        (0.0, 5.0, 0.0),
        (5.0, 0.0, 9.0),
        (0.0, 9.0, 0.0),
    )

    # Slightly stronger pump and longer lifetimes make the phase-invariant
    # population/ESA contribution easier to see after phase averaging.
    pump_E0_MV_per_cm: float = 0.30
    probe_E0_MV_per_cm: float = 0.008
    pump_laser_energy_eV: float = 1.55
    probe_laser_energy_eV: float = 1.62
    pump_sigma_fs: float = 12.0
    probe_sigma_fs: float = 7.0
    probe_phase_rad: float = 0.0
    pump_phase_cases_rad: tuple[float, ...] = (
        0.0,
        0.5 * math.pi,
        math.pi,
        1.5 * math.pi,
    )

    T1_2_to_1_fs: float = 500.0
    T1_1_to_0_fs: float = 1200.0
    Tphi_1_fs: float = 120.0
    Tphi_2_fs: float = 100.0

    # Fixed full-window propagation for all delays.
    t_start_fs: float = -1500.0
    t_end_fs: float = 450.0
    dt_fs: float = 0.2

    number_density_m3: float = 1.0e24
    window: str | None = "hann"
    subtract_mean: bool = True
    rel_threshold: float = 1.0e-6
    zero_padding_factor: int = 4

    plot_energy_range_eV: tuple[float, float] = (1.35, 1.90)
    plot_use_wavelength: bool = False
    cmap: str = "plasma"
    figure_dpi: int = 180

    use_checkpoints: bool = True
    force_run: bool = False


def _json_safe(value: Any) -> Any:
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _json_safe(value.to_dict())
    if hasattr(value, "__dataclass_fields__"):
        return _json_safe(asdict(value))
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, complex):
        return {"real": float(value.real), "imag": float(value.imag)}
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def write_csv_rows(path: Path, rows: list[dict[str, Any]]) -> Path:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


def safe_delay_label(delay_fs: float) -> str:
    value = 0.0 if abs(float(delay_fs)) < 1e-12 else float(delay_fs)
    text = f"{value:.6f}".rstrip("0").rstrip(".")
    if text == "-0":
        text = "0"
    return text.replace("-", "m").replace(".", "p")


def phase_label(phase: float) -> str:
    phase = float(phase) % (2.0 * np.pi)
    if np.isclose(phase, 0.0):
        return "0"
    if np.isclose(phase, 0.5 * np.pi):
        return "pi2"
    if np.isclose(phase, np.pi):
        return "pi"
    if np.isclose(phase, 1.5 * np.pi):
        return "3pi2"
    return f"{phase:.4f}".rstrip("0").rstrip(".").replace(".", "p")


def make_gaussian_lab_phase_locked(
    *,
    E0_MV_per_cm: float,
    laser_energy_eV: float,
    center_fs: float,
    sigma_fs: float,
    phase_rad: float,
    name: str,
    metadata: dict[str, Any] | None = None,
) -> GaussianCarrierFieldPhysical:
    """Move envelope center but keep lab-frame carrier phase cos(omega*t+phase).

    This is the v3-like convention.
    """

    return make_default_gaussian_carrier_field(
        E0_MV_per_cm=float(E0_MV_per_cm),
        laser_energy_eV=float(laser_energy_eV),
        pulse_center_fs=float(center_fs),
        pulse_sigma_fs=float(sigma_fs),
        phase_rad=float(phase_rad),
        name=name,
        metadata={
            **dict(metadata or {}),
            "shift_policy": "lab_phase_locked_envelope_center",
            "field_expression": "E(t)=2E0 exp[-(t-center)^2/(2sigma^2)] cos(omega*t+phase)",
        },
    )


def make_gaussian_cep_locked(
    *,
    E0_MV_per_cm: float,
    laser_energy_eV: float,
    center_fs: float,
    sigma_fs: float,
    phase_rad: float,
    name: str,
    metadata: dict[str, Any] | None = None,
) -> GaussianCarrierFieldPhysical:
    """Move the full pulse field, keeping carrier-envelope phase fixed.

    Desired expression:

        E(t) = envelope(t-center) * cos(omega*(t-center) + phase_rad)

    GaussianCarrierFieldPhysical implements:

        E(t) = envelope(t-center) * cos(omega*t + phase_lab)

    Therefore:

        phase_lab = phase_rad - omega*center
    """

    template = make_default_gaussian_carrier_field(
        E0_MV_per_cm=float(E0_MV_per_cm),
        laser_energy_eV=float(laser_energy_eV),
        pulse_center_fs=float(center_fs),
        pulse_sigma_fs=float(sigma_fs),
        phase_rad=0.0,
        name=name,
        metadata={},
    )
    omega = float(template.omega_L_fs_inv)
    phase_lab = float(phase_rad) - omega * float(center_fs)

    return GaussianCarrierFieldPhysical(
        E0_MV_per_cm=float(E0_MV_per_cm),
        omega_L_fs_inv=omega,
        center_fs=float(center_fs),
        sigma_fs=float(sigma_fs),
        phase_rad=float(phase_lab),
        name=name,
        metadata={
            "laser_energy_eV": float(laser_energy_eV),
            **dict(metadata or {}),
            "phase_rad_input": float(phase_rad),
            "phase_rad_lab": float(phase_lab),
            "shift_policy": "cep_locked_full_field_time_shift",
            "field_expression": "E(t)=2E0 exp[-(t-center)^2/(2sigma^2)] cos(omega*(t-center)+phase0)",
        },
    )


def make_probe_reference_field(config: DemoConfig) -> GaussianCarrierFieldPhysical:
    return make_gaussian_lab_phase_locked(
        E0_MV_per_cm=config.probe_E0_MV_per_cm,
        laser_energy_eV=config.probe_laser_energy_eV,
        center_fs=config.probe_center_fs,
        sigma_fs=config.probe_sigma_fs,
        phase_rad=config.probe_phase_rad,
        name="probe",
        metadata={"role": "probe_only_reference"},
    )


def make_ta_field(
    config: DemoConfig,
    *,
    delay_fs: float,
    pump_phase_rad: float,
    field_policy: str,
    name: str,
) -> FieldPhySeries:
    delay = float(delay_fs)
    probe_center = float(config.probe_center_fs)
    pump_center = probe_center - delay

    if field_policy == "cep_locked_delay_changes_carrier_phase":
        factory = make_gaussian_cep_locked
    elif field_policy == "lab_phase_locked_envelope_only":
        factory = make_gaussian_lab_phase_locked
    else:
        raise ValueError(f"Unsupported field_policy: {field_policy!r}")

    pump = factory(
        E0_MV_per_cm=config.pump_E0_MV_per_cm,
        laser_energy_eV=config.pump_laser_energy_eV,
        center_fs=pump_center,
        sigma_fs=config.pump_sigma_fs,
        phase_rad=float(pump_phase_rad),
        name="pump",
        metadata={
            "role": "pump",
            "delay_fs": delay,
            "center_fs": pump_center,
            "parent_field": name,
        },
    )
    probe = factory(
        E0_MV_per_cm=config.probe_E0_MV_per_cm,
        laser_energy_eV=config.probe_laser_energy_eV,
        center_fs=probe_center,
        sigma_fs=config.probe_sigma_fs,
        phase_rad=float(config.probe_phase_rad),
        name="probe",
        metadata={
            "role": "probe",
            "delay_fs": delay,
            "center_fs": probe_center,
            "parent_field": name,
        },
    )

    return FieldPhySeries(
        fields=(pump, probe),
        sub_field_names=("pump", "probe"),
        name=name,
        metadata={
            "experiment": "TA",
            "delay_fs": delay,
            "pump_phase_rad": float(pump_phase_rad),
            "probe_phase_rad": float(config.probe_phase_rad),
            "field_policy": field_policy,
            "pump_center_fs": pump_center,
            "probe_center_fs": probe_center,
            "delay_convention": "pump_center_fs = probe_center_fs - delay_fs; positive delay means pump before probe",
        },
    )


def make_physical_params(
    config: DemoConfig,
    field,
    *,
    case_name: str,
    description: str,
) -> NLevelPhysicalParams:
    return NLevelPhysicalParams(
        energies_eV=tuple(float(x) for x in config.energies_eV),
        dipole_matrix_D=tuple(tuple(float(v) for v in row) for row in config.dipole_matrix_D),
        t_start_fs=float(config.t_start_fs),
        t_end_fs=float(config.t_end_fs),
        dt_fs=float(config.dt_fs),
        field=field,
        basis=tuple(config.basis),
        relaxation_channels=(
            RelaxationChannel(
                name="relaxation_2_to_1",
                from_level=2,
                to_level=1,
                T1_fs=float(config.T1_2_to_1_fs),
            ),
            RelaxationChannel(
                name="relaxation_1_to_0",
                from_level=1,
                to_level=0,
                T1_fs=float(config.T1_1_to_0_fs),
            ),
        ),
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
        input_description=description,
        input_metadata={
            "example_name": config.example_name,
            "case_name": case_name,
            "response_definition": "S_TA = omega*Im[P_pump_probe/E_probe] - omega*Im[P_probe_only/E_probe]",
            "sign_convention": "absorption = omega * Im[P(omega)/E_probe(omega)]",
            "model_note": "Three-level ladder demo; parameters are diagnostic rather than fitted.",
        },
    )


def run_with_checkpoint(
    params: NLevelPhysicalParams,
    *,
    normalizer: ParaNormalizer,
    output_dir: Path,
    case_key: str,
    config: DemoConfig,
):
    if not config.use_checkpoints:
        return run_case(params, normalizer=normalizer)

    ckp = output_dir / "checkpoints" / f"{case_key}.ckp"
    return run_case(
        params,
        normalizer=normalizer,
        load_ckp=ckp,
        save_ckp=ckp,
        force_run=bool(config.force_run),
    )


def response_from_result(result, probe_field, config: DemoConfig) -> dict[str, np.ndarray]:
    physical = result.physical_params
    if physical is None:
        raise ValueError("DynamicsResult.physical_params is required.")

    t_fs = np.asarray(result.times_fs, dtype=float)
    e_probe = np.asarray(probe_field(t_fs), dtype=float)
    p_t = polarization_C_per_m2(
        result.density_array(),
        physical.dipole_matrix_D,
        float(config.number_density_m3),
    )

    return lab_frame_absorption_response(
        time_fs=t_fs,
        polarization_C_per_m2=p_t,
        field=e_probe,
        window=config.window,
        subtract_mean=bool(config.subtract_mean),
        rel_threshold=float(config.rel_threshold),
        zero_padding_factor=int(config.zero_padding_factor),
        return_intermediates=True,
    )


def assert_same_axis(name: str, reference: np.ndarray, current: np.ndarray) -> None:
    if reference.shape != current.shape:
        raise ValueError(f"{name} axis shape mismatch: {reference.shape} vs {current.shape}")
    diff = float(np.max(np.abs(reference - current))) if reference.size else 0.0
    if diff > 1e-12:
        raise ValueError(f"{name} axis mismatch. max_abs_diff={diff:.6e}")


def compute_map(
    config: DemoConfig,
    *,
    delays_fs: tuple[float, ...],
    output_dir: Path,
    normalizer: ParaNormalizer,
    probe_field,
    probe_response: dict[str, np.ndarray],
    pump_phase_rad: float,
    field_policy: str,
    map_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    energy_ref = np.asarray(probe_response["energy_eV"], dtype=float)
    omega_ref = np.asarray(probe_response["omega_fs_inv"], dtype=float)
    s_probe = np.asarray(probe_response["absorption"], dtype=float)

    rows = []
    for delay_fs in delays_fs:
        print(f"[{map_name}] delay={delay_fs:g} fs, pump_phase={pump_phase_rad:.6g}")
        field = make_ta_field(
            config,
            delay_fs=float(delay_fs),
            pump_phase_rad=float(pump_phase_rad),
            field_policy=field_policy,
            name=f"{map_name}_delay_{safe_delay_label(delay_fs)}_fs",
        )
        params = make_physical_params(
            config,
            field,
            case_name=f"{map_name}_delay_{safe_delay_label(delay_fs)}_fs",
            description=f"TA pump+probe case for {map_name}, delay={delay_fs:g} fs.",
        )
        result = run_with_checkpoint(
            params,
            normalizer=normalizer,
            output_dir=output_dir,
            case_key=f"{map_name}_delay_{safe_delay_label(delay_fs)}_fs",
            config=config,
        )
        response = response_from_result(result, probe_field, config)

        energy = np.asarray(response["energy_eV"], dtype=float)
        omega = np.asarray(response["omega_fs_inv"], dtype=float)
        assert_same_axis("energy_eV", energy_ref, energy)
        assert_same_axis("omega_fs_inv", omega_ref, omega)

        s_pump_probe = np.asarray(response["absorption"], dtype=float)
        rows.append(s_pump_probe - s_probe)

    return energy_ref, omega_ref, np.vstack(rows)


def finite_values(arr: np.ndarray) -> np.ndarray:
    values = np.asarray(arr, dtype=float).ravel()
    return values[np.isfinite(values)]


def map_stats(name: str, arr: np.ndarray, *, reference_maxabs: float | None = None) -> dict[str, Any]:
    values = finite_values(arr)
    if values.size == 0:
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

    maxabs = float(np.max(np.abs(values)))
    ratio = np.nan
    if reference_maxabs is not None and reference_maxabs > 0:
        ratio = maxabs / float(reference_maxabs)

    return {
        "name": name,
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "rms": float(np.sqrt(np.mean(values**2))),
        "maxabs": maxabs,
        "p95abs": float(np.percentile(np.abs(values), 95.0)),
        "p99abs": float(np.percentile(np.abs(values), 99.0)),
        "ratio_to_reference_maxabs": ratio,
    }


def robust_vlim(arrays: list[np.ndarray], percentile: float = 99.0) -> float:
    merged = np.concatenate([finite_values(item) for item in arrays])
    merged = merged[np.isfinite(merged)]
    if merged.size == 0:
        return 1.0
    value = float(np.nanpercentile(np.abs(merged), percentile))
    return value if value > 0 else 1.0


def normalize_map_for_diagnostic(arr: np.ndarray, *, scale: str = "p99abs") -> np.ndarray:
    values = finite_values(arr)
    if values.size == 0:
        return np.asarray(arr, dtype=float)

    if scale == "maxabs":
        denom = float(np.max(np.abs(values)))
    elif scale == "p99abs":
        denom = float(np.percentile(np.abs(values), 99.0))
    elif scale == "rms":
        denom = float(np.sqrt(np.mean(values**2)))
    else:
        raise ValueError(f"Unknown diagnostic normalization scale: {scale!r}")

    if denom <= 0 or not np.isfinite(denom):
        return np.asarray(arr, dtype=float)

    return np.asarray(arr, dtype=float) / denom


def normalize_for_panel_display(
    values: np.ndarray,
    *,
    scale_mode: str = "p99abs",
) -> tuple[np.ndarray, float, float, float]:
    """Normalize one displayed map to roughly [-1, 1].

    Returns
    -------
    normalized_values
        values / scale_used
    scale_used
        The p99abs, maxabs, or rms scale used for display normalization.
    raw_min
        Minimum of the unnormalized displayed values.
    raw_max
        Maximum of the unnormalized displayed values.
    """

    raw = np.asarray(values, dtype=float)
    finite = raw[np.isfinite(raw)]

    if finite.size == 0:
        return raw, 1.0, np.nan, np.nan

    raw_min = float(np.min(finite))
    raw_max = float(np.max(finite))

    if scale_mode == "maxabs":
        scale = float(np.max(np.abs(finite)))
    elif scale_mode == "p99abs":
        scale = float(np.percentile(np.abs(finite), 99.0))
    elif scale_mode == "rms":
        scale = float(np.sqrt(np.mean(finite**2)))
    else:
        raise ValueError(f"Unknown scale_mode: {scale_mode!r}")

    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0

    normalized = raw / scale
    return normalized, scale, raw_min, raw_max


def to_plot_x(energy_eV: np.ndarray, *, use_wavelength: bool) -> tuple[np.ndarray, str]:
    if use_wavelength:
        return HC_EV_NM / energy_eV, "Probe wavelength (nm)"
    return energy_eV, "Probe photon energy (eV)"


def apply_plot_energy_mask(
    x: np.ndarray,
    values: np.ndarray,
    config: DemoConfig,
) -> tuple[np.ndarray, np.ndarray]:
    if config.plot_energy_range_eV is None:
        return x, values

    e_min, e_max = config.plot_energy_range_eV

    if config.plot_use_wavelength:
        x_min = HC_EV_NM / e_max
        x_max = HC_EV_NM / e_min
        mask = (x >= min(x_min, x_max)) & (x <= max(x_min, x_max))
    else:
        mask = (x >= e_min) & (x <= e_max)

    if not np.any(mask):
        return x, values

    return x[mask], values[:, mask]


def prepare_plot_arrays(
    energy_eV: np.ndarray,
    values: np.ndarray,
    config: DemoConfig,
) -> tuple[np.ndarray, np.ndarray, str]:
    x, xlabel = to_plot_x(energy_eV, use_wavelength=config.plot_use_wavelength)
    plot_values = np.asarray(values, dtype=float)

    if config.plot_use_wavelength:
        order = np.argsort(x)
        x = x[order]
        plot_values = plot_values[:, order]

    x, plot_values = apply_plot_energy_mask(x, plot_values, config)
    return x, plot_values, xlabel


def displayed_energy_map_values(
    energy_eV: np.ndarray,
    values: np.ndarray,
    config: DemoConfig,
) -> np.ndarray:
    _x, plot_values, _xlabel = prepare_plot_arrays(energy_eV, values, config)
    return plot_values


def plot_one_map(
    *,
    path: Path,
    title: str,
    energy_eV: np.ndarray,
    delays_fs: np.ndarray,
    values: np.ndarray,
    config: DemoConfig,
    vlim: float | None,
) -> Path:
    """Plot one TA map.

    If vlim is None, normalize the displayed energy range by its own p99abs,
    use a fixed color scale [-1, 1], and annotate raw min/max/scale.
    """

    fig, ax = plt.subplots(figsize=(6.6, 4.7))
    x, plot_values, xlabel = prepare_plot_arrays(energy_eV, values, config)

    if vlim is None:
        plot_values, scale_used, raw_min, raw_max = normalize_for_panel_display(
            plot_values,
            scale_mode="p99abs",
        )
        local_vmin = -1.0
        local_vmax = 1.0
        cbar_label = "Normalized S_TA (displayed map / p99abs)"
        ax.text(
            0.02,
            0.98,
            (
                f"min={raw_min:.2e}\n"
                f"max={raw_max:.2e}\n"
                f"scale={scale_used:.2e}"
            ),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            bbox={
                "facecolor": "white",
                "alpha": 0.72,
                "edgecolor": "none",
                "pad": 2,
            },
        )
        title = f"{title}\nnormalized display, colorbar fixed to [-1, 1]"
    else:
        local_vmin = -float(vlim)
        local_vmax = float(vlim)
        cbar_label = "S_TA (arb., current sign)"

    mesh = ax.pcolormesh(
        x,
        delays_fs,
        plot_values,
        shading="auto",
        cmap=config.cmap,
        vmin=local_vmin,
        vmax=local_vmax,
    )

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Pump-probe delay (fs)")

    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label(cbar_label)
    if vlim is None:
        cbar.set_ticks([-1.0, -0.5, 0.0, 0.5, 1.0])

    fig.tight_layout()

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=int(config.figure_dpi))
    plt.close(fig)
    return path


def plot_compare(
    *,
    path: Path,
    energy_eV: np.ndarray,
    delays_fs: np.ndarray,
    maps: list[tuple[str, np.ndarray]],
    config: DemoConfig,
    shared_vlim: float | None,
) -> Path:
    """Plot 2x3 comparison.

    If shared_vlim is None, each panel is normalized by its displayed p99abs and
    plotted with the same fixed colorbar [-1, 1]. The raw min/max/scale are
    annotated in the upper-left corner of each panel.
    """

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(16.0, 8.4),
        sharey=True,
        constrained_layout=False,
    )
    axes_flat = axes.ravel()

    last_mesh = None
    for ax, (title, values) in zip(axes_flat, maps):
        x, plot_values, xlabel = prepare_plot_arrays(energy_eV, values, config)

        if shared_vlim is None:
            plot_values, scale_used, raw_min, raw_max = normalize_for_panel_display(
                plot_values,
                scale_mode="p99abs",
            )
            last_mesh = ax.pcolormesh(
                x,
                delays_fs,
                plot_values,
                shading="auto",
                cmap=config.cmap,
                vmin=-1.0,
                vmax=1.0,
            )
            ax.text(
                0.02,
                0.98,
                (
                    f"min={raw_min:.2e}\n"
                    f"max={raw_max:.2e}\n"
                    f"scale={scale_used:.2e}"
                ),
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=8,
                bbox={
                    "facecolor": "white",
                    "alpha": 0.72,
                    "edgecolor": "none",
                    "pad": 2,
                },
            )
        else:
            last_mesh = ax.pcolormesh(
                x,
                delays_fs,
                plot_values,
                shading="auto",
                cmap=config.cmap,
                vmin=-float(shared_vlim),
                vmax=float(shared_vlim),
            )

        ax.set_title(title)
        ax.set_xlabel(xlabel)

    for ax in axes[:, 0]:
        ax.set_ylabel("Pump-probe delay (fs)")

    mode = "shared raw scale" if shared_vlim is not None else "per-panel normalized to [-1, 1]"
    fig.suptitle(f"TA phase handling comparison ({mode})", y=0.965)

    # Leave explicit room for an external colorbar.
    fig.subplots_adjust(
        left=0.07,
        right=0.86,
        bottom=0.08,
        top=0.91,
        wspace=0.28,
        hspace=0.32,
    )

    if last_mesh is not None:
        cax = fig.add_axes([0.89, 0.15, 0.018, 0.68])
        cbar = fig.colorbar(last_mesh, cax=cax)
        if shared_vlim is None:
            cbar.set_label("Normalized S_TA (each panel / displayed p99abs)")
            cbar.set_ticks([-1.0, -0.5, 0.0, 0.5, 1.0])
        else:
            cbar.set_label("S_TA (arb., current sign)")

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=int(config.figure_dpi))
    plt.close(fig)
    return path


def run_demo(config: DemoConfig, *, output_dir: Path, quick: bool = False) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = output_dir / "data"
    fig_dir = output_dir / "figures"

    delays = tuple(float(x) for x in (config.quick_probe_delays_fs if quick else config.probe_delays_fs))
    delays_array = np.asarray(delays, dtype=float)
    normalizer = ParaNormalizer(auto_scale=True)

    probe_field = make_probe_reference_field(config)
    probe_params = make_physical_params(
        config,
        probe_field,
        case_name="probe_only",
        description="Probe-only reference shared by all TA maps.",
    )
    probe_result = run_with_checkpoint(
        probe_params,
        normalizer=normalizer,
        output_dir=output_dir,
        case_key="probe_only",
        config=config,
    )
    probe_response = response_from_result(probe_result, probe_field, config)

    energy_eV = np.asarray(probe_response["energy_eV"], dtype=float)
    omega_fs_inv = np.asarray(probe_response["omega_fs_inv"], dtype=float)

    phase_maps: list[np.ndarray] = []
    phase_names: list[str] = []

    for phase in config.pump_phase_cases_rad:
        label = phase_label(phase)
        map_name = f"phase_{label}"
        _, _, ta_map = compute_map(
            config,
            delays_fs=delays,
            output_dir=output_dir,
            normalizer=normalizer,
            probe_field=probe_field,
            probe_response=probe_response,
            pump_phase_rad=float(phase),
            field_policy="cep_locked_delay_changes_carrier_phase",
            map_name=map_name,
        )
        phase_maps.append(ta_map)
        phase_names.append(label)

    phase_stack = np.stack(phase_maps, axis=0)
    ta_phase_avg = np.mean(phase_stack, axis=0)

    # Diagnostic only: normalize each phase case before averaging.
    # This is not the physical phase-cycling result.
    phase_stack_unitnorm = np.stack(
        [normalize_map_for_diagnostic(item, scale="p99abs") for item in phase_maps],
        axis=0,
    )
    ta_phase_avg_unitnorm_diagnostic = np.mean(phase_stack_unitnorm, axis=0)

    _, _, ta_fixed_carrier = compute_map(
        config,
        delays_fs=delays,
        output_dir=output_dir,
        normalizer=normalizer,
        probe_field=probe_field,
        probe_response=probe_response,
        pump_phase_rad=0.0,
        field_policy="lab_phase_locked_envelope_only",
        map_name="fixed_carrier",
    )

    all_maps = phase_maps + [ta_phase_avg, ta_fixed_carrier]
    shared_vlim = robust_vlim(all_maps, percentile=99.0)

    phase_mean_maxabs = float(
        np.mean(
            [
                map_stats(f"phase_{label}", ta_map)["maxabs"]
                for label, ta_map in zip(phase_names, phase_maps)
            ]
        )
    )

    stats_rows = []
    for label, ta_map in zip(phase_names, phase_maps):
        stats_rows.append(
            map_stats(f"TA_phase_{label}_full_energy", ta_map, reference_maxabs=phase_mean_maxabs)
        )
        stats_rows.append(
            map_stats(
                f"TA_phase_{label}_displayed_energy",
                displayed_energy_map_values(energy_eV, ta_map, config),
                reference_maxabs=phase_mean_maxabs,
            )
        )

    stats_rows.append(
        map_stats("TA_phase_avg_raw_full_energy", ta_phase_avg, reference_maxabs=phase_mean_maxabs)
    )
    stats_rows.append(
        map_stats(
            "TA_phase_avg_raw_displayed_energy",
            displayed_energy_map_values(energy_eV, ta_phase_avg, config),
            reference_maxabs=phase_mean_maxabs,
        )
    )
    stats_rows.append(
        map_stats(
            "TA_phase_avg_unitnorm_diagnostic_displayed_energy",
            displayed_energy_map_values(energy_eV, ta_phase_avg_unitnorm_diagnostic, config),
        )
    )
    stats_rows.append(
        map_stats("TA_fixed_carrier_full_energy", ta_fixed_carrier, reference_maxabs=phase_mean_maxabs)
    )
    stats_rows.append(
        map_stats(
            "TA_fixed_carrier_displayed_energy",
            displayed_energy_map_values(energy_eV, ta_fixed_carrier, config),
            reference_maxabs=phase_mean_maxabs,
        )
    )

    data_dir.mkdir(parents=True, exist_ok=True)
    stats_csv = write_csv_rows(data_dir / "map_stats.csv", stats_rows)
    stats_json = write_json(data_dir / "map_stats.json", {"map_stats": stats_rows})

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

    for label, ta_map in zip(phase_names, phase_maps):
        filename = phase_filenames.get(label, f"ta_phase_case_{label}.png")
        figure_paths[f"phase_case_{label}"] = str(
            plot_one_map(
                path=fig_dir / filename,
                title=f"TA map: {phase_titles.get(label, label)}",
                energy_eV=energy_eV,
                delays_fs=delays_array,
                values=ta_map,
                config=config,
                vlim=shared_vlim,
            )
        )

    figure_paths["phase_avg"] = str(
        plot_one_map(
            path=fig_dir / "ta_phase_avg.png",
            title="TA map: 4-step pump-phase average, shared raw scale",
            energy_eV=energy_eV,
            delays_fs=delays_array,
            values=ta_phase_avg,
            config=config,
            vlim=shared_vlim,
        )
    )
    figure_paths["phase_avg_autoscale"] = str(
        plot_one_map(
            path=fig_dir / "ta_phase_avg_autoscale.png",
            title="TA map: 4-step pump-phase average",
            energy_eV=energy_eV,
            delays_fs=delays_array,
            values=ta_phase_avg,
            config=config,
            vlim=None,
        )
    )
    figure_paths["phase_avg_unitnorm_diagnostic"] = str(
        plot_one_map(
            path=fig_dir / "ta_phase_avg_unitnorm_diagnostic.png",
            title="Diagnostic: mean of unit-normalized phase maps",
            energy_eV=energy_eV,
            delays_fs=delays_array,
            values=ta_phase_avg_unitnorm_diagnostic,
            config=config,
            vlim=None,
        )
    )
    figure_paths["fixed_carrier"] = str(
        plot_one_map(
            path=fig_dir / "ta_fixed_carrier.png",
            title="TA map: fixed lab-frame carrier phase, shared raw scale",
            energy_eV=energy_eV,
            delays_fs=delays_array,
            values=ta_fixed_carrier,
            config=config,
            vlim=shared_vlim,
        )
    )
    figure_paths["fixed_carrier_autoscale"] = str(
        plot_one_map(
            path=fig_dir / "ta_fixed_carrier_autoscale.png",
            title="TA map: fixed lab-frame carrier phase",
            energy_eV=energy_eV,
            delays_fs=delays_array,
            values=ta_fixed_carrier,
            config=config,
            vlim=None,
        )
    )

    compare_maps = [
        ("phase 0", phase_maps[0]),
        ("phase π/2", phase_maps[1]),
        ("phase π", phase_maps[2]),
        ("phase 3π/2", phase_maps[3]),
        ("phase average", ta_phase_avg),
        ("fixed carrier", ta_fixed_carrier),
    ]

    figure_paths["compare_shared"] = str(
        plot_compare(
            path=fig_dir / "ta_phase_cycling_compare.png",
            energy_eV=energy_eV,
            delays_fs=delays_array,
            maps=compare_maps,
            config=config,
            shared_vlim=shared_vlim,
        )
    )
    figure_paths["compare_autoscale"] = str(
        plot_compare(
            path=fig_dir / "ta_phase_cycling_compare_autoscale.png",
            energy_eV=energy_eV,
            delays_fs=delays_array,
            maps=compare_maps,
            config=config,
            shared_vlim=None,
        )
    )

    npz_path = data_dir / "ta_phase_cycling_comparison.npz"
    np.savez_compressed(
        npz_path,
        delays_fs=delays_array,
        energy_eV=energy_eV,
        omega_fs_inv=omega_fs_inv,
        wavelength_nm=HC_EV_NM / energy_eV,
        TA_phase_cases=phase_stack,
        TA_phase_avg=ta_phase_avg,
        TA_phase_avg_unitnorm_diagnostic=ta_phase_avg_unitnorm_diagnostic,
        TA_fixed_carrier=ta_fixed_carrier,
        phase_values_rad=np.asarray(config.pump_phase_cases_rad, dtype=float),
        phase_labels=np.asarray(phase_names, dtype=str),
    )

    meta = {
        "example_name": config.example_name,
        "quick": bool(quick),
        "output_dir": output_dir,
        "data_npz": npz_path,
        "stats_csv": stats_csv,
        "stats_json": stats_json,
        "figures": figure_paths,
        "phase_cases": {
            "pump_phase_rad": list(config.pump_phase_cases_rad),
            "probe_phase_rad": config.probe_phase_rad,
            "physical_phase_average": "unweighted arithmetic mean across four pump phases",
            "diagnostic_unitnorm_average": (
                "Each phase map is divided by its own p99 abs value before averaging. "
                "This is not a physical phase-cycling result."
            ),
        },
        "field_policies": {
            "phase_cases": {
                "name": "cep_locked_delay_changes_carrier_phase",
                "expression": "E(t)=envelope(t-center)*cos(omega*(t-center)+phase0)",
                "expected_behavior": "delay changes relative optical carrier phase and can generate ripples",
            },
            "fixed_carrier": {
                "name": "lab_phase_locked_envelope_only",
                "expression": "E(t)=envelope(t-center)*cos(omega*t+phase0)",
                "expected_behavior": "v3-like envelope shift; usually smoother in this direct TA observable",
            },
        },
        "model_parameters": {
            "basis": config.basis,
            "energies_eV": config.energies_eV,
            "dipole_matrix_D": config.dipole_matrix_D,
            "transition_notes": [
                "g<->e near 1.55 eV gives bleach/stimulated-emission-like feature under current sign convention.",
                "e<->f near 1.70 eV gives excited-state-absorption-like feature under current sign convention.",
            ],
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
            "n_time_points": int(np.asarray(probe_result.times_fs).size),
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
        "shared_color_scale": {
            "vlim": shared_vlim,
            "mode": "symmetric ±99th percentile abs over phase cases, raw phase average, and fixed carrier",
        },
        "normalized_display": {
            "enabled_for": [
                "ta_phase_avg_autoscale.png",
                "ta_phase_avg_unitnorm_diagnostic.png",
                "ta_fixed_carrier_autoscale.png",
                "ta_phase_cycling_compare_autoscale.png",
            ],
            "normalization": "displayed map divided by displayed-range p99abs",
            "colorbar": "fixed to [-1, 1]",
            "annotation": "raw min/max/scale shown in upper-left panel text",
        },
    }
    meta_path = write_json(output_dir / "meta.json", meta)

    print("TA phase-cycling comparison finished.")
    print(f"n delays          : {len(delays)}")
    print(f"energy points     : {energy_eV.size}")
    print(f"output directory  : {output_dir}")
    print(f"data npz          : {npz_path}")
    print(f"stats csv         : {stats_csv}")
    print(f"metadata          : {meta_path}")
    print(f"compare shared    : {figure_paths['compare_shared']}")
    print(f"compare normalized: {figure_paths['compare_autoscale']}")

    return {
        "output_dir": str(output_dir),
        "data_npz": str(npz_path),
        "stats_csv": str(stats_csv),
        "stats_json": str(stats_json),
        "meta_json": str(meta_path),
        "figures": figure_paths,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for checkpoints, data, metadata, and figures.",
    )
    parser.add_argument(
        "--force-run",
        action="store_true",
        help="Ignore existing checkpoints and rerun all simulations.",
    )
    parser.add_argument(
        "--no-checkpoints",
        action="store_true",
        help="Run without checkpoint load/save.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use a smaller delay grid for smoke testing.",
    )
    parser.add_argument(
        "--wavelength",
        action="store_true",
        help="Plot wavelength instead of photon energy on the x-axis.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_config = DemoConfig()
    config = DemoConfig(
        **{
            **asdict(base_config),
            "force_run": bool(args.force_run),
            "use_checkpoints": not bool(args.no_checkpoints),
            "plot_use_wavelength": bool(args.wavelength),
        }
    )
    run_demo(config, output_dir=Path(args.output_dir), quick=bool(args.quick))


if __name__ == "__main__":
    main()

