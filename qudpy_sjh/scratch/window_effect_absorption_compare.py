"""Compare FFT window effects for two-level and three-level absorption spectra.

This script is a diagnostic workflow, not a new spectroscopy implementation.

It deliberately reuses existing QuDPy APIs:
    - NLevelPhysicalParams / RelaxationChannel / PureDephasingChannel
    - make_default_gaussian_carrier_field
    - run_case
    - polarization_C_per_m2
    - lab_frame_fft_response

Only the `window` argument passed to `lab_frame_fft_response(...)` is changed.
No FFT axis, small-denominator mask, or complex division is reimplemented here.

Current QuDPy `apply_time_window(...)` supports:
    None / "none" / "hann"

Therefore this script compares:
    window=None
    window="hann"

If you want Hamming later, extend `qudpy_sjh.utils.spectroscopy.apply_time_window`
first, then add "hamming" to WINDOWS below.
"""

from __future__ import annotations

from pathlib import Path
import csv
import json
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

if __package__ is None or __package__ == "":
    # This file is intended to live at:
    #   QuDPy/qudpy_sjh/scratch/window_effect_absorption_compare.py
    # so parents[2] is the repository root.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from qudpy_sjh.utils.core import (
    NLevelPhysicalParams,
    ParaNormalizer,
    PureDephasingChannel,
    RelaxationChannel,
    run_case,
)
from qudpy_sjh.utils.fields import make_default_gaussian_carrier_field
from qudpy_sjh.utils.spectroscopy import lab_frame_fft_response_legacy, polarization_C_per_m2


EXAMPLE_NAME = "window_effect_absorption_compare"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / EXAMPLE_NAME

NUMBER_DENSITY_M3 = 1.0e24

# Keep this aligned with the current implementation of apply_time_window().
WINDOWS: tuple[str | None, ...] = (None, "hann")

# Plot only the physically relevant range, but save all returned spectral points.
PLOT_ENERGY_RANGE_EV = (1.10, 2.10)


def _window_label(window: str | None) -> str:
    return "none" if window is None else str(window)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("rows must not be empty.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _make_probe_field():
    """Construct a weak broadband Gaussian lab-frame probe field."""

    return make_default_gaussian_carrier_field(
        E0_MV_per_cm=0.005,
        laser_energy_eV=1.60,
        pulse_center_fs=0.0,
        pulse_sigma_fs=8.0,
        phase_rad=0.0,
        name="weak_broadband_probe",
        metadata={
            "example_name": EXAMPLE_NAME,
            "role": "probe",
            "note": "Shared weak probe for window-effect diagnostic.",
        },
    )


def _make_two_level_params(field) -> NLevelPhysicalParams:
    return NLevelPhysicalParams(
        energies_eV=(0.0, 1.55),
        dipole_matrix_D=(
            (0.0, 5.0),
            (5.0, 0.0),
        ),
        t_start_fs=-200.0,
        t_end_fs=500.0,
        dt_fs=0.2,
        field=field,
        basis=("g", "e"),
        relaxation_channels=(
            RelaxationChannel(
                name="relaxation_1_to_0",
                from_level=1,
                to_level=0,
                T1_fs=1000.0,
            ),
        ),
        pure_dephasing_channels=(
            PureDephasingChannel(
                name="pure_dephasing_level_1",
                level=1,
                Tphi_fs=120.0,
            ),
        ),
        solver_mode="lab_exact",
        input_description="Two-level absorption spectrum for FFT window comparison.",
        input_metadata={
            "example_name": EXAMPLE_NAME,
            "system_name": "two_level",
        },
    )


def _make_three_level_params(field) -> NLevelPhysicalParams:
    return NLevelPhysicalParams(
        energies_eV=(0.0, 1.48, 1.72),
        dipole_matrix_D=(
            (0.0, 4.0, 3.0),
            (4.0, 0.0, 0.0),
            (3.0, 0.0, 0.0),
        ),
        t_start_fs=-200.0,
        t_end_fs=500.0,
        dt_fs=0.2,
        field=field,
        basis=("g", "e1", "e2"),
        relaxation_channels=(
            RelaxationChannel(
                name="relaxation_1_to_0",
                from_level=1,
                to_level=0,
                T1_fs=1000.0,
            ),
            RelaxationChannel(
                name="relaxation_2_to_0",
                from_level=2,
                to_level=0,
                T1_fs=900.0,
            ),
        ),
        pure_dephasing_channels=(
            PureDephasingChannel(
                name="pure_dephasing_level_1",
                level=1,
                Tphi_fs=120.0,
            ),
            PureDephasingChannel(
                name="pure_dephasing_level_2",
                level=2,
                Tphi_fs=100.0,
            ),
        ),
        solver_mode="lab_exact",
        input_description="Three-level absorption spectrum for FFT window comparison.",
        input_metadata={
            "example_name": EXAMPLE_NAME,
            "system_name": "three_level",
        },
    )


def _polarization_from_result(result) -> np.ndarray:
    physical = result.physical_params
    if physical is None:
        raise ValueError("DynamicsResult.physical_params is required.")

    return polarization_C_per_m2(
        result.density_array(),
        physical.dipole_matrix_D,
        NUMBER_DENSITY_M3,
    )


def _response_from_result(result, field, *, window: str | None) -> dict[str, np.ndarray]:
    t_fs = np.asarray(result.times_fs, dtype=float)
    if t_fs.ndim != 1 or t_fs.size < 2:
        raise ValueError("result.times_fs must be a 1D array with at least two points.")

    E_t = np.asarray(field(t_fs), dtype=float)
    P_t = _polarization_from_result(result)

    # rhoij is required by lab_frame_fft_response. In this diagnostic, the plotted
    # absorption-like quantity is based on P_over_E, not rhoij_over_E.
    return lab_frame_fft_response_legacy(
        t_fs=t_fs,
        E_MV_per_cm=E_t,
        P_C_per_m2=P_t,
        rhoij=result.matrix_element(0, 1),
        window=window,
        subtract_mean=True,
        rel_threshold=1e-6,
        zero_padding_factor=4,
    )


def _collect_spectra(system_name: str, result, field) -> list[dict[str, Any]]:
    spectra: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []

    for window in WINDOWS:
        label = _window_label(window)
        response = _response_from_result(result, field, window=window)

        energy = np.asarray(response["energy_eV"], dtype=float)
        omega = np.asarray(response["omega_fs_inv"], dtype=float)
        absorption = np.asarray(response["neg_omega_im_P_over_E"], dtype=float)
        p_over_e = np.asarray(response["P_over_E"], dtype=np.complex128)
        abs_e_fft = np.asarray(response["abs_E_fft"], dtype=float)

        order = np.argsort(energy)
        energy = energy[order]
        omega = omega[order]
        absorption = absorption[order]
        p_over_e = p_over_e[order]
        abs_e_fft = abs_e_fft[order]

        spectra.append(
            {
                "system_name": system_name,
                "window": label,
                "energy_eV": energy,
                "omega_fs_inv": omega,
                "absorption": absorption,
                "P_over_E": p_over_e,
                "abs_E_fft": abs_e_fft,
                "n_points": int(energy.size),
            }
        )

        for idx in range(energy.size):
            rows.append(
                {
                    "system_name": system_name,
                    "window": label,
                    "energy_eV": float(energy[idx]),
                    "omega_fs_inv": float(omega[idx]),
                    "neg_omega_im_P_over_E": float(absorption[idx]),
                    "Re_P_over_E": float(np.real(p_over_e[idx])),
                    "Im_P_over_E": float(np.imag(p_over_e[idx])),
                    "abs_P_over_E": float(np.abs(p_over_e[idx])),
                    "abs_E_fft": float(abs_e_fft[idx]),
                }
            )

    _write_rows(OUTPUT_DIR / f"{system_name}_window_spectra.csv", rows)
    return spectra


def _plot_absorption(all_spectra: dict[str, list[dict[str, Any]]]) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), sharex=True)

    for ax, system_name, title in zip(
        axes,
        ("two_level", "three_level"),
        ("Two-level absorption", "Three-level absorption"),
    ):
        for item in all_spectra[system_name]:
            energy = item["energy_eV"]
            absorption = item["absorption"]

            mask = (energy >= PLOT_ENERGY_RANGE_EV[0]) & (energy <= PLOT_ENERGY_RANGE_EV[1])
            if not np.any(mask):
                mask = np.ones_like(energy, dtype=bool)

            ax.plot(
                energy[mask],
                absorption[mask],
                linewidth=1.5,
                label=f"window={item['window']}",
            )

        ax.set_title(title)
        ax.set_xlabel("Energy (eV)")
        ax.set_ylabel(r"$-\omega\,\mathrm{Im}[P(\omega)/E(\omega)]$")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)

    fig.suptitle("FFT window effect on lab-frame absorption-like spectra")
    fig.tight_layout()

    path = OUTPUT_DIR / "window_effect_absorption_compare.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _plot_probe_fft(all_spectra: dict[str, list[dict[str, Any]]]) -> Path:
    """Optional diagnostic: show how the field spectrum changes with window."""

    fig, ax = plt.subplots(figsize=(6.4, 4.0))

    # The probe field is shared by both systems, so use one system's response.
    for item in all_spectra["two_level"]:
        energy = item["energy_eV"]
        abs_e_fft = item["abs_E_fft"]

        mask = (energy >= PLOT_ENERGY_RANGE_EV[0]) & (energy <= PLOT_ENERGY_RANGE_EV[1])
        if not np.any(mask):
            mask = np.ones_like(energy, dtype=bool)

        y = abs_e_fft[mask]
        y_max = float(np.max(y)) if y.size else 1.0
        if y_max > 0:
            y = y / y_max

        ax.plot(
            energy[mask],
            y,
            linewidth=1.5,
            label=f"window={item['window']}",
        )

    ax.set_title("Probe FFT support after mask")
    ax.set_xlabel("Energy (eV)")
    ax.set_ylabel(r"normalized $|E(\omega)|$")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()

    path = OUTPUT_DIR / "window_effect_probe_fft.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    field = _make_probe_field()
    normalizer = ParaNormalizer(auto_scale=True)

    cases = {
        "two_level": _make_two_level_params(field),
        "three_level": _make_three_level_params(field),
    }

    all_spectra: dict[str, list[dict[str, Any]]] = {}
    sanity: dict[str, Any] = {}

    for system_name, params in cases.items():
        print(f"Running {system_name}...")
        result = run_case(params, normalizer=normalizer)

        all_spectra[system_name] = _collect_spectra(system_name, result, field)
        sanity[system_name] = {
            "dimension": int(result.dimension()),
            "max_trace_error": float(result.max_trace_error()),
            "max_hermiticity_error": float(result.max_hermiticity_error()),
            "n_time_points": int(np.asarray(result.times_fs).size),
            "spectral_points_by_window": {
                item["window"]: item["n_points"]
                for item in all_spectra[system_name]
            },
        }

    absorption_fig = _plot_absorption(all_spectra)
    probe_fft_fig = _plot_probe_fft(all_spectra)

    metadata = {
        "example_name": EXAMPLE_NAME,
        "description": (
            "Diagnostic comparison of FFT window choices in lab-frame "
            "absorption-like spectra. The script reuses lab_frame_fft_response "
            "and changes only its window argument."
        ),
        "windows": [_window_label(w) for w in WINDOWS],
        "window_note": (
            "Current QuDPy apply_time_window supports None/'none'/'hann'. "
            "Add hamming to the shared spectroscopy helper before using it here."
        ),
        "number_density_m3": NUMBER_DENSITY_M3,
        "plot_energy_range_eV": list(PLOT_ENERGY_RANGE_EV),
        "probe_field": field.to_dict(),
        "outputs": {
            "absorption_figure": str(absorption_fig),
            "probe_fft_figure": str(probe_fft_fig),
            "two_level_csv": str(OUTPUT_DIR / "two_level_window_spectra.csv"),
            "three_level_csv": str(OUTPUT_DIR / "three_level_window_spectra.csv"),
        },
        "sanity": sanity,
    }
    _write_json(OUTPUT_DIR / "window_effect_absorption_compare_metadata.json", metadata)

    print("Window-effect absorption comparison finished.")
    print(f"absorption figure: {absorption_fig}")
    print(f"probe FFT figure : {probe_fft_fig}")
    print(f"output dir       : {OUTPUT_DIR}")


if __name__ == "__main__":
    main()