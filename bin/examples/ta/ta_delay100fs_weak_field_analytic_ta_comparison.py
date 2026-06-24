#!/usr/bin/env python3
"""Read a TA delay spectrum and compare it with a weak-field analytic TA reference.

This script reads the TA spectra written by the base demo, selects one delay
(default 100 fs), and compares the numerical spectrum with a three-level
weak-field analytic reference.

Model assumed by the analytic reference:

* 0 <-> 1: GSB/SE band, centered at E_01;
* 1 <-> 2: ESA band, centered at E_12;
* weak pump, no population relaxation, no pulse overlap;
* pure-dephasing homogeneous Lorentzian linewidths;
* the common pump-created excited population cancels after global normalization.

Unlike a unit-peak Lorentz template, this reference keeps the leading TA
amplitude prefactors:

    S_ref(E) = sign * [2 * mu_01^2 * E * L_01(E)
                       - mu_12^2 * E * L_12(E)]

where the factor 2 in the 0<->1 band represents GSB + SE, and the optional
factor E approximates the omega factor in omega * Im[P/E].

This is still not a full finite-pulse response calculation. It is a weak-field
analytic amplitude reference for checking whether the relative strengths of the
0<->1 and 1<->2 peaks are consistent with dipoles, pathway multiplicity, and
pure-dephasing linewidths.
"""

from __future__ import annotations

import argparse
import csv
import os.path
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


BASE_EXAMPLE_NAME = "ta_three_level_intrinsic_response_phase_cycling_demo_no_relaxation"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / BASE_EXAMPLE_NAME
DEFAULT_SPECTRA_CSV = DEFAULT_OUTPUT_DIR / "data" / "ta_all_delay_spectra.csv"


def normalized_maxabs(values: np.ndarray, *, name: str) -> np.ndarray:
    """Normalize by max absolute value; raise if the signal is zero."""
    array = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values.")
    scale = float(np.max(np.abs(array)))
    if scale == 0.0:
        raise ValueError(f"{name} is identically zero and cannot be normalized.")
    return array / scale


def lorentzian_energy(energy_eV: np.ndarray, *, center_eV: float, gamma_eV: float) -> np.ndarray:
    """Return an energy-domain Lorentzian with peak height 1/gamma_eV."""
    if gamma_eV <= 0.0:
        raise ValueError("gamma_eV must be positive.")
    delta_eV = np.asarray(energy_eV, dtype=float) - float(center_eV)
    return float(gamma_eV) / (delta_eV**2 + float(gamma_eV)**2)


def read_delay_spectrum(path: Path, *, delay_fs: float, signal_column: str) -> tuple[np.ndarray, np.ndarray, float]:
    """Read exactly one delay spectrum from a CSV file."""
    if not path.exists():
        raise FileNotFoundError(
            f"Missing spectra CSV: {path}\n"
            "Run ta_three_level_intrinsic_response_phase_cycling_demo.py first."
        )
    rows: list[tuple[float, float, float]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or signal_column not in reader.fieldnames:
            raise ValueError(f"CSV does not contain signal column {signal_column!r}.")
        for row in reader:
            row_delay = float(row["delay_fs"])
            if np.isclose(row_delay, delay_fs, rtol=0.0, atol=1.0e-9):
                rows.append((float(row["energy_eV"]), float(row[signal_column]), row_delay))
    if not rows:
        raise ValueError(f"Requested delay {delay_fs:g} fs is absent from {path}.")
    actual_delays = {item[2] for item in rows}
    if len(actual_delays) != 1:
        raise ValueError(f"Ambiguous matched delays: {sorted(actual_delays)}")
    rows.sort(key=lambda item: item[0])
    energy = np.asarray([item[0] for item in rows], dtype=float)
    signal = np.asarray([item[1] for item in rows], dtype=float)
    if np.any(np.diff(energy) <= 0.0):
        raise ValueError("Selected spectrum must have strictly increasing energy values.")
    return energy, signal, float(rows[0][2])


def restrict_energy_range(
    energy_eV: np.ndarray,
    signal: np.ndarray,
    *,
    energy_min_eV: float,
    energy_max_eV: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Restrict the selected spectrum to the requested energy range."""
    if energy_min_eV >= energy_max_eV:
        raise ValueError("energy_min_eV must be smaller than energy_max_eV.")
    mask = (energy_eV >= energy_min_eV) & (energy_eV <= energy_max_eV)
    if not np.any(mask):
        raise ValueError("The requested energy window contains no spectrum points.")
    return np.asarray(energy_eV[mask], dtype=float), np.asarray(signal[mask], dtype=float)


def make_theory_reference(
    energy_eV: np.ndarray,
    *,
    energy_01_eV: float,
    energy_12_eV: float,
    gamma_1_fs_inv: float,
    gamma_2_fs_inv: float,
    mu_01: float,
    mu_12: float,
    hbar_eV_fs: float = 0.6582119569,
    include_omega_factor: bool = True,
    overall_sign: float = 1.0,
) -> dict[str, np.ndarray | float | bool]:
    """Construct a weak-field analytic TA reference for a 3-level model.

    Level-projector pure dephasing convention:

        L_n = sqrt(gamma_n) |n><n|

    gives coherence decay rates:

        Gamma_01 = gamma_1 / 2
        Gamma_12 = (gamma_1 + gamma_2) / 2

    The reference keeps leading TA amplitude factors:

        0<->1 GSB/SE band: 2 * mu_01^2 * L_01
        1<->2 ESA band:    1 * mu_12^2 * L_12

    The common weak-pump excited population is omitted because it cancels after
    global normalization for the no-relaxation comparison.
    """
    if gamma_1_fs_inv < 0.0 or gamma_2_fs_inv < 0.0:
        raise ValueError("Pure-dephasing rates must be non-negative.")
    if mu_01 <= 0.0 or mu_12 <= 0.0:
        raise ValueError("Transition dipoles must be positive.")
    if overall_sign not in (-1.0, 1.0):
        raise ValueError("overall_sign should be either +1 or -1.")

    gamma_01_fs_inv = 0.5 * gamma_1_fs_inv
    gamma_12_fs_inv = 0.5 * (gamma_1_fs_inv + gamma_2_fs_inv)
    gamma_01_eV = hbar_eV_fs * gamma_01_fs_inv
    gamma_12_eV = hbar_eV_fs * gamma_12_fs_inv

    line_01 = lorentzian_energy(energy_eV, center_eV=energy_01_eV, gamma_eV=gamma_01_eV)
    line_12 = lorentzian_energy(energy_eV, center_eV=energy_12_eV, gamma_eV=gamma_12_eV)

    # Weak-field TA pathway weights.
    gsb_se_weighted = 2.0 * (mu_01**2) * line_01
    esa_weighted = 1.0 * (mu_12**2) * line_12

    if include_omega_factor:
        # The demo readout is proportional to omega * Im[P/E].  On an energy
        # axis, the relative omega factor can be represented by energy_eV.
        gsb_se_weighted = energy_eV * gsb_se_weighted
        esa_weighted = energy_eV * esa_weighted

    signed_ta_reference = overall_sign * (gsb_se_weighted - esa_weighted)

    # Unit-peak template retained only as a linewidth/position diagnostic.
    unit_peak_template = normalized_maxabs(line_01, name="Lorentz 0-1") - normalized_maxabs(
        line_12, name="Lorentz 1-2"
    )

    return {
        "line_01": line_01,
        "line_12": line_12,
        "unit_peak_template": unit_peak_template,
        "gsb_se_weighted": gsb_se_weighted,
        "esa_weighted": esa_weighted,
        "signed_ta_reference": signed_ta_reference,
        "gamma_01_fs_inv": gamma_01_fs_inv,
        "gamma_12_fs_inv": gamma_12_fs_inv,
        "gamma_01_eV": gamma_01_eV,
        "gamma_12_eV": gamma_12_eV,
        "mu_01": mu_01,
        "mu_12": mu_12,
        "include_omega_factor": include_omega_factor,
        "overall_sign": overall_sign,
    }


def write_comparison_csv(
    path: Path,
    *,
    energy_eV: np.ndarray,
    signal: np.ndarray,
    signal_norm: np.ndarray,
    theory: dict[str, np.ndarray | float | bool],
) -> Path:
    """Save numerical data and analytic-reference curves."""
    path.parent.mkdir(parents=True, exist_ok=True)
    line_01_norm = normalized_maxabs(np.asarray(theory["line_01"]), name="Lorentz 0-1")
    line_12_norm = normalized_maxabs(np.asarray(theory["line_12"]), name="Lorentz 1-2")
    unit_template_norm = normalized_maxabs(np.asarray(theory["unit_peak_template"]), name="unit template")
    gsb_se_raw = np.asarray(theory["gsb_se_weighted"], dtype=float)
    esa_raw = np.asarray(theory["esa_weighted"], dtype=float)
    signed_raw = np.asarray(theory["signed_ta_reference"], dtype=float)
    signed_norm = normalized_maxabs(signed_raw, name="weak-field analytic TA reference")

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "energy_eV",
                "wavelength_nm",
                "TA_signal_raw",
                "TA_signal_normalized",
                "Lorentz_01_unit_peak",
                "Lorentz_12_unit_peak",
                "unit_peak_template_normalized",
                "GSB_SE_weighted_raw",
                "ESA_weighted_raw",
                "weak_field_TA_reference_raw",
                "weak_field_TA_reference_normalized",
            ),
        )
        writer.writeheader()
        for index, energy in enumerate(energy_eV):
            writer.writerow(
                {
                    "energy_eV": float(energy),
                    "wavelength_nm": float(1239.8419843320026 / energy),
                    "TA_signal_raw": float(signal[index]),
                    "TA_signal_normalized": float(signal_norm[index]),
                    "Lorentz_01_unit_peak": float(line_01_norm[index]),
                    "Lorentz_12_unit_peak": float(line_12_norm[index]),
                    "unit_peak_template_normalized": float(unit_template_norm[index]),
                    "GSB_SE_weighted_raw": float(gsb_se_raw[index]),
                    "ESA_weighted_raw": float(esa_raw[index]),
                    "weak_field_TA_reference_raw": float(signed_raw[index]),
                    "weak_field_TA_reference_normalized": float(signed_norm[index]),
                }
            )
    return path


def signal_metrics(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "max_abs": float(np.max(np.abs(array))),
        "rms": float(np.sqrt(np.mean(array**2))),
        "peak_to_peak": float(np.max(array) - np.min(array)),
    }


def plot_comparison(
    path: Path,
    *,
    energy_eV: np.ndarray,
    signal_norm: np.ndarray,
    theory: dict[str, np.ndarray | float | bool],
    delay_fs: float,
    signal_column: str,
    dpi: int,
) -> Path:
    """Plot numerical spectrum against weak-field analytic reference."""
    line_01 = normalized_maxabs(np.asarray(theory["line_01"]), name="Lorentz 0-1")
    line_12 = normalized_maxabs(np.asarray(theory["line_12"]), name="Lorentz 1-2")
    unit_template = normalized_maxabs(np.asarray(theory["unit_peak_template"]), name="unit template")

    signed = normalized_maxabs(
        np.asarray(theory["signed_ta_reference"]),
        name="weak-field analytic TA reference",
    )
    gsb_se = np.asarray(theory["gsb_se_weighted"], dtype=float)
    esa = np.asarray(theory["esa_weighted"], dtype=float)
    component_scale = float(np.max(np.abs(np.asarray(theory["signed_ta_reference"], dtype=float))))
    if component_scale == 0.0:
        raise ValueError("weak-field analytic TA reference is zero.")
    gsb_se_component = gsb_se / component_scale
    esa_component = -esa / component_scale * float(theory["overall_sign"])
    if float(theory["overall_sign"]) < 0:
        gsb_se_component = -gsb_se_component

    fig, ax_main = plt.subplots(1, 1, figsize=(6.4, 4.2), sharex=True)
    ax_main.plot(energy_eV, signal_norm, color="black", linewidth=2.0, label=f"{signal_column}, normalized")
    ax_main.plot(
        energy_eV,
        signed,
        color="C3",
        linewidth=1.8,
        linestyle="--",
        label="weak-field analytic TA reference",
    )
    ax_main.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    ax_main.set(xlabel="Probe photon energy (eV)", ylabel="Global max-abs normalized signal")
    ax_main.set_title(f"TA at {delay_fs:g} fs: numerical spectrum vs. amplitude reference")
    ax_main.legend(loc="best")
    ax_main.grid(alpha=0.25)
    fig.tight_layout()

    fig_components, ax_components = plt.subplots(1, 1, figsize=(6.4, 4.2), sharex=True)
    ax_components.plot(energy_eV, gsb_se_component, color="C0", label="+ 2 mu_01^2 L_01 component")
    ax_components.plot(energy_eV, esa_component, color="C2", label="- mu_12^2 L_12 component")
    ax_components.plot(energy_eV, signed, color="C3", linewidth=1.8, linestyle="--", label="sum, globally normalized")
    ax_components.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    ax_components.set(xlabel="Probe photon energy (eV)", ylabel="Normalized by summed reference max abs")
    ax_components.set_title("Weighted analytic components")
    ax_components.legend(loc="best")
    ax_components.grid(alpha=0.25)
    fig_components.tight_layout()

    fig_template, ax_template = plt.subplots(1, 1, figsize=(6.4, 4.2), sharex=True)
    ax_template.plot(energy_eV, signal_norm, color="black", linewidth=2.0, label=f"{signal_column}, normalized")
    ax_template.plot(energy_eV, unit_template, color="C4", linestyle="--", label="unit-peak Lorentz template")
    ax_template.plot(energy_eV, line_01, color="C0", alpha=0.7, label="unit Lorentz 0↔1")
    ax_template.plot(energy_eV, -line_12, color="C2", alpha=0.7, label="- unit Lorentz 1↔2")
    ax_template.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    ax_template.set(xlabel="Probe photon energy (eV)", ylabel="Unit-peak normalized template")
    ax_template.set_title("Linewidth-only unit-peak template")
    ax_template.legend(loc="best")
    ax_template.grid(alpha=0.25)
    fig_template.tight_layout()

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    fig_components.savefig(os.path.join(os.path.dirname(path), "ta_delay_weak_field_components.png"), dpi=dpi)
    fig_template.savefig(os.path.join(os.path.dirname(path), "ta_delay_unit_peak_lorentz_template.png"), dpi=dpi)
    plt.close(fig)
    plt.close(fig_components)
    plt.close(fig_template)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spectra-csv", type=Path, default=DEFAULT_SPECTRA_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--delay-fs", type=float, default=100.0)
    parser.add_argument("--signal-column", default="TA_phase_avg")
    parser.add_argument("--energy-min-eV", type=float, default=1.35)
    parser.add_argument("--energy-max-eV", type=float, default=1.90)
    parser.add_argument("--energy-01-eV", type=float, default=1.55)
    parser.add_argument("--energy-12-eV", type=float, default=1.70)
    parser.add_argument("--gamma-1-fs-inv", type=float, default=1.0 / 120.0)
    parser.add_argument("--gamma-2-fs-inv", type=float, default=1.0 / 100.0)
    parser.add_argument("--mu-01", type=float, default=5.0)
    parser.add_argument("--mu-12", type=float, default=9.0)
    parser.add_argument("--no-omega-factor", action="store_true")
    parser.add_argument("--overall-sign", type=float, choices=(-1.0, 1.0), default=1.0)
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    energy, signal, actual_delay = read_delay_spectrum(
        args.spectra_csv,
        delay_fs=args.delay_fs,
        signal_column=args.signal_column,
    )
    energy, signal = restrict_energy_range(
        energy,
        signal,
        energy_min_eV=args.energy_min_eV,
        energy_max_eV=args.energy_max_eV,
    )
    signal_norm = normalized_maxabs(signal, name=args.signal_column)
    theory = make_theory_reference(
        energy,
        energy_01_eV=args.energy_01_eV,
        energy_12_eV=args.energy_12_eV,
        gamma_1_fs_inv=args.gamma_1_fs_inv,
        gamma_2_fs_inv=args.gamma_2_fs_inv,
        mu_01=args.mu_01,
        mu_12=args.mu_12,
        include_omega_factor=not args.no_omega_factor,
        overall_sign=args.overall_sign,
    )

    data_path = args.output_dir / "data" / f"ta_delay_{actual_delay:g}fs_weak_field_analytic_ta_comparison.csv"
    figure_path = args.output_dir / "figures" / "plot" / f"ta_delay_{actual_delay:g}fs_weak_field_analytic_ta_comparison.png"
    write_comparison_csv(data_path, energy_eV=energy, signal=signal, signal_norm=signal_norm, theory=theory)
    plot_comparison(
        figure_path,
        energy_eV=energy,
        signal_norm=signal_norm,
        theory=theory,
        delay_fs=actual_delay,
        signal_column=args.signal_column,
        dpi=args.dpi,
    )

    signal_info = signal_metrics(signal_norm)
    ref_info = signal_metrics(normalized_maxabs(np.asarray(theory["signed_ta_reference"]), name="reference"))

    print(f"delay = {actual_delay:g} fs")
    print(f"gamma_01 = {float(theory['gamma_01_fs_inv']):.6g} fs^-1; gamma_12 = {float(theory['gamma_12_fs_inv']):.6g} fs^-1")
    print(f"gamma_01_E = {float(theory['gamma_01_eV']):.6g} eV; gamma_12_E = {float(theory['gamma_12_eV']):.6g} eV")
    print(f"mu_01 = {float(theory['mu_01']):.6g}; mu_12 = {float(theory['mu_12']):.6g}")
    print(f"include omega factor = {bool(theory['include_omega_factor'])}; overall sign = {float(theory['overall_sign']):.0f}")
    print(f"numerical normalized metrics: {signal_info}")
    print(f"reference normalized metrics: {ref_info}")
    print(f"comparison CSV: {data_path}")
    print(f"comparison plot: {figure_path}")


if __name__ == "__main__":
    main()
