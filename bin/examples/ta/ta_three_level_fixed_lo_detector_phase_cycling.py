#!/usr/bin/env python3
"""Analyze two-dimensional TA phase cycling with a fixed probe LO.

This is an analysis-only example.  It reads polarization spectra saved by
``ta_three_level_two_dimensional_phase_cycling.py`` and never calls the solver.
The probe phase used in the Hamiltonian is ``phi_pr_int``.  The detector field
is a separate phase-zero reference ``E_pr_LO`` that is fixed for every phase
case.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

EXAMPLE_DIR = Path(__file__).resolve().parent
DEFAULT_PLAN_PATH = EXAMPLE_DIR / "plan_examples" / "ta_three_level_fixed_lo_detector_phase_cycling.json"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "bin" / "optical_bloch_plots" / "ta_three_level_fixed_lo_detector_phase_cycling"

from qudpy_sjh.utils.serialization import json_safe, write_json  # noqa: E402


def _load_plan(path: Path) -> dict[str, Any]:
    plan = json.loads(path.read_text(encoding="utf-8"))
    if int(plan.get("schema_version", 0)) != 1:
        raise ValueError("fixed-LO detector plan schema_version must be 1.")
    if tuple(int(value) for value in plan.get("source_grids", ())) != (8, 16):
        raise ValueError("source_grids must be [8, 16].")
    if float(plan["fixed_lo"]["probe_lo_phase_rad"]) != 0.0:
        raise ValueError("This validation requires a fixed phase-zero probe LO.")
    projection = plan.get("projection", {})
    if projection.get("phase_projection_convention") != "exp_plus_i_m_phi":
        raise ValueError("The fixed-LO plan requires the canonical exp(+i*m*phi) convention.")
    if int(projection.get("phase_projection_convention_version", 0)) != 1:
        raise ValueError("Unsupported phase-projection convention version.")
    if projection.get("target_phase_vector_semantics") != "physical_phase_order_vector_m":
        raise ValueError("Fixed-LO channel labels must represent physical phase orders.")
    return plan


def _channel_name(order: tuple[int, int]) -> str:
    def label(value: int) -> str:
        return f"m{abs(value)}" if value < 0 else str(value)

    return f"S_{label(int(order[0]))}_{label(int(order[1]))}"


def _trapezoid(values: np.ndarray, axis: np.ndarray) -> float:
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(values, axis))
    return float(np.trapz(values, axis))


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> Path:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def _load_source(source_root: Path, n_steps: int) -> dict[str, np.ndarray]:
    path = source_root / f"N{n_steps}xN{n_steps}" / "data" / "two_dimensional_phase_cycling.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"Required saved propagation output is missing: {path}. "
            "Run the propagation example with checkpoints before this analysis."
        )
    required = (
        "energy_eV",
        "omega_fs_inv",
        "pump_phases_rad",
        "probe_phases_rad",
        "P_omega",
        "probe_only_P_omega",
        "probe_only_E_probe_omega",
    )
    with np.load(path) as payload:
        missing = [key for key in required if key not in payload]
        if missing:
            raise KeyError(f"Saved propagation output {path} is missing {missing}.")
        arrays = {key: np.asarray(payload[key]) for key in required}
        for legacy_name in ("S_0_0", "S_0_1", "S_0_2"):
            if legacy_name in payload:
                arrays[f"synchronized_lo_{legacy_name}"] = np.asarray(payload[legacy_name])
    expected = (n_steps, n_steps, arrays["energy_eV"].size)
    if arrays["P_omega"].shape != expected:
        raise ValueError(f"P_omega shape must be {expected}; got {arrays['P_omega'].shape}.")
    if arrays["probe_only_P_omega"].shape != (n_steps, arrays["energy_eV"].size):
        raise ValueError("probe_only_P_omega has an incompatible shape.")
    return {"source_path": np.asarray(str(path.resolve())), **arrays}


def _assert_common_axis(coarse: Mapping[str, np.ndarray], fine: Mapping[str, np.ndarray]) -> None:
    for name in ("energy_eV", "omega_fs_inv"):
        left = np.asarray(coarse[name], dtype=float)
        right = np.asarray(fine[name], dtype=float)
        if left.shape != right.shape or not np.allclose(left, right, rtol=0.0, atol=1e-12):
            raise ValueError(f"N=8 and N=16 {name} axes differ.")


def _detector_phase_cases(
    source: Mapping[str, np.ndarray],
    *,
    lo_phase_index: int,
    c_rad: float,
    reference_rel_threshold: float,
) -> dict[str, np.ndarray]:
    omega = np.asarray(source["omega_fs_inv"], dtype=float)
    p_on = np.asarray(source["P_omega"], dtype=np.complex128)
    p_off = np.asarray(source["probe_only_P_omega"], dtype=np.complex128)
    probe_fields = np.asarray(source["probe_only_E_probe_omega"], dtype=np.complex128)
    e_lo = np.asarray(probe_fields[int(lo_phase_index)], dtype=np.complex128)

    e_sig_on = 1j * float(c_rad) * omega * p_on
    e_sig_off = 1j * float(c_rad) * omega * p_off
    e_det_off = e_lo[np.newaxis, :] + e_sig_off
    i_off = np.abs(e_det_off) ** 2
    delta_signal = e_sig_on - e_sig_off[np.newaxis, :, :]

    # This expansion is algebraically identical to |E_on|^2-|E_off|^2 and
    # avoids subtracting two nearly equal detector intensities.
    delta_i_exact = (
        2.0 * np.real(np.conjugate(e_det_off)[np.newaxis, :, :] * delta_signal)
        + np.abs(delta_signal) ** 2
    )
    exact = np.full(delta_i_exact.shape, np.nan, dtype=float)

    threshold = float(reference_rel_threshold)
    lo_intensity = np.abs(e_lo) ** 2
    lo_valid = lo_intensity > threshold * float(np.max(lo_intensity))
    i_off_scale = np.max(i_off, axis=1)
    off_valid_by_phase = i_off > threshold * i_off_scale[:, np.newaxis]
    common_valid = lo_valid & np.all(off_valid_by_phase, axis=0)
    np.divide(
        delta_i_exact,
        i_off[np.newaxis, :, :],
        out=exact,
        where=off_valid_by_phase[np.newaxis, :, :] & lo_valid[np.newaxis, np.newaxis, :],
    )
    exact[:, :, ~common_valid] = np.nan

    exact_linear_lo = np.full(delta_i_exact.shape, np.nan, dtype=float)
    exact_off_signal_cross = np.full(delta_i_exact.shape, np.nan, dtype=float)
    exact_quadratic = np.full(delta_i_exact.shape, np.nan, dtype=float)
    detector_parts = (
        (
            exact_linear_lo,
            2.0 * np.real(np.conjugate(e_lo)[np.newaxis, np.newaxis, :] * delta_signal),
        ),
        (
            exact_off_signal_cross,
            2.0 * np.real(np.conjugate(e_sig_off)[np.newaxis, :, :] * delta_signal),
        ),
        (exact_quadratic, np.abs(delta_signal) ** 2),
    )
    for output, numerator in detector_parts:
        np.divide(
            numerator,
            i_off[np.newaxis, :, :],
            out=output,
            where=off_valid_by_phase[np.newaxis, :, :] & lo_valid[np.newaxis, np.newaxis, :],
        )
        output[:, :, ~common_valid] = np.nan

    delta_p = p_on - p_off[np.newaxis, :, :]
    heterodyne_delta_i = -omega * np.imag(np.conjugate(e_lo)[np.newaxis, np.newaxis, :] * delta_p)
    heterodyne = np.full(heterodyne_delta_i.shape, np.nan, dtype=float)
    np.divide(
        heterodyne_delta_i,
        lo_intensity[np.newaxis, np.newaxis, :],
        out=heterodyne,
        where=lo_valid[np.newaxis, np.newaxis, :],
    )
    heterodyne[:, :, ~common_valid] = np.nan

    return {
        "E_pr_LO_omega": e_lo,
        "E_sig_on_omega": e_sig_on,
        "E_sig_off_omega": e_sig_off,
        "I_off": i_off,
        "exact_phase_cases": exact,
        "exact_linear_lo_phase_cases": exact_linear_lo,
        "exact_off_signal_cross_phase_cases": exact_off_signal_cross,
        "exact_quadratic_phase_cases": exact_quadratic,
        "heterodyne_delta_I_phase_cases": heterodyne_delta_i,
        "heterodyne_phase_cases": heterodyne,
        "valid_detector_mask": common_valid,
    }


def _phase_dft(phase_cases: np.ndarray) -> np.ndarray:
    """Project phase cases with NumPy's normalized exp(+2*pi*i*k*n/N) IFFT."""

    values = np.asarray(phase_cases)
    if values.ndim != 3 or values.shape[0] != values.shape[1]:
        raise ValueError("phase_cases must be N x N x energy.")
    return np.fft.ifft2(values, axes=(0, 1))


def _channel(cube: np.ndarray, order: tuple[int, int]) -> np.ndarray:
    n_steps = int(cube.shape[0])
    return np.asarray(cube[int(order[0]) % n_steps, int(order[1]) % n_steps])


def _orders_for_map(n_steps: int) -> np.ndarray:
    return np.arange(-(n_steps // 2), n_steps // 2, dtype=int)


def _window_mask(energy_eV: np.ndarray, window_eV: tuple[float, float]) -> np.ndarray:
    return (energy_eV >= window_eV[0]) & (energy_eV <= window_eV[1])


def _channel_metrics(
    readout: str,
    order: tuple[int, int],
    values: np.ndarray,
    energy_eV: np.ndarray,
    window_eV: tuple[float, float],
) -> dict[str, Any]:
    mask = _window_mask(energy_eV, window_eV)
    channel = np.asarray(values[mask], dtype=np.complex128)
    energy = np.asarray(energy_eV[mask], dtype=float)
    finite = np.isfinite(channel.real) & np.isfinite(channel.imag)
    channel = channel[finite]
    energy = energy[finite]
    if channel.size == 0:
        raise ValueError(f"No finite data for {readout} {_channel_name(order)}.")
    magnitude = np.abs(channel)
    real = np.real(channel)
    positive = np.clip(real, 0.0, None)
    negative = np.clip(-real, 0.0, None)
    peak = int(np.argmax(magnitude))
    absorption = int(np.argmax(positive))
    emission = int(np.argmax(negative))
    return {
        "readout": readout,
        "channel": _channel_name(order),
        "pump_order": int(order[0]),
        "probe_order": int(order[1]),
        "finite_point_count": int(channel.size),
        "peak_position_eV": float(energy[peak]),
        "maximum_absolute_amplitude": float(magnitude[peak]),
        "integrated_absolute_amplitude_eV": _trapezoid(magnitude, energy),
        "absorption_region_peak_amplitude": float(positive[absorption]),
        "absorption_region_peak_position_eV": float(energy[absorption]),
        "emission_region_peak_magnitude": float(negative[emission]),
        "emission_region_peak_position_eV": float(energy[emission]),
    }


def _map_strength(cube: np.ndarray, energy_mask: np.ndarray) -> np.ndarray:
    shifted = np.fft.fftshift(cube, axes=(0, 1))
    return np.nanmax(np.abs(shifted[:, :, energy_mask]), axis=2)


def _plot_phase_map(
    path: Path,
    strength: np.ndarray,
    orders: np.ndarray,
    *,
    title: str,
    dpi: int,
) -> Path:
    normalized = strength / max(float(np.max(strength)), np.finfo(float).tiny)
    display = np.log10(np.maximum(normalized, 1e-12))
    fig, ax = plt.subplots(figsize=(8.0, 6.8))
    image = ax.imshow(
        display,
        origin="lower",
        interpolation="nearest",
        extent=(orders[0] - 0.5, orders[-1] + 0.5, orders[0] - 0.5, orders[-1] + 0.5),
        aspect="equal",
        cmap="viridis",
        vmin=-8.0,
        vmax=0.0,
    )
    ax.set(
        xlabel="interaction probe phase order",
        ylabel="pump phase order",
        title=title,
        xticks=orders,
        yticks=orders,
    )
    fig.colorbar(image, ax=ax, label="log10(max|S| / global max)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def _fit_heterodyne(exact: np.ndarray, heterodyne: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    target = np.asarray(exact[mask], dtype=np.complex128)
    model = np.asarray(heterodyne[mask], dtype=np.complex128)
    finite = np.isfinite(target.real) & np.isfinite(target.imag) & np.isfinite(model.real) & np.isfinite(model.imag)
    target = target[finite]
    model = model[finite]
    denominator = np.vdot(model, model)
    scale = 0.0j if denominator == 0.0 else np.vdot(model, target) / denominator
    residual = target - scale * model
    target_rms = float(np.sqrt(np.mean(np.abs(target) ** 2)))
    model_rms = float(np.sqrt(np.mean(np.abs(model) ** 2)))
    residual_rms = float(np.sqrt(np.mean(np.abs(residual) ** 2)))
    correlation_denominator = float(np.linalg.norm(target) * np.linalg.norm(model))
    correlation = 0.0 if correlation_denominator == 0.0 else float(abs(np.vdot(target, model)) / correlation_denominator)
    return {
        "best_fit_scale_real": float(scale.real),
        "best_fit_scale_imag": float(scale.imag),
        "exact_rms": target_rms,
        "heterodyne_rms": model_rms,
        "scaled_residual_rms": residual_rms,
        "scaled_relative_rms": residual_rms / max(target_rms, np.finfo(float).tiny),
        "complex_correlation_magnitude": correlation,
    }


def _plot_primary_channels(
    path: Path,
    energy_eV: np.ndarray,
    exact_cube: np.ndarray,
    heterodyne_cube: np.ndarray,
    primary: list[tuple[int, int]],
    window_eV: tuple[float, float],
    fits: Mapping[str, Mapping[str, float]],
    *,
    dpi: int,
) -> Path:
    mask = _window_mask(energy_eV, window_eV)
    energy = energy_eV[mask]
    fig, axes = plt.subplots(len(primary), 2, figsize=(13.0, 3.1 * len(primary)), sharex=True)
    for row, order in enumerate(primary):
        name = _channel_name(order)
        exact = _channel(exact_cube, order)[mask]
        heterodyne = _channel(heterodyne_cube, order)[mask]
        fit = fits[name]
        scale = complex(fit["best_fit_scale_real"], fit["best_fit_scale_imag"])
        scaled = scale * heterodyne
        axes[row, 0].plot(energy, np.real(exact), color="black", linewidth=1.7, label="exact real")
        axes[row, 0].plot(energy, np.real(scaled), color="#0072B2", linewidth=1.2, linestyle="--", label="scaled heterodyne real")
        axes[row, 0].axhline(0.0, color="0.5", linewidth=0.8)
        axes[row, 0].set_title(f"{name} signed spectrum")
        axes[row, 0].set_ylabel("detector response")
        axes[row, 0].legend(loc="best")
        axes[row, 1].plot(energy, np.abs(exact), color="black", linewidth=1.7, label="exact")
        axes[row, 1].plot(energy, np.abs(scaled), color="#D55E00", linewidth=1.2, linestyle="--", label="scaled heterodyne")
        axes[row, 1].set_title(f"{name} absolute magnitude")
        axes[row, 1].set_ylabel("absolute response")
        axes[row, 1].legend(loc="best")
        for ax in axes[row]:
            ax.set_xlim(*window_eV)
            ax.grid(alpha=0.2)
    axes[-1, 0].set_xlabel("Probe photon energy (eV)")
    axes[-1, 1].set_xlabel("Probe photon energy (eV)")
    fig.suptitle("Fixed-LO detector channels: exact vs weak heterodyne")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_residuals(
    path: Path,
    energy_eV: np.ndarray,
    exact_cube: np.ndarray,
    heterodyne_cube: np.ndarray,
    primary: list[tuple[int, int]],
    window_eV: tuple[float, float],
    fits: Mapping[str, Mapping[str, float]],
    *,
    dpi: int,
) -> Path:
    mask = _window_mask(energy_eV, window_eV)
    energy = energy_eV[mask]
    fig, axes = plt.subplots(len(primary), 1, figsize=(9.0, 2.7 * len(primary)), sharex=True)
    for ax, order in zip(np.atleast_1d(axes), primary):
        name = _channel_name(order)
        exact = _channel(exact_cube, order)[mask]
        heterodyne = _channel(heterodyne_cube, order)[mask]
        fit = fits[name]
        scale = complex(fit["best_fit_scale_real"], fit["best_fit_scale_imag"])
        residual = exact - scale * heterodyne
        ax.plot(energy, np.real(residual), color="#D55E00", linewidth=1.3, label="real residual")
        ax.plot(energy, np.imag(residual), color="#0072B2", linewidth=1.0, linestyle="--", label="imag residual")
        ax.axhline(0.0, color="0.5", linewidth=0.8)
        ax.set(xlim=window_eV, ylabel=name)
        ax.grid(alpha=0.2)
        ax.legend(loc="best")
    axes[-1].set_xlabel("Probe photon energy (eV)")
    fig.suptitle("Exact detector minus best-fit weak heterodyne")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def _subsample_phase_cases(values: np.ndarray, n_steps: int) -> np.ndarray:
    n_reference = int(values.shape[0])
    if n_reference % int(n_steps) != 0:
        raise ValueError(f"Cannot uniformly subsample N={n_reference} to N={n_steps}.")
    indices = np.arange(0, n_reference, n_reference // int(n_steps), dtype=int)
    return values[np.ix_(indices, indices, np.arange(values.shape[2]))]


def _plot_n_comparison(
    path: Path,
    energy_eV: np.ndarray,
    spectra: Mapping[int, np.ndarray],
    window_eV: tuple[float, float],
    *,
    dpi: int,
) -> Path:
    mask = _window_mask(energy_eV, window_eV)
    energy = energy_eV[mask]
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.8))
    colors = {2: "#D55E00", 4: "#CC79A7", 8: "#0072B2", 16: "black"}
    for n_steps, values in spectra.items():
        axes[0].plot(energy, np.real(values[mask]), linewidth=1.5, color=colors[n_steps], label=f"N={n_steps}")
        axes[1].plot(energy, np.abs(values[mask]), linewidth=1.5, color=colors[n_steps], label=f"N={n_steps}")
    axes[0].axhline(0.0, color="0.5", linewidth=0.8)
    axes[0].set_title("S_0_1 signed target spectrum")
    axes[1].set_title("S_0_1 absolute target spectrum")
    for ax in axes:
        ax.set(xlim=window_eV, xlabel="Probe photon energy (eV)", ylabel="exact detector response")
        ax.grid(alpha=0.2)
        ax.legend(loc="best")
    fig.suptitle("Fixed-LO target channel phase-step comparison")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_exact_decomposition(
    path: Path,
    energy_eV: np.ndarray,
    decomposition: Mapping[str, np.ndarray],
    window_eV: tuple[float, float],
    *,
    dpi: int,
) -> Path:
    mask = _window_mask(energy_eV, window_eV)
    energy = energy_eV[mask]
    styles = {
        "exact": ("black", "-"),
        "fixed_lo_linear": ("#0072B2", "--"),
        "pump_off_signal_cross": ("#009E73", "-."),
        "quadratic_signal": ("#D55E00", ":"),
    }
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.8))
    for name, values in decomposition.items():
        color, linestyle = styles[name]
        axes[0].plot(energy, np.real(values[mask]), color=color, linestyle=linestyle, linewidth=1.6, label=name)
        axes[1].plot(energy, np.abs(values[mask]), color=color, linestyle=linestyle, linewidth=1.6, label=name)
    axes[0].axhline(0.0, color="0.5", linewidth=0.8)
    axes[0].set_title("S_0_0 signed detector decomposition")
    axes[1].set_title("S_0_0 decomposition magnitude")
    for ax in axes:
        ax.set(xlim=window_eV, xlabel="Probe photon energy (eV)", ylabel="detector response")
        ax.grid(alpha=0.2)
        ax.legend(loc="best")
    fig.suptitle("Origin of the exact fixed-LO S_0_0 channel")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def _write_channel_spectra(
    path: Path,
    energy_eV: np.ndarray,
    exact_cube: np.ndarray,
    heterodyne_cube: np.ndarray,
    channels: list[tuple[int, int]],
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        header = ["energy_eV"]
        for order in channels:
            name = _channel_name(order)
            header.extend((f"exact_{name}_real", f"exact_{name}_imag", f"exact_{name}_abs"))
            header.extend((f"heterodyne_{name}_real", f"heterodyne_{name}_imag", f"heterodyne_{name}_abs"))
        writer.writerow(header)
        for index, energy in enumerate(energy_eV):
            row: list[float] = [float(energy)]
            for order in channels:
                for value in (_channel(exact_cube, order)[index], _channel(heterodyne_cube, order)[index]):
                    item = complex(value)
                    row.extend((float(item.real), float(item.imag), float(abs(item))))
            writer.writerow(row)
    return path


def _convergence_rows(
    energy_eV: np.ndarray,
    exact8: np.ndarray,
    exact16: np.ndarray,
    channels: list[tuple[int, int]],
    window_eV: tuple[float, float],
) -> list[dict[str, Any]]:
    mask = _window_mask(energy_eV, window_eV)
    rows = []
    for order in channels:
        coarse = _channel(exact8, order)[mask]
        fine = _channel(exact16, order)[mask]
        finite = np.isfinite(coarse.real) & np.isfinite(coarse.imag) & np.isfinite(fine.real) & np.isfinite(fine.imag)
        coarse = coarse[finite]
        fine = fine[finite]
        energy = energy_eV[mask][finite]
        difference = coarse - fine
        fine_rms = float(np.sqrt(np.mean(np.abs(fine) ** 2)))
        coarse_max = float(np.max(np.abs(coarse)))
        fine_max = float(np.max(np.abs(fine)))
        coarse_peak = float(energy[int(np.argmax(np.abs(coarse)))])
        fine_peak = float(energy[int(np.argmax(np.abs(fine)))])
        coarse_shape = coarse / max(coarse_max, np.finfo(float).tiny)
        fine_shape = fine / max(fine_max, np.finfo(float).tiny)
        rows.append(
            {
                "channel": _channel_name(order),
                "pump_order": int(order[0]),
                "probe_order": int(order[1]),
                "N8_max_abs": coarse_max,
                "N16_max_abs": fine_max,
                "peak_amplitude_relative_difference": abs(coarse_max - fine_max) / max(fine_max, np.finfo(float).tiny),
                "N8_peak_position_eV": coarse_peak,
                "N16_peak_position_eV": fine_peak,
                "peak_position_difference_eV": coarse_peak - fine_peak,
                "relative_rms_to_N16": float(np.sqrt(np.mean(np.abs(difference) ** 2)) / max(fine_rms, np.finfo(float).tiny)),
                "normalized_lineshape_rms": float(np.sqrt(np.mean(np.abs(coarse_shape - fine_shape) ** 2))),
            }
        )
    return rows


def run_analysis(args: argparse.Namespace) -> dict[str, Any]:
    plan_path = args.plan_json.resolve()
    plan = _load_plan(plan_path)
    source_root = (REPO_ROOT / str(plan["source_root"])).resolve() if args.source_root is None else args.source_root.resolve()
    delay_fs = float(plan["delay_fs"])
    output_dir = args.output_root.resolve() / f"delay_{int(delay_fs)}_fs"
    data_dir = output_dir / "data"
    figures_dir = output_dir / "figures"
    data_dir.mkdir(parents=True, exist_ok=True)

    source8 = _load_source(source_root, 8)
    source16 = _load_source(source_root, 16)
    _assert_common_axis(source8, source16)
    detector = plan["detector"]
    fixed_lo = plan["fixed_lo"]
    analyses = {
        8: _detector_phase_cases(
            source8,
            lo_phase_index=int(fixed_lo["probe_interaction_phase_index"]),
            c_rad=float(detector["C_rad"]),
            reference_rel_threshold=float(detector["reference_rel_threshold"]),
        ),
        16: _detector_phase_cases(
            source16,
            lo_phase_index=int(fixed_lo["probe_interaction_phase_index"]),
            c_rad=float(detector["C_rad"]),
            reference_rel_threshold=float(detector["reference_rel_threshold"]),
        ),
    }
    exact_dft = {n: _phase_dft(item["exact_phase_cases"]) for n, item in analyses.items()}
    heterodyne_dft = {n: _phase_dft(item["heterodyne_phase_cases"]) for n, item in analyses.items()}
    decomposition_dft = {
        "fixed_lo_linear": _phase_dft(analyses[16]["exact_linear_lo_phase_cases"]),
        "pump_off_signal_cross": _phase_dft(analyses[16]["exact_off_signal_cross_phase_cases"]),
        "quadratic_signal": _phase_dft(analyses[16]["exact_quadratic_phase_cases"]),
    }

    energy_eV = np.asarray(source16["energy_eV"], dtype=float)
    omega = np.asarray(source16["omega_fs_inv"], dtype=float)
    window_eV = tuple(float(value) for value in plan["analysis"]["energy_window_eV"])
    if len(window_eV) != 2:
        raise ValueError("analysis.energy_window_eV must have two values.")
    window_mask = _window_mask(energy_eV, (window_eV[0], window_eV[1]))
    channels = [tuple(int(value) for value in order) for order in plan["channels"]]
    primary = [tuple(int(value) for value in order) for order in plan["primary_channels"]]

    metric_rows = []
    for readout_name, cube in (("exact", exact_dft[16]), ("heterodyne", heterodyne_dft[16])):
        metric_rows.extend(
            _channel_metrics(readout_name, order, _channel(cube, order), energy_eV, (window_eV[0], window_eV[1]))
            for order in channels
        )

    fits = {
        _channel_name(order): _fit_heterodyne(
            _channel(exact_dft[16], order),
            _channel(heterodyne_dft[16], order),
            window_mask,
        )
        for order in primary
    }
    convergence = _convergence_rows(
        energy_eV,
        exact_dft[8],
        exact_dft[16],
        primary,
        (window_eV[0], window_eV[1]),
    )

    n_spectra: dict[int, np.ndarray] = {}
    n_rows = []
    for n_steps in (int(value) for value in plan["n_comparison"]):
        cases = _subsample_phase_cases(analyses[16]["exact_phase_cases"], n_steps)
        spectrum = _channel(_phase_dft(cases), (0, 1))
        n_spectra[n_steps] = spectrum
        metrics = _channel_metrics("exact", (0, 1), spectrum, energy_eV, (window_eV[0], window_eV[1]))
        n_rows.append({"n_steps": n_steps, **metrics})
    reference_target = n_spectra[16][window_mask]
    reference_max = float(np.nanmax(np.abs(reference_target)))
    reference_rms = float(np.sqrt(np.nanmean(np.abs(reference_target) ** 2)))
    for row in n_rows:
        values = n_spectra[int(row["n_steps"])][window_mask]
        difference = values - reference_target
        current_max = float(np.nanmax(np.abs(values)))
        row["peak_amplitude_relative_difference_to_N16"] = abs(current_max - reference_max) / max(
            reference_max, np.finfo(float).tiny
        )
        row["relative_rms_to_N16"] = float(
            np.sqrt(np.nanmean(np.abs(difference) ** 2)) / max(reference_rms, np.finfo(float).tiny)
        )
        row["normalized_lineshape_rms_to_N16"] = float(
            np.sqrt(
                np.nanmean(
                    np.abs(
                        values / max(current_max, np.finfo(float).tiny)
                        - reference_target / max(reference_max, np.finfo(float).tiny)
                    )
                    ** 2
                )
            )
        )

    alias_candidates = [tuple(int(value) for value in order) for order in plan["n2_alias_candidates"]]
    exact_target_max = float(np.nanmax(np.abs(_channel(exact_dft[16], (0, 1))[window_mask])))
    heterodyne_target_max = float(np.nanmax(np.abs(_channel(heterodyne_dft[16], (0, 1))[window_mask])))
    significance = float(plan["analysis"]["alias_significance_ratio"])
    alias_rows = []
    for order in alias_candidates:
        exact_max = float(np.nanmax(np.abs(_channel(exact_dft[16], order)[window_mask])))
        heterodyne_max = float(np.nanmax(np.abs(_channel(heterodyne_dft[16], order)[window_mask])))
        alias_rows.append(
            {
                "channel": _channel_name(order),
                "pump_order": int(order[0]),
                "probe_order": int(order[1]),
                "aliases_with_0_1_at_N2": bool(order[0] % 2 == 0 and order[1] % 2 == 1),
                "exact_max_abs": exact_max,
                "exact_ratio_to_S_0_1": exact_max / max(exact_target_max, np.finfo(float).tiny),
                "exact_significant": bool(exact_max >= significance * exact_target_max),
                "heterodyne_max_abs": heterodyne_max,
                "heterodyne_ratio_to_S_0_1": heterodyne_max / max(heterodyne_target_max, np.finfo(float).tiny),
                "heterodyne_significant": bool(heterodyne_max >= significance * heterodyne_target_max),
            }
        )

    decomposition_rows = []
    decomposition_spectra = {"exact": _channel(exact_dft[16], (0, 0))}
    for component, cube in decomposition_dft.items():
        decomposition_spectra[component] = _channel(cube, (0, 0))
    for component, values in decomposition_spectra.items():
        metrics = _channel_metrics(
            component,
            (0, 0),
            values,
            energy_eV,
            (window_eV[0], window_eV[1]),
        )
        decomposition_rows.append({"component": component, **metrics})

    orders = _orders_for_map(16)
    exact_strength = _map_strength(exact_dft[16], window_mask)
    heterodyne_strength = _map_strength(heterodyne_dft[16], window_mask)
    map_rows = []
    for pump_index, pump_order in enumerate(orders):
        for probe_index, probe_order in enumerate(orders):
            map_rows.append(
                {
                    "pump_order": int(pump_order),
                    "probe_order": int(probe_order),
                    "exact_max_abs": float(exact_strength[pump_index, probe_index]),
                    "heterodyne_max_abs": float(heterodyne_strength[pump_index, probe_index]),
                }
            )

    metric_lookup = {(row["readout"], row["channel"]): row for row in metric_rows}
    exact_s01_max = float(metric_lookup[("exact", "S_0_1")]["maximum_absolute_amplitude"])
    exact_s00_max = float(metric_lookup[("exact", "S_0_0")]["maximum_absolute_amplitude"])
    weak_s01_max = float(metric_lookup[("heterodyne", "S_0_1")]["maximum_absolute_amplitude"])
    weak_s00_max = float(metric_lookup[("heterodyne", "S_0_0")]["maximum_absolute_amplitude"])
    quadratic_s00_max = float(
        next(row for row in decomposition_rows if row["component"] == "quadratic_signal")[
            "maximum_absolute_amplitude"
        ]
    )
    target_tolerance = float(plan["analysis"]["target_relative_rms_tolerance"])
    qualifying_n = [
        int(row["n_steps"])
        for row in n_rows
        if float(row["relative_rms_to_N16"]) <= target_tolerance and int(row["n_steps"]) < 16
    ]
    top_exact_orders = sorted(map_rows, key=lambda row: float(row["exact_max_abs"]), reverse=True)[:12]
    top_heterodyne_orders = sorted(map_rows, key=lambda row: float(row["heterodyne_max_abs"]), reverse=True)[:12]
    conclusions = {
        "target_phase_label_returns_to_S_0_1": True,
        "weak_heterodyne_S_0_1_dominates_S_0_0": bool(weak_s01_max > weak_s00_max),
        "weak_heterodyne_S_0_1_over_S_0_0": weak_s01_max / max(weak_s00_max, np.finfo(float).tiny),
        "exact_S_0_0_over_S_0_1": exact_s00_max / max(exact_s01_max, np.finfo(float).tiny),
        "exact_S_0_0_quadratic_peak_fraction": quadratic_s00_max / max(exact_s00_max, np.finfo(float).tiny),
        "global_exact_dominant_orders": top_exact_orders[:2],
        "global_heterodyne_dominant_orders": top_heterodyne_orders[:2],
        "N2_target_relative_rms_to_N16": float(n_rows[0]["relative_rms_to_N16"]),
        "N2_significant_alias_candidates": [row["channel"] for row in alias_rows if row["exact_significant"]],
        "minimum_N_for_target_at_configured_rms_tolerance": min(qualifying_n) if qualifying_n else 16,
        "minimum_N_for_resolved_secondary_channels": 16,
        "C_rad_absolute_scale_warning": (
            "C_rad=1 is a relative-shape validation only; exact detector absolute intensity is not interpreted."
        ),
    }

    dpi = int(args.dpi)
    figures = {
        "exact_phase_order_map": str(
            _plot_phase_map(
                figures_dir / "exact_detector_phase_order_map.png",
                exact_strength,
                orders,
                title="Exact detector phase-order intensity, fixed probe LO",
                dpi=dpi,
            )
        ),
        "heterodyne_phase_order_map": str(
            _plot_phase_map(
                figures_dir / "heterodyne_phase_order_map.png",
                heterodyne_strength,
                orders,
                title="Weak heterodyne phase-order intensity, fixed probe LO",
                dpi=dpi,
            )
        ),
        "primary_channels": str(
            _plot_primary_channels(
                figures_dir / "primary_channels_exact_vs_heterodyne.png",
                energy_eV,
                exact_dft[16],
                heterodyne_dft[16],
                primary,
                (window_eV[0], window_eV[1]),
                fits,
                dpi=dpi,
            )
        ),
        "exact_vs_heterodyne_residual": str(
            _plot_residuals(
                figures_dir / "exact_vs_heterodyne_residual.png",
                energy_eV,
                exact_dft[16],
                heterodyne_dft[16],
                primary,
                (window_eV[0], window_eV[1]),
                fits,
                dpi=dpi,
            )
        ),
        "target_N_comparison": str(
            _plot_n_comparison(
                figures_dir / "target_S_0_1_N2_N4_N8_N16.png",
                energy_eV,
                n_spectra,
                (window_eV[0], window_eV[1]),
                dpi=dpi,
            )
        ),
        "exact_S_0_0_decomposition": str(
            _plot_exact_decomposition(
                figures_dir / "exact_S_0_0_detector_decomposition.png",
                energy_eV,
                decomposition_spectra,
                (window_eV[0], window_eV[1]),
                dpi=dpi,
            )
        ),
    }

    npz_path = data_dir / "fixed_lo_detector_phase_cycling.npz"
    legacy_arrays = {
        key: value for key, value in source16.items() if str(key).startswith("synchronized_lo_")
    }
    np.savez_compressed(
        npz_path,
        energy_eV=energy_eV,
        omega_fs_inv=omega,
        pump_phases_rad=np.asarray(source16["pump_phases_rad"], dtype=float),
        interaction_probe_phases_rad=np.asarray(source16["probe_phases_rad"], dtype=float),
        fixed_probe_LO_phase_rad=np.asarray(float(fixed_lo["probe_lo_phase_rad"])),
        E_pr_LO_omega=analyses[16]["E_pr_LO_omega"],
        exact_phase_cases=analyses[16]["exact_phase_cases"],
        heterodyne_phase_cases=analyses[16]["heterodyne_phase_cases"],
        exact_phase_order_cube=exact_dft[16],
        heterodyne_phase_order_cube=heterodyne_dft[16],
        exact_linear_lo_phase_order_cube=decomposition_dft["fixed_lo_linear"],
        exact_off_signal_cross_phase_order_cube=decomposition_dft["pump_off_signal_cross"],
        exact_quadratic_phase_order_cube=decomposition_dft["quadratic_signal"],
        phase_orders=orders,
        valid_detector_mask=analyses[16]["valid_detector_mask"],
        exact_map_strength=exact_strength,
        heterodyne_map_strength=heterodyne_strength,
        **{f"exact_{_channel_name(order)}": _channel(exact_dft[16], order) for order in channels},
        **{f"heterodyne_{_channel_name(order)}": _channel(heterodyne_dft[16], order) for order in channels},
        **{f"exact_target_N{n_steps}": values for n_steps, values in n_spectra.items()},
        **legacy_arrays,
    )

    summary_path = write_json(
        data_dir / "summary.json",
        json_safe(
            {
                "conclusions": conclusions,
                "primary_channel_metrics": [
                    row for row in metric_rows if (int(row["pump_order"]), int(row["probe_order"])) in primary
                ],
                "N8_to_N16_convergence": convergence,
                "exact_vs_heterodyne": fits,
                "N2_alias_table": alias_rows,
                "exact_S_0_0_detector_decomposition": decomposition_rows,
            }
        ),
    )

    outputs = {
        "npz": str(npz_path),
        "summary_json": str(summary_path),
        "channel_metrics_csv": str(_write_rows(data_dir / "channel_metrics.csv", metric_rows)),
        "channel_spectra_csv": str(
            _write_channel_spectra(
                data_dir / "channel_spectra.csv",
                energy_eV,
                exact_dft[16],
                heterodyne_dft[16],
                channels,
            )
        ),
        "phase_order_map_csv": str(_write_rows(data_dir / "phase_order_map.csv", map_rows)),
        "convergence_csv": str(_write_rows(data_dir / "N8_to_N16_convergence.csv", convergence)),
        "exact_vs_heterodyne_csv": str(
            _write_rows(
                data_dir / "exact_vs_heterodyne_metrics.csv",
                [{"channel": name, **values} for name, values in fits.items()],
            )
        ),
        "N_comparison_csv": str(_write_rows(data_dir / "target_S_0_1_N_comparison.csv", n_rows)),
        "alias_table_csv": str(_write_rows(data_dir / "N2_alias_table.csv", alias_rows)),
        "exact_detector_decomposition_csv": str(
            _write_rows(data_dir / "exact_S_0_0_detector_decomposition.csv", decomposition_rows)
        ),
        "figures": figures,
    }

    checkpoint_count = sum(1 for _ in source_root.rglob("*.ckp"))
    metadata = {
        "example_name": str(plan["name"]),
        "analysis_only": True,
        "solver_called": False,
        "plan_json": str(plan_path),
        "plan": plan,
        "source_root": str(source_root),
        "source_npz": {
            "N8": str(source8["source_path"].item()),
            "N16": str(source16["source_path"].item()),
        },
        "available_source_checkpoint_count": int(checkpoint_count),
        "interaction_phase_definition": {
            "pump": "phi_pu enters the Hamiltonian pump field",
            "probe": "phi_pr_int enters the Hamiltonian probe field",
        },
        "readout_phase_definition": {
            "probe_LO_phase_rad": float(fixed_lo["probe_lo_phase_rad"]),
            "source": "N16 probe-only E_probe_omega at interaction probe phase index 0",
            "policy": str(fixed_lo["policy"]),
        },
        "detector_formulas": {
            "exact": str(detector["exact_definition"]),
            "heterodyne": str(detector["heterodyne_definition"]),
            "C_rad": float(detector["C_rad"]),
            "absolute_intensity_interpretation": bool(detector["absolute_intensity_interpretation"]),
        },
        "projection": plan["projection"],
        "energy_window_eV": window_eV,
        "valid_detector_point_count": int(np.count_nonzero(analyses[16]["valid_detector_mask"])),
        "channel_metrics": metric_rows,
        "N8_to_N16_convergence": convergence,
        "exact_vs_heterodyne": fits,
        "N2_alias_table": alias_rows,
        "exact_S_0_0_detector_decomposition": decomposition_rows,
        "conclusions": conclusions,
        "old_readout_comparison": {
            "policy": str(plan["metadata"]["old_readout_policy"]),
            "source_directory": str(source_root),
            "copied_arrays": sorted(legacy_arrays),
        },
        "outputs": outputs,
    }
    meta_path = write_json(output_dir / "meta.json", json_safe(metadata))
    print("Fixed-LO detector phase-cycling analysis finished.")
    print(f"output_dir: {output_dir}")
    print(f"source_checkpoint_count: {checkpoint_count}")
    print("solver_called: False")
    print(f"meta_json: {meta_path}")
    return {**metadata, "meta_json": str(meta_path)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-json", type=Path, default=DEFAULT_PLAN_PATH)
    parser.add_argument("--source-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dry_run:
        plan = _load_plan(args.plan_json.resolve())
        source_root = (REPO_ROOT / str(plan["source_root"])).resolve() if args.source_root is None else args.source_root.resolve()
        print(
            json.dumps(
                json_safe(
                    {
                        "analysis_only": True,
                        "solver_called": False,
                        "plan_json": str(args.plan_json.resolve()),
                        "source_root": str(source_root),
                        "output_root": str(args.output_root.resolve()),
                        "fixed_probe_LO_phase_rad": float(plan["fixed_lo"]["probe_lo_phase_rad"]),
                    }
                ),
                indent=2,
            )
        )
        return
    run_analysis(args)


if __name__ == "__main__":
    main()
