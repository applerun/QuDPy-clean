#!/usr/bin/env python3
"""Run two-dimensional pump/probe phase cycling for three selected channels.

The system, pulses, time grid, and delay convention come from
``ta_three_level_phase_cycling_v2_legacy_output_system_maker.py``.  Each phase
case is a complete pump+probe master-equation propagation.  Probe-only
references are propagated once per probe phase.  The projected quantity is

    (A_pump_probe - A_probe_only) / A_probe_only

with ``A = omega * Im[P(omega) * conj(E_probe(omega))]``.  The script stores
P(t), Fourier intermediates, per-case relative responses, and only the phase
channels declared by the JSON plan.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, replace
import importlib.util
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

EXAMPLE_DIR = Path(__file__).resolve().parent
DEFAULT_PLAN_PATH = (
    EXAMPLE_DIR
    / "plan_examples"
    / "ta_three_level_two_dimensional_phase_cycling.json"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "bin"
    / "optical_bloch_plots"
    / "ta_three_level_two_dimensional_phase_cycling"
)

from qudpy_sjh.experiments import (  # noqa: E402
    PHASE_PROJECTION_CONVENTION,
    PHASE_PROJECTION_CONVENTION_VERSION,
    TARGET_PHASE_VECTOR_SEMANTICS,
    PhaseGrid,
    fourier_project_phase_cases,
)
from qudpy_sjh.experiments.ta import TADelayCenters, TASingleDelayPlan  # noqa: E402
from qudpy_sjh.utils.core import ParaNormalizer  # noqa: E402
from qudpy_sjh.utils.serialization import json_safe, write_json  # noqa: E402


def _load_module(path: Path, module_name: str):
    if not path.exists():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_plan(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if int(payload.get("schema_version", 0)) != 1:
        raise ValueError("phase-cycling plan schema_version must be 1.")
    channels = tuple(payload.get("channels", ()))
    names = tuple(str(item.get("name", "")).strip() for item in channels)
    if names != ("S_0_0", "S_0_1", "S_0_2"):
        raise ValueError("This validation must contain only S_0_0, S_0_1, and S_0_2.")
    for item in channels:
        target = item.get("target_phase_vector", {})
        if set(target) != {"pump", "probe"}:
            raise ValueError("Each target_phase_vector must contain pump and probe.")
        if any(float(value) != float(int(value)) for value in target.values()):
            raise ValueError("Phase-order coefficients must be integers.")
    projection = payload.get("projection", {})
    if projection.get("phase_projection_convention") != PHASE_PROJECTION_CONVENTION:
        raise ValueError("The phase-cycling plan requires the canonical exp(+i*m*phi) convention.")
    if int(projection.get("phase_projection_convention_version", 0)) != PHASE_PROJECTION_CONVENTION_VERSION:
        raise ValueError("Unsupported phase-projection convention version.")
    if projection.get("target_phase_vector_semantics") != TARGET_PHASE_VECTOR_SEMANTICS:
        raise ValueError("target_phase_vector must represent the physical phase-order vector m.")
    return payload


def _load_baseline_context(plan: Mapping[str, Any]):
    baseline_path = EXAMPLE_DIR / str(plan["baseline_example"])
    baseline = _load_module(
        baseline_path,
        "ta_two_dimensional_phase_cycling_baseline",
    )
    runner = baseline._load_runner_module()
    legacy = runner._load_module(
        runner.LEGACY_DEMO_PATH,
        "ta_two_dimensional_phase_cycling_legacy",
    )
    smoke_v2 = runner._load_module(
        runner.SMOKE_V2_PATH,
        "ta_two_dimensional_phase_cycling_smoke_v2",
    )
    return baseline_path, baseline, runner, legacy, smoke_v2


def _uniform_phases(n_steps: int) -> tuple[float, ...]:
    steps = int(n_steps)
    if steps < 1:
        raise ValueError("phase-grid sizes must be >= 1.")
    return tuple(2.0 * math.pi * index / steps for index in range(steps))


def _safe_delay_label(delay_fs: float) -> str:
    value = float(delay_fs)
    if np.isclose(value, round(value), rtol=0.0, atol=1.0e-12):
        return str(int(round(value))).replace("-", "m")
    return f"{value:.6g}".replace("-", "m").replace(".", "p")


def _trapezoid(values: np.ndarray, axis: np.ndarray) -> float:
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(values, axis))
    return float(np.trapz(values, axis))


def _assert_axis(name: str, reference: np.ndarray, current: np.ndarray) -> None:
    if reference.shape != current.shape or not np.allclose(
        reference,
        current,
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise ValueError(f"{name} differs between phase cases.")


def _power_readout(
    polarization_omega: np.ndarray,
    probe_field_omega: np.ndarray,
    omega_fs_inv: np.ndarray,
) -> np.ndarray:
    """Return the real probe-heterodyne power-like spectrum."""

    return np.asarray(omega_fs_inv, dtype=float) * np.imag(
        np.asarray(polarization_omega, dtype=np.complex128)
        * np.conjugate(np.asarray(probe_field_omega, dtype=np.complex128))
    )


def _relative_power_response(
    pump_probe: np.ndarray,
    probe_only: np.ndarray,
    *,
    reference_rel_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build phase-case relative responses on a common valid energy mask."""

    pp = np.asarray(pump_probe, dtype=float)
    reference = np.asarray(probe_only, dtype=float)
    if pp.ndim != 3 or reference.ndim != 2 or pp.shape[1:] != reference.shape:
        raise ValueError(
            "pump_probe must be N_pump x N_probe x energy and probe_only "
            "must be N_probe x energy."
        )
    threshold = float(reference_rel_threshold)
    if threshold <= 0.0:
        raise ValueError("reference_rel_threshold must be > 0.")
    reference_scale = np.max(np.abs(reference), axis=1)
    if np.any(reference_scale == 0.0):
        raise ValueError("At least one probe-only power spectrum is identically zero.")
    per_probe_valid = np.abs(reference) > threshold * reference_scale[:, np.newaxis]
    common_valid = np.all(per_probe_valid, axis=0)
    relative = np.full(pp.shape, np.nan, dtype=float)
    denominator = reference[np.newaxis, :, :]
    np.divide(
        pp - denominator,
        denominator,
        out=relative,
        where=per_probe_valid[np.newaxis, :, :],
    )
    relative[:, :, ~common_valid] = np.nan
    return relative, common_valid, reference_scale


def _make_case_plan(
    base_plan,
    *,
    phase_vector: Mapping[str, float],
    case_name: str,
    case_index: int,
    scope: str = "two_dimensional_pump_probe_phase_cycling",
):
    phase_metadata = {
        "phase_case_index": int(case_index),
        "phase_vector": {str(key): float(value) for key, value in phase_vector.items()},
        "scope": str(scope),
    }
    field_metadata = dict(base_plan.field_plan.metadata)
    field_metadata["phase_cycling"] = phase_metadata
    field_plan = replace(
        base_plan.field_plan,
        phase_vector=dict(phase_vector),
        case_name=case_name,
        metadata=field_metadata,
    )
    input_metadata = dict(base_plan.input_metadata)
    input_metadata["phase_cycling"] = phase_metadata
    return replace(
        base_plan,
        field_plan=field_plan,
        case_name=case_name,
        input_metadata=input_metadata,
    )


def _channel_metrics(
    name: str,
    values: np.ndarray,
    energy_eV: np.ndarray,
    *,
    window_eV: tuple[float, float],
) -> dict[str, Any]:
    mask = (energy_eV >= window_eV[0]) & (energy_eV <= window_eV[1])
    if not np.any(mask):
        raise ValueError(f"No spectrum points lie in energy window {window_eV}.")
    energy = np.asarray(energy_eV[mask], dtype=float)
    channel = np.asarray(values[mask], dtype=np.complex128)
    finite = np.isfinite(channel.real) & np.isfinite(channel.imag)
    energy = energy[finite]
    channel = channel[finite]
    if energy.size == 0:
        raise ValueError(f"Channel {name} has no finite points in energy window {window_eV}.")
    magnitude = np.abs(channel)
    real = np.real(channel)
    imag = np.imag(channel)
    peak_index = int(np.argmax(magnitude))
    positive = np.clip(real, 0.0, None)
    negative_magnitude = np.clip(-real, 0.0, None)
    absorption_index = int(np.argmax(positive))
    emission_index = int(np.argmax(negative_magnitude))
    return {
        "channel": name,
        "energy_window_min_eV": float(window_eV[0]),
        "energy_window_max_eV": float(window_eV[1]),
        "finite_point_count": int(energy.size),
        "peak_position_eV": float(energy[peak_index]),
        "maximum_absolute_amplitude": float(magnitude[peak_index]),
        "integrated_absolute_amplitude_eV": _trapezoid(magnitude, energy),
        "maximum_absolute_real_part": float(np.max(np.abs(real))),
        "maximum_absolute_imaginary_part": float(np.max(np.abs(imag))),
        "absorption_region_peak_amplitude": float(positive[absorption_index]),
        "absorption_region_peak_position_eV": float(energy[absorption_index]),
        "absorption_region_integrated_amplitude_eV": _trapezoid(positive, energy),
        "emission_region_peak_magnitude": float(negative_magnitude[emission_index]),
        "emission_region_peak_position_eV": float(energy[emission_index]),
        "emission_region_integrated_magnitude_eV": _trapezoid(negative_magnitude, energy),
    }


def _write_metrics_csv(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def _write_channel_spectra_csv(
    path: Path,
    energy_eV: np.ndarray,
    channels: Mapping[str, np.ndarray],
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    names = tuple(channels)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        header = ["energy_eV"]
        for name in names:
            header.extend((f"{name}_real", f"{name}_imag", f"{name}_abs"))
        writer.writerow(header)
        for index, energy in enumerate(energy_eV):
            row: list[float] = [float(energy)]
            for name in names:
                value = complex(channels[name][index])
                row.extend((float(value.real), float(value.imag), float(abs(value))))
            writer.writerow(row)
    return path


def _write_phase_case_csv(
    path: Path,
    phase_vectors: list[dict[str, float]],
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("case_index", "pump_phase_rad", "probe_phase_rad"))
        for index, vector in enumerate(phase_vectors):
            writer.writerow((index, vector["pump"], vector["probe"]))
    return path


def _plot_channels(
    output_dir: Path,
    energy_eV: np.ndarray,
    channels: Mapping[str, np.ndarray],
    *,
    window_eV: tuple[float, float],
    dpi: int,
    title_suffix: str,
) -> dict[str, Any]:
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    mask = (energy_eV >= window_eV[0]) & (energy_eV <= window_eV[1])
    energy = energy_eV[mask]
    paths: dict[str, Any] = {}

    fig, axes = plt.subplots(len(channels), 2, figsize=(13.0, 10.0), sharex=True)
    for row, (name, raw_values) in enumerate(channels.items()):
        values = np.asarray(raw_values[mask], dtype=np.complex128)
        signed_ax = axes[row, 0]
        magnitude_ax = axes[row, 1]
        signed_ax.plot(energy, np.real(values), color="black", linewidth=1.6, label="real")
        signed_ax.plot(energy, np.imag(values), color="#0072B2", linewidth=1.2, linestyle="--", label="imag")
        signed_ax.axhline(0.0, color="0.5", linewidth=0.8)
        signed_ax.set_title(f"{name} signed spectrum")
        signed_ax.set_ylabel("channel amplitude")
        signed_ax.legend(loc="best")
        magnitude_ax.plot(energy, np.abs(values), color="#D55E00", linewidth=1.6)
        magnitude_ax.set_title(f"{name} absolute magnitude")
        magnitude_ax.set_ylabel("absolute amplitude")
        for ax in (signed_ax, magnitude_ax):
            ax.set_xlim(*window_eV)
            ax.grid(alpha=0.2)
    axes[-1, 0].set_xlabel("Probe photon energy (eV)")
    axes[-1, 1].set_xlabel("Probe photon energy (eV)")
    fig.suptitle(f"Two-dimensional phase-order channels, {title_suffix}")
    fig.tight_layout()
    combined_path = figures_dir / "phase_order_channels.png"
    fig.savefig(combined_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    paths["combined"] = str(combined_path)

    for name, raw_values in channels.items():
        values = np.asarray(raw_values[mask], dtype=np.complex128)
        fig, ax = plt.subplots(figsize=(8.5, 5.2))
        ax.plot(energy, np.real(values), color="black", linewidth=1.8, label="real")
        ax.plot(energy, np.imag(values), color="#0072B2", linewidth=1.3, linestyle="--", label="imag")
        ax.axhline(0.0, color="0.5", linewidth=0.8)
        ax.set(xlim=window_eV, xlabel="Probe photon energy (eV)", ylabel="channel amplitude", title=f"{name} signed spectrum, {title_suffix}")
        ax.grid(alpha=0.2)
        ax.legend(loc="best")
        fig.tight_layout()
        signed_path = figures_dir / f"{name}_signed.png"
        fig.savefig(signed_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8.5, 5.2))
        ax.plot(energy, np.abs(values), color="#D55E00", linewidth=1.8)
        ax.set(xlim=window_eV, xlabel="Probe photon energy (eV)", ylabel="absolute amplitude", title=f"{name} absolute magnitude, {title_suffix}")
        ax.grid(alpha=0.2)
        fig.tight_layout()
        magnitude_path = figures_dir / f"{name}_absolute.png"
        fig.savefig(magnitude_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        paths[name] = {"signed": str(signed_path), "absolute": str(magnitude_path)}
    return paths


def _plot_biased_overlay(
    output_dir: Path,
    energy_eV: np.ndarray,
    phase_responses: np.ndarray,
    phase_average: np.ndarray,
    *,
    window_eV: tuple[float, float],
    dpi: int,
    title_suffix: str,
    delay_fs: float,
) -> Path:
    figures_dir = output_dir / "figures" / "preview"
    figures_dir.mkdir(parents=True, exist_ok=True)
    displayed = (energy_eV >= window_eV[0]) & (energy_eV <= window_eV[1])
    energy = np.asarray(energy_eV[displayed], dtype=float)
    cases = np.asarray(phase_responses[:, :, displayed], dtype=float).reshape(-1, energy.size)
    average = np.real(np.asarray(phase_average[displayed], dtype=np.complex128))

    fig, ax_average = plt.subplots(figsize=(8.6, 5.4))
    ax_cases = ax_average.twinx()
    for index, spectrum in enumerate(cases):
        ax_cases.plot(
            energy,
            spectrum,
            linewidth=0.65,
            color="#D55E00",
            alpha=0.16,
            label="individual phase cases" if index == 0 else None,
        )
    ax_average.plot(energy, average, linewidth=2.2, color="black", label="S_0_0 (phase average)")
    ax_average.axhline(0.0, linewidth=0.8, linestyle="--", color="black", alpha=0.5)
    ax_cases.axhline(0.0, linewidth=0.8, linestyle="--", color="#D55E00", alpha=0.35)
    ax_average.set_title(f"Biased overlay relative readout, {title_suffix}")
    ax_average.set_xlabel("Probe photon energy (eV)")
    ax_average.set_ylabel("S_0_0 relative response", color="black")
    ax_cases.set_ylabel("Individual phase-case relative response", color="#D55E00")
    ax_average.set_xlim(*window_eV)
    ax_average.tick_params(axis="y", labelcolor="black")
    ax_cases.tick_params(axis="y", labelcolor="#D55E00")

    average_finite = average[np.isfinite(average)]
    cases_finite = cases[np.isfinite(cases)]
    if average_finite.size and np.max(np.abs(average_finite)) > 0.0:
        limit = 1.08 * float(np.max(np.abs(average_finite)))
        ax_average.set_ylim(-limit, limit)
    if cases_finite.size and np.max(np.abs(cases_finite)) > 0.0:
        limit = 1.08 * float(np.max(np.abs(cases_finite)))
        ax_cases.set_ylim(-limit, limit)
    lines_average, labels_average = ax_average.get_legend_handles_labels()
    lines_cases, labels_cases = ax_cases.get_legend_handles_labels()
    ax_average.legend(lines_average + lines_cases, labels_average + labels_cases, loc="best")
    fig.tight_layout()
    output_path = figures_dir / f"biased_overlay_lineout_{_safe_delay_label(delay_fs)}_fs.png"
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _write_convergence(
    delay_dir: Path,
    *,
    window_eV: tuple[float, float],
    dpi: int,
) -> dict[str, Any] | None:
    n8_path = delay_dir / "N8xN8" / "data" / "two_dimensional_phase_cycling.npz"
    n16_path = delay_dir / "N16xN16" / "data" / "two_dimensional_phase_cycling.npz"
    if not n8_path.exists() or not n16_path.exists():
        return None
    with np.load(n8_path) as n8, np.load(n16_path) as n16:
        energy = np.asarray(n16["energy_eV"], dtype=float)
        _assert_axis("N8/N16 energy_eV", np.asarray(n8["energy_eV"], dtype=float), energy)
        mask = (energy >= window_eV[0]) & (energy <= window_eV[1])
        rows = []
        channel_data = {}
        for name in ("S_0_0", "S_0_1", "S_0_2"):
            coarse = np.asarray(n8[name], dtype=np.complex128)
            fine = np.asarray(n16[name], dtype=np.complex128)
            finite = (
                mask
                & np.isfinite(coarse.real)
                & np.isfinite(coarse.imag)
                & np.isfinite(fine.real)
                & np.isfinite(fine.imag)
            )
            diff = coarse[finite] - fine[finite]
            fine_window = fine[finite]
            coarse_window = coarse[finite]
            if fine_window.size == 0:
                raise ValueError(f"No common finite N8/N16 points for {name} in {window_eV}.")
            fine_l2 = float(np.sqrt(np.mean(np.abs(fine_window) ** 2)))
            rows.append(
                {
                    "channel": name,
                    "common_finite_point_count": int(fine_window.size),
                    "N8_max_abs": float(np.max(np.abs(coarse_window))),
                    "N16_max_abs": float(np.max(np.abs(fine_window))),
                    "max_abs_difference": float(np.max(np.abs(diff))),
                    "rms_complex_difference": float(np.sqrt(np.mean(np.abs(diff) ** 2))),
                    "relative_rms_to_N16": float(np.sqrt(np.mean(np.abs(diff) ** 2)) / max(fine_l2, np.finfo(float).tiny)),
                }
            )
            channel_data[name] = (coarse, fine, finite)

    data_dir = delay_dir / "convergence"
    data_dir.mkdir(parents=True, exist_ok=True)
    csv_path = _write_metrics_csv(data_dir / "N8_vs_N16.csv", rows)
    json_path = write_json(data_dir / "N8_vs_N16.json", json_safe({"rows": rows, "energy_window_eV": window_eV}))
    fig, axes = plt.subplots(3, 2, figsize=(13.0, 10.0), sharex=True)
    for row, name in enumerate(("S_0_0", "S_0_1", "S_0_2")):
        coarse, fine, finite = channel_data[name]
        channel_energy = energy[finite]
        axes[row, 0].plot(channel_energy, np.abs(coarse[finite]), label="N=8", linewidth=1.5)
        axes[row, 0].plot(channel_energy, np.abs(fine[finite]), label="N=16", linewidth=1.2, linestyle="--")
        axes[row, 0].set_title(f"{name} magnitude convergence")
        axes[row, 0].set_ylabel("absolute amplitude")
        axes[row, 0].legend(loc="best")
        axes[row, 1].plot(channel_energy, np.abs(coarse[finite] - fine[finite]), color="#D55E00", linewidth=1.4)
        axes[row, 1].set_title(f"{name} |N8 - N16|")
        axes[row, 1].set_ylabel("absolute difference")
        for ax in axes[row]:
            ax.set_xlim(*window_eV)
            ax.grid(alpha=0.2)
    axes[-1, 0].set_xlabel("Probe photon energy (eV)")
    axes[-1, 1].set_xlabel("Probe photon energy (eV)")
    fig.tight_layout()
    figure_path = data_dir / "N8_vs_N16.png"
    fig.savefig(figure_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return {"csv": str(csv_path), "json": str(json_path), "figure": str(figure_path), "rows": rows}


def _grid_from_plan(plan: Mapping[str, Any], *, convergence: bool) -> tuple[int, int]:
    key = "convergence_grid" if convergence else "default_grid"
    grid = plan[key]
    return int(grid["n_pump"]), int(grid["n_probe"])


def build_manifest(
    plan_path: Path,
    *,
    convergence: bool,
    output_dir: Path,
) -> dict[str, Any]:
    plan = _load_plan(plan_path)
    n_pump, n_probe = _grid_from_plan(plan, convergence=convergence)
    delay_fs = float(plan["delay_fs"])
    return {
        "plan_json": str(plan_path.resolve()),
        "baseline_example": str((EXAMPLE_DIR / str(plan["baseline_example"])).resolve()),
        "delay_fs": delay_fs,
        "delay_convention": "delay_fs = probe_center_fs - pump_center_fs",
        "n_pump": n_pump,
        "n_probe": n_probe,
        "n_phase_cases": n_pump * n_probe,
        "pump_phases_rad": _uniform_phases(n_pump),
        "probe_phases_rad": _uniform_phases(n_probe),
        "channels": plan["channels"],
        "projection": plan["projection"],
        "output_dir": str(output_dir.resolve()),
    }


def run_validation(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    plan_path = args.plan_json.resolve()
    plan = _load_plan(plan_path)
    n_pump, n_probe = _grid_from_plan(plan, convergence=bool(args.convergence))
    if args.n_pump is not None:
        n_pump = int(args.n_pump)
    if args.n_probe is not None:
        n_probe = int(args.n_probe)
    if n_pump < 1 or n_probe < 1:
        raise ValueError("n_pump and n_probe must be >= 1.")

    delay_fs = float(plan["delay_fs"] if args.delay_fs is None else args.delay_fs)
    delay_label = _safe_delay_label(delay_fs)
    delay_dir = args.output_dir.resolve() / f"delay_{delay_label}_fs"
    run_dir = delay_dir / f"N{n_pump}xN{n_probe}"
    data_dir = run_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    baseline_path, baseline, runner, legacy, smoke_v2 = _load_baseline_context(plan)
    config = replace(
        legacy.DemoConfig(),
        use_checkpoints=not bool(args.no_checkpoints),
        force_run=bool(args.force_run),
    )
    pump, probe = smoke_v2._make_pulses(config)
    base_params, builder_metadata = baseline._build_system_maker_base_params(
        legacy,
        smoke_v2,
        config,
        probe,
    )
    readout = smoke_v2._make_readout(config)
    ta_plan = TASingleDelayPlan(
        base_params=base_params,
        pump=pump,
        probe=probe,
        delay=TADelayCenters(delay_fs=delay_fs, probe_center_fs=float(config.probe_center_fs)),
        normalizer=ParaNormalizer(),
        readout=readout,
        case_name=f"two_dimensional_phase_cycling_delay_{delay_label}_fs",
        metadata={"plan_json": str(plan_path)},
    )
    base_plan = ta_plan.make_pump_probe_plan()
    probe_only_base_plan = ta_plan.make_probe_only_plan()

    pump_phases = _uniform_phases(n_pump)
    probe_phases = _uniform_phases(n_probe)
    phase_grid = PhaseGrid({"pump": pump_phases, "probe": probe_phases})
    phase_vectors = list(phase_grid.iter_phase_vectors())
    print(
        f"[2d-phase-cycling] delay={delay_fs:g} fs, "
        f"grid={n_pump}x{n_probe}, cases={len(phase_vectors)}"
    )

    polarization_rows: list[np.ndarray] = []
    p_omega_rows: list[np.ndarray] = []
    e_omega_rows: list[np.ndarray] = []
    p_over_e_rows: list[np.ndarray] = []
    energy_eV: np.ndarray | None = None
    omega_fs_inv: np.ndarray | None = None
    time_fs: np.ndarray | None = None
    max_trace_error = 0.0
    max_hermiticity_error = 0.0
    case_timings_s: list[float] = []
    probe_only_timings_s: list[float] = []

    for index, phase_vector in enumerate(phase_vectors):
        pump_index = index // n_probe
        probe_index = index % n_probe
        case_key = f"pu_{pump_index:03d}_pr_{probe_index:03d}_delay_{delay_label}_fs"
        case_name = f"two_dimensional_phase_cycling_{case_key}"
        case_plan = _make_case_plan(
            base_plan,
            phase_vector=phase_vector,
            case_name=case_name,
            case_index=index,
        )
        case_started = time.perf_counter()
        result = runner._execute_with_checkpoint(
            case_plan,
            output_dir=run_dir,
            case_key=case_key,
            config=config,
        )
        case_timings_s.append(time.perf_counter() - case_started)
        if result.readout is None or result.readout.spectrum is None:
            raise ValueError("Each phase case requires an absorption readout spectrum.")
        spectrum = result.readout.spectrum
        required = ("absorption", "energy_eV", "omega_fs_inv", "P_omega", "E_omega", "P_over_E")
        missing = [key for key in required if key not in spectrum]
        if missing:
            raise KeyError(f"Phase-case spectrum is missing: {missing}")
        local_energy = np.asarray(spectrum["energy_eV"], dtype=float)
        local_omega = np.asarray(spectrum["omega_fs_inv"], dtype=float)
        local_time = np.asarray(result.readout.time_fs, dtype=float)
        if energy_eV is None:
            energy_eV = local_energy
            omega_fs_inv = local_omega
            time_fs = local_time
        else:
            assert omega_fs_inv is not None and time_fs is not None
            _assert_axis("energy_eV", energy_eV, local_energy)
            _assert_axis("omega_fs_inv", omega_fs_inv, local_omega)
            _assert_axis("time_fs", time_fs, local_time)
        polarization_rows.append(np.asarray(result.readout.polarization_C_per_m2))
        p_omega_rows.append(np.asarray(spectrum["P_omega"], dtype=np.complex128))
        e_omega_rows.append(np.asarray(spectrum["E_omega"], dtype=np.complex128))
        p_over_e_rows.append(np.asarray(spectrum["P_over_E"], dtype=np.complex128))
        max_trace_error = max(max_trace_error, float(result.dynamics_result.max_trace_error()))
        max_hermiticity_error = max(max_hermiticity_error, float(result.dynamics_result.max_hermiticity_error()))
        print(
            f"  case {index + 1:03d}/{len(phase_vectors):03d}: "
            f"phi_pu={phase_vector['pump']:.6f}, phi_pr={phase_vector['probe']:.6f}, "
            f"wall={case_timings_s[-1]:.2f}s"
        )

    assert energy_eV is not None and omega_fs_inv is not None and time_fs is not None
    polarization_t = np.stack(polarization_rows, axis=0).reshape(n_pump, n_probe, time_fs.size)
    p_omega = np.stack(p_omega_rows, axis=0).reshape(n_pump, n_probe, energy_eV.size)
    e_omega = np.stack(e_omega_rows, axis=0).reshape(n_pump, n_probe, energy_eV.size)
    p_over_e = np.stack(p_over_e_rows, axis=0).reshape(n_pump, n_probe, energy_eV.size)

    probe_only_polarization_rows: list[np.ndarray] = []
    probe_only_p_omega_rows: list[np.ndarray] = []
    probe_only_e_omega_rows: list[np.ndarray] = []
    for probe_index, probe_phase in enumerate(probe_phases):
        phase_vector = {"probe": float(probe_phase)}
        case_key = f"probe_only_pr_{probe_index:03d}_delay_{delay_label}_fs"
        case_name = f"two_dimensional_phase_cycling_{case_key}"
        case_plan = _make_case_plan(
            probe_only_base_plan,
            phase_vector=phase_vector,
            case_name=case_name,
            case_index=probe_index,
            scope="two_dimensional_probe_only_phase_reference",
        )
        case_started = time.perf_counter()
        result = runner._execute_with_checkpoint(
            case_plan,
            output_dir=run_dir,
            case_key=case_key,
            config=config,
        )
        probe_only_timings_s.append(time.perf_counter() - case_started)
        if result.readout is None or result.readout.spectrum is None:
            raise ValueError("Each probe-only phase requires an absorption readout spectrum.")
        spectrum = result.readout.spectrum
        required = ("energy_eV", "omega_fs_inv", "P_omega", "E_omega")
        missing = [key for key in required if key not in spectrum]
        if missing:
            raise KeyError(f"Probe-only phase spectrum is missing: {missing}")
        _assert_axis("probe-only energy_eV", energy_eV, np.asarray(spectrum["energy_eV"], dtype=float))
        _assert_axis("probe-only omega_fs_inv", omega_fs_inv, np.asarray(spectrum["omega_fs_inv"], dtype=float))
        _assert_axis("probe-only time_fs", time_fs, np.asarray(result.readout.time_fs, dtype=float))
        probe_only_polarization_rows.append(np.asarray(result.readout.polarization_C_per_m2))
        probe_only_p_omega_rows.append(np.asarray(spectrum["P_omega"], dtype=np.complex128))
        probe_only_e_omega_rows.append(np.asarray(spectrum["E_omega"], dtype=np.complex128))
        max_trace_error = max(max_trace_error, float(result.dynamics_result.max_trace_error()))
        max_hermiticity_error = max(max_hermiticity_error, float(result.dynamics_result.max_hermiticity_error()))
        print(
            f"  probe reference {probe_index + 1:03d}/{n_probe:03d}: "
            f"phi_pr={probe_phase:.6f}, wall={probe_only_timings_s[-1]:.2f}s"
        )

    probe_only_polarization_t = np.stack(probe_only_polarization_rows, axis=0)
    probe_only_p_omega = np.stack(probe_only_p_omega_rows, axis=0)
    probe_only_e_omega = np.stack(probe_only_e_omega_rows, axis=0)
    pump_probe_power = _power_readout(p_omega, e_omega, omega_fs_inv)
    probe_only_power = _power_readout(probe_only_p_omega, probe_only_e_omega, omega_fs_inv)
    reference_rel_threshold = float(plan["analysis"]["reference_rel_threshold"])
    phase_spectra, valid_reference_mask, probe_only_reference_scale = _relative_power_response(
        pump_probe_power,
        probe_only_power,
        reference_rel_threshold=reference_rel_threshold,
    )
    flat_spectra = phase_spectra.reshape(n_pump * n_probe, energy_eV.size)

    projection_normalize = bool(plan["projection"]["normalize"])
    channels: dict[str, np.ndarray] = {}
    for item in plan["channels"]:
        name = str(item["name"])
        target = {str(key): int(value) for key, value in item["target_phase_vector"].items()}
        channels[name] = fourier_project_phase_cases(
            flat_spectra,
            phase_vectors,
            target,
            phase_axis=0,
            normalize=projection_normalize,
        )

    window = tuple(float(value) for value in plan["analysis"]["energy_window_eV"])
    if len(window) != 2:
        raise ValueError("analysis.energy_window_eV must contain two values.")
    metrics = [
        _channel_metrics(name, values, energy_eV, window_eV=(window[0], window[1]))
        for name, values in channels.items()
    ]
    by_name = {row["channel"]: row for row in metrics}
    reference_max = float(by_name["S_0_1"]["maximum_absolute_amplitude"])
    ratios = {
        "max_abs_S_0_0_over_S_0_1": float(by_name["S_0_0"]["maximum_absolute_amplitude"] / max(reference_max, np.finfo(float).tiny)),
        "max_abs_S_0_2_over_S_0_1": float(by_name["S_0_2"]["maximum_absolute_amplitude"] / max(reference_max, np.finfo(float).tiny)),
    }

    npz_path = data_dir / "two_dimensional_phase_cycling.npz"
    np.savez_compressed(
        npz_path,
        time_fs=time_fs,
        energy_eV=energy_eV,
        omega_fs_inv=omega_fs_inv,
        pump_phases_rad=np.asarray(pump_phases),
        probe_phases_rad=np.asarray(probe_phases),
        phase_spectra=phase_spectra,
        relative_response_phase_cases=phase_spectra,
        valid_reference_mask=valid_reference_mask,
        pump_probe_power_readout=pump_probe_power,
        probe_only_power_readout=probe_only_power,
        probe_only_reference_scale=probe_only_reference_scale,
        polarization_t_C_per_m2=polarization_t,
        probe_only_polarization_t_C_per_m2=probe_only_polarization_t,
        P_omega=p_omega,
        E_probe_omega=e_omega,
        P_over_E_probe=p_over_e,
        probe_only_P_omega=probe_only_p_omega,
        probe_only_E_probe_omega=probe_only_e_omega,
        **channels,
    )
    metrics_csv = _write_metrics_csv(data_dir / "channel_metrics.csv", metrics)
    spectra_csv = _write_channel_spectra_csv(data_dir / "channel_spectra.csv", energy_eV, channels)
    cases_csv = _write_phase_case_csv(data_dir / "phase_cases.csv", phase_vectors)
    figures = _plot_channels(
        run_dir,
        energy_eV,
        channels,
        window_eV=(window[0], window[1]),
        dpi=int(config.figure_dpi),
        title_suffix=f"Npu={n_pump}, Npr={n_probe}, delay={delay_fs:g} fs",
    )
    figures["biased_overlay"] = str(
        _plot_biased_overlay(
            run_dir,
            energy_eV,
            phase_spectra,
            channels["S_0_0"],
            window_eV=(window[0], window[1]),
            dpi=int(config.figure_dpi),
            title_suffix=f"Npu={n_pump}, Npr={n_probe}, delay={delay_fs:g} fs",
            delay_fs=delay_fs,
        )
    )

    elapsed_s = time.perf_counter() - started
    metadata = {
        "example_name": "ta_three_level_two_dimensional_phase_cycling",
        "plan_json": str(plan_path),
        "plan": plan,
        "baseline_example": str(baseline_path),
        "delay": ta_plan.delay.to_dict(),
        "phase_grid": phase_grid.to_dict(),
        "n_pump": n_pump,
        "n_probe": n_probe,
        "n_phase_cases": len(phase_vectors),
        "phase_projection_convention": PHASE_PROJECTION_CONVENTION,
        "phase_projection_convention_version": PHASE_PROJECTION_CONVENTION_VERSION,
        "target_phase_vector_semantics": TARGET_PHASE_VECTOR_SEMANTICS,
        "projection_normalize": projection_normalize,
        "projection_definition": plan["projection"]["definition"],
        "readout_definition": (
            "relative_response = (A_pump_probe - A_probe_only) / A_probe_only; "
            "A = omega * Im[P(omega) * conj(E_probe(omega))]"
        ),
        "readout_field_phase_policy": (
            "E_probe is selected from each concrete phase case; probe-only is propagated "
            "once for every probe phase"
        ),
        "reference_rel_threshold": reference_rel_threshold,
        "valid_reference_point_count": int(np.count_nonzero(valid_reference_mask)),
        "invalid_reference_point_count": int(valid_reference_mask.size - np.count_nonzero(valid_reference_mask)),
        "analysis": plan["analysis"],
        "metrics": metrics,
        "ratios": ratios,
        "numerical_diagnostics": {
            "max_trace_error": max_trace_error,
            "max_hermiticity_error": max_hermiticity_error,
            "elapsed_s": elapsed_s,
            "pump_probe_case_count": len(case_timings_s),
            "pump_probe_mean_case_s": float(np.mean(case_timings_s)),
            "pump_probe_max_case_s": float(np.max(case_timings_s)),
            "probe_only_case_count": len(probe_only_timings_s),
            "probe_only_mean_case_s": float(np.mean(probe_only_timings_s)),
            "probe_only_max_case_s": float(np.max(probe_only_timings_s)),
        },
        "baseline_config": asdict(config),
        "base_params_builder": builder_metadata,
        "outputs": {
            "npz": str(npz_path),
            "metrics_csv": str(metrics_csv),
            "spectra_csv": str(spectra_csv),
            "phase_cases_csv": str(cases_csv),
            "figures": figures,
        },
    }
    meta_path = write_json(run_dir / "meta.json", json_safe(metadata))
    convergence = _write_convergence(
        delay_dir,
        window_eV=(window[0], window[1]),
        dpi=int(config.figure_dpi),
    )
    print("Two-dimensional phase cycling finished.")
    print(f"output_dir: {run_dir}")
    print(f"elapsed_s: {elapsed_s:.3f}")
    print(f"ratios: {ratios}")
    print(f"meta_json: {meta_path}")
    if convergence is not None:
        print(f"convergence_json: {convergence['json']}")
    return {**metadata, "meta_json": str(meta_path), "convergence": convergence}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-json", type=Path, default=DEFAULT_PLAN_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--delay-fs", type=float, default=None)
    parser.add_argument("--n-pump", type=int, default=None)
    parser.add_argument("--n-probe", type=int, default=None)
    parser.add_argument("--convergence", action="store_true", help="Use the plan's N=16 convergence grid.")
    parser.add_argument("--force-run", action="store_true")
    parser.add_argument("--no-checkpoints", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dry_run:
        print(
            json.dumps(
                json_safe(
                    build_manifest(
                        args.plan_json.resolve(),
                        convergence=bool(args.convergence),
                        output_dir=args.output_dir,
                    )
                ),
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    run_validation(args)


if __name__ == "__main__":
    main()
