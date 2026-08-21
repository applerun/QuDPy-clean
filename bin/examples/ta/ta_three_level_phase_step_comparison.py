#!/usr/bin/env python3
"""Compare N=2, N=3, N=4, N=8, and N=16 pump-phase averages.

The simulation, system, pulse, and readout parameters come from
``ta_three_level_phase_cycling_v2_legacy_output_system_maker.py``.  The only
scheme-dependent parameter is the number of uniformly spaced pump phases.

The inherited delay convention is:

    pump_center_fs = probe_center_fs - delay_fs

so positive delay means that the pump arrives before the probe.
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
from typing import Any, Callable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

BASELINE_EXAMPLE_PATH = (
    Path(__file__).resolve().parent
    / "ta_three_level_phase_cycling_v2_legacy_output_system_maker.py"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "bin"
    / "optical_bloch_plots"
    / "ta_three_level_phase_step_comparison"
)
DELAYS_FS = (-100.0, 0.0, 100.0)
PHASE_STEP_COUNTS = (2, 3, 4, 8, 16)
SCHEME_COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00")

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


def phase_values_rad(n_steps: int) -> tuple[float, ...]:
    steps = int(n_steps)
    if steps not in PHASE_STEP_COUNTS:
        raise ValueError(f"n_steps must be one of {PHASE_STEP_COUNTS}; got {steps}.")
    return tuple(2.0 * math.pi * index / steps for index in range(steps))


def build_scheme_config(config: Any, *, n_steps: int) -> Any:
    """Select the requested delays and phase grid without changing physics."""

    phases = phase_values_rad(n_steps)
    return replace(
        config,
        probe_delays_fs=DELAYS_FS,
        quick_probe_delays_fs=DELAYS_FS,
        preview_delays_fs=DELAYS_FS,
        pump_phase_cases_rad=phases,
    )


def _load_baseline_context() -> tuple[Any, Any, Any]:
    baseline = _load_module(
        BASELINE_EXAMPLE_PATH,
        "ta_three_level_phase_step_comparison_baseline",
    )
    runner = baseline._load_runner_module()
    legacy = runner._load_module(
        runner.LEGACY_DEMO_PATH,
        "ta_three_level_phase_step_comparison_legacy_config",
    )
    return baseline, runner, legacy


def build_run_manifest() -> dict[str, Any]:
    baseline, runner, legacy = _load_baseline_context()
    base_config = legacy.DemoConfig()
    schemes = []
    shared_snapshot = None
    for n_steps in PHASE_STEP_COUNTS:
        config = build_scheme_config(base_config, n_steps=n_steps)
        snapshot = asdict(config)
        snapshot.pop("pump_phase_cases_rad")
        if shared_snapshot is None:
            shared_snapshot = snapshot
        elif json_safe(snapshot) != json_safe(shared_snapshot):
            raise AssertionError("Phase-step schemes differ outside pump_phase_cases_rad.")
        schemes.append(
            {
                "n_steps": n_steps,
                "phase_values_rad": phase_values_rad(n_steps),
                "output_subdir": f"N{n_steps}",
            }
        )

    probe_center_fs = float(base_config.probe_center_fs)
    return {
        "example_name": "ta_three_level_phase_step_comparison",
        "baseline_example": BASELINE_EXAMPLE_PATH,
        "legacy_output_runner": runner.__file__,
        "delay_convention": "pump_center_fs = probe_center_fs - delay_fs",
        "positive_delay": "pump arrives before probe",
        "delays_fs": DELAYS_FS,
        "probe_center_fs": probe_center_fs,
        "pump_centers_fs": [probe_center_fs - delay for delay in DELAYS_FS],
        "n_delay_phase_schemes": len(DELAYS_FS) * len(PHASE_STEP_COUNTS),
        "schemes": schemes,
        "shared_config_except_phase_grid": shared_snapshot,
        "system_maker_builder": baseline._build_system_maker_base_params.__name__,
    }


def _runner_args(args: argparse.Namespace, output_dir: Path) -> argparse.Namespace:
    return argparse.Namespace(
        output_dir=output_dir,
        force_run=bool(args.force_run or args.force),
        no_checkpoints=bool(args.no_checkpoints),
        quick=False,
        wavelength=bool(args.wavelength),
        max_delays=None,
    )


def _config_transform(n_steps: int) -> Callable[[Any], Any]:
    return lambda config: build_scheme_config(config, n_steps=n_steps)


def _load_scheme_arrays(path: Path, *, n_steps: int) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        required = {
            "delays_fs",
            "energy_eV",
            "omega_fs_inv",
            "TA_phase_cases",
            "TA_phase_avg",
            "phase_values_rad",
        }
        missing = sorted(required.difference(payload.files))
        if missing:
            raise KeyError(f"N={n_steps} NPZ is missing fields: {missing}")
        result = {name: np.asarray(payload[name]) for name in required}

    expected_phases = np.asarray(phase_values_rad(n_steps), dtype=float)
    if not np.allclose(
        result["delays_fs"], np.asarray(DELAYS_FS), rtol=0.0, atol=1.0e-12
    ):
        raise ValueError(f"N={n_steps} delay axis does not match {DELAYS_FS}.")
    if not np.allclose(
        result["phase_values_rad"], expected_phases, rtol=0.0, atol=1.0e-12
    ):
        raise ValueError(f"N={n_steps} phase values do not match the uniform grid.")
    expected_shape = (
        n_steps,
        len(DELAYS_FS),
        result["energy_eV"].size,
    )
    if result["TA_phase_cases"].shape != expected_shape:
        raise ValueError(
            f"N={n_steps} phase stack has shape {result['TA_phase_cases'].shape}; "
            f"expected {expected_shape}."
        )
    if result["TA_phase_avg"].shape != expected_shape[1:]:
        raise ValueError(
            f"N={n_steps} phase average has shape {result['TA_phase_avg'].shape}; "
            f"expected {expected_shape[1:]}."
        )
    return result


def _assert_shared_axes(arrays_by_steps: dict[int, dict[str, np.ndarray]]) -> None:
    reference_steps = max(PHASE_STEP_COUNTS)
    reference = arrays_by_steps[reference_steps]
    for n_steps, arrays in arrays_by_steps.items():
        for axis_name in ("delays_fs", "energy_eV", "omega_fs_inv"):
            current = np.asarray(arrays[axis_name], dtype=float)
            expected = np.asarray(reference[axis_name], dtype=float)
            if current.shape != expected.shape or not np.allclose(
                current, expected, rtol=0.0, atol=1.0e-12
            ):
                raise ValueError(
                    f"N={n_steps} {axis_name} axis differs from N={reference_steps}."
                )


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> Path:
    if not rows:
        raise ValueError("rows must not be empty.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def _comparison_rows(
    arrays_by_steps: dict[int, dict[str, np.ndarray]],
) -> list[dict[str, Any]]:
    reference_steps = max(PHASE_STEP_COUNTS)
    energy = np.asarray(arrays_by_steps[reference_steps]["energy_eV"], dtype=float)
    rows = []
    for delay_index, delay_fs in enumerate(DELAYS_FS):
        for energy_index, energy_eV in enumerate(energy):
            values = {
                n_steps: float(arrays["TA_phase_avg"][delay_index, energy_index])
                for n_steps, arrays in arrays_by_steps.items()
            }
            row = {
                "delay_fs": delay_fs,
                "pump_center_fs": -delay_fs,
                "energy_index": energy_index,
                "energy_eV": float(energy_eV),
            }
            row.update(
                {
                    f"TA_phase_avg_N{n_steps}": values[n_steps]
                    for n_steps in PHASE_STEP_COUNTS
                }
            )
            row.update(
                {
                    f"N{n_steps}_minus_N{reference_steps}": (
                        values[n_steps] - values[reference_steps]
                    )
                    for n_steps in PHASE_STEP_COUNTS
                    if n_steps != reference_steps
                }
            )
            rows.append(row)
    return rows


def _stats_rows(
    arrays_by_steps: dict[int, dict[str, np.ndarray]],
) -> list[dict[str, Any]]:
    reference_steps = max(PHASE_STEP_COUNTS)
    reference = np.asarray(
        arrays_by_steps[reference_steps]["TA_phase_avg"], dtype=float
    )
    rows = []
    for n_steps, arrays in arrays_by_steps.items():
        values = np.asarray(arrays["TA_phase_avg"], dtype=float)
        for delay_index, delay_fs in enumerate(DELAYS_FS):
            spectrum = values[delay_index]
            reference_spectrum = reference[delay_index]
            difference = spectrum - reference_spectrum
            reference_norm = float(np.linalg.norm(reference_spectrum))
            rows.append(
                {
                    "n_phase_steps": n_steps,
                    "reference_n_phase_steps": reference_steps,
                    "delay_fs": delay_fs,
                    "maxabs": float(np.max(np.abs(spectrum))),
                    "rms": float(np.sqrt(np.mean(spectrum**2))),
                    "maxabs_difference_from_reference": float(
                        np.max(np.abs(difference))
                    ),
                    "relative_l2_difference_from_reference": (
                        None
                        if reference_norm == 0.0
                        else float(np.linalg.norm(difference) / reference_norm)
                    ),
                }
            )
    return rows


def _save_comparison_npz(
    path: Path,
    arrays_by_steps: dict[int, dict[str, np.ndarray]],
) -> Path:
    reference = arrays_by_steps[max(PHASE_STEP_COUNTS)]
    payload: dict[str, Any] = {
        "delays_fs": np.asarray(DELAYS_FS, dtype=float),
        "probe_center_fs": np.asarray(0.0, dtype=float),
        "pump_centers_fs": -np.asarray(DELAYS_FS, dtype=float),
        "energy_eV": reference["energy_eV"],
        "omega_fs_inv": reference["omega_fs_inv"],
    }
    for n_steps, arrays in arrays_by_steps.items():
        payload[f"phase_values_rad_N{n_steps}"] = arrays["phase_values_rad"]
        payload[f"TA_phase_cases_N{n_steps}"] = arrays["TA_phase_cases"]
        payload[f"TA_phase_avg_N{n_steps}"] = arrays["TA_phase_avg"]
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)
    return path


def _plot_map_comparison(
    path: Path,
    arrays_by_steps: dict[int, dict[str, np.ndarray]],
    *,
    energy_xlim_eV: tuple[float, float],
    dpi: int,
) -> Path:
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    energy = np.asarray(
        arrays_by_steps[max(PHASE_STEP_COUNTS)]["energy_eV"], dtype=float
    )
    maps = [np.asarray(arrays_by_steps[n]["TA_phase_avg"], dtype=float) for n in PHASE_STEP_COUNTS]
    displayed = (energy >= energy_xlim_eV[0]) & (energy <= energy_xlim_eV[1])
    finite = np.concatenate(
        [
            np.abs(values[:, displayed][np.isfinite(values[:, displayed])])
            for values in maps
        ]
    )
    vlim = float(np.percentile(finite, 99.0)) if finite.size else 1.0
    if vlim <= 0.0:
        vlim = 1.0

    fig, axes = plt.subplots(2, 3, figsize=(15.0, 8.2), sharex=True, sharey=True)
    axes_flat = axes.ravel()
    mesh = None
    for ax, n_steps, values in zip(axes_flat, PHASE_STEP_COUNTS, maps):
        mesh = ax.pcolormesh(
            energy,
            np.asarray(DELAYS_FS),
            values,
            shading="auto",
            cmap="RdBu_r",
            norm=TwoSlopeNorm(vmin=-vlim, vcenter=0.0, vmax=vlim),
        )
        ax.set_title(f"N={n_steps} phase average")
        ax.set_xlabel("Probe photon energy (eV)")
        ax.set_xlim(*energy_xlim_eV)
        ax.set_yticks(DELAYS_FS)
    for ax in axes[:, 0]:
        ax.set_ylabel("Pump-probe delay (fs)")
    for ax in axes_flat[len(PHASE_STEP_COUNTS):]:
        ax.set_visible(False)
    fig.subplots_adjust(
        left=0.07,
        right=0.86,
        bottom=0.09,
        top=0.89,
        wspace=0.16,
        hspace=0.28,
    )
    if mesh is not None:
        colorbar_ax = fig.add_axes([0.88, 0.15, 0.015, 0.68])
        fig.colorbar(
            mesh,
            cax=colorbar_ax,
            label="S_TA (arb., shared scale)",
            format="%.1e",
        )
    fig.suptitle("Pump phase-step comparison")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def _plot_delay_lineouts(
    output_dir: Path,
    arrays_by_steps: dict[int, dict[str, np.ndarray]],
    *,
    energy_xlim_eV: tuple[float, float],
    dpi: int,
) -> list[Path]:
    import matplotlib.pyplot as plt

    energy = np.asarray(
        arrays_by_steps[max(PHASE_STEP_COUNTS)]["energy_eV"], dtype=float
    )
    displayed = (energy >= energy_xlim_eV[0]) & (energy <= energy_xlim_eV[1])
    paths = []
    for delay_index, delay_fs in enumerate(DELAYS_FS):
        fig, ax = plt.subplots(figsize=(8.0, 5.0))
        displayed_spectra = []
        for n_steps, color in zip(PHASE_STEP_COUNTS, SCHEME_COLORS):
            spectrum = np.asarray(arrays_by_steps[n_steps]["TA_phase_avg"], dtype=float)[delay_index]
            ax.plot(energy, spectrum, linewidth=1.6, color=color, label=f"N={n_steps}")
            displayed_spectra.append(spectrum[displayed])
        ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--", alpha=0.55)
        ax.set_xlim(*energy_xlim_eV)
        finite = np.concatenate(
            [values[np.isfinite(values)] for values in displayed_spectra]
        )
        if finite.size:
            y_limit = float(np.max(np.abs(finite)))
            if y_limit > 0.0:
                ax.set_ylim(-1.08 * y_limit, 1.08 * y_limit)
        ax.set_xlabel("Probe photon energy (eV)")
        ax.set_ylabel("Phase-averaged S_TA (arb.)")
        ax.set_title(f"Phase-step comparison at delay = {delay_fs:g} fs")
        ax.legend()
        fig.tight_layout()
        label = str(int(delay_fs)).replace("-", "m")
        path = output_dir / f"phase_step_comparison_delay_{label}_fs.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=dpi)
        plt.close(fig)
        paths.append(path)
    return paths


def _phase_fraction_label(index: int, n_steps: int) -> str:
    if index == 0:
        return "0"
    numerator = 2 * int(index)
    denominator = int(n_steps)
    divisor = math.gcd(numerator, denominator)
    numerator //= divisor
    denominator //= divisor
    if denominator == 1:
        return "pi" if numerator == 1 else f"{numerator}pi"
    prefix = "" if numerator == 1 else str(numerator)
    return f"{prefix}pi/{denominator}"


def _plot_biased_overlays(
    output_dir: Path,
    arrays_by_steps: dict[int, dict[str, np.ndarray]],
    *,
    energy_xlim_eV: tuple[float, float],
    dpi: int,
) -> dict[str, list[Path]]:
    import matplotlib.pyplot as plt

    energy = np.asarray(
        arrays_by_steps[max(PHASE_STEP_COUNTS)]["energy_eV"], dtype=float
    )
    displayed = (energy >= energy_xlim_eV[0]) & (energy <= energy_xlim_eV[1])
    line_styles = ("-", "--", "-.", ":")
    paths_by_scheme: dict[str, list[Path]] = {}

    for n_steps in PHASE_STEP_COUNTS:
        phase_cases = np.asarray(
            arrays_by_steps[n_steps]["TA_phase_cases"], dtype=float
        )
        phase_avg = np.asarray(
            arrays_by_steps[n_steps]["TA_phase_avg"], dtype=float
        )
        scheme_paths = []
        for delay_index, delay_fs in enumerate(DELAYS_FS):
            spectra = phase_cases[:, delay_index, :]
            avg_spectrum = phase_avg[delay_index]
            fig, ax_avg = plt.subplots(figsize=(8.6, 5.4))
            ax_phase = ax_avg.twinx()

            for phase_index, spectrum in enumerate(spectra):
                phase_label = _phase_fraction_label(phase_index, n_steps)
                ax_phase.plot(
                    energy,
                    spectrum,
                    linestyle=line_styles[phase_index % len(line_styles)],
                    linewidth=1.0,
                    color="red",
                    alpha=0.62,
                    label=f"phase {phase_label}",
                )
            ax_avg.plot(
                energy,
                avg_spectrum,
                linewidth=2.2,
                color="black",
                label="phase average",
            )
            ax_avg.axhline(
                0.0, linewidth=0.8, linestyle="--", color="black", alpha=0.5
            )
            ax_phase.axhline(
                0.0, linewidth=0.8, linestyle="--", color="red", alpha=0.35
            )
            ax_avg.set_title(
                f"N={n_steps} biased overlay TA lineout at delay = {delay_fs:g} fs"
            )
            ax_avg.set_xlabel("Probe photon energy (eV)")
            ax_avg.set_ylabel("Phase-averaged S_TA", color="black")
            ax_phase.set_ylabel("Single-phase S_TA", color="red")
            ax_avg.set_xlim(*energy_xlim_eV)
            ax_avg.tick_params(axis="y", labelcolor="black")
            ax_phase.tick_params(axis="y", labelcolor="red")

            avg_finite = avg_spectrum[displayed]
            avg_finite = avg_finite[np.isfinite(avg_finite)]
            phase_finite = spectra[:, displayed]
            phase_finite = phase_finite[np.isfinite(phase_finite)]
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
                    bbox={
                        "facecolor": "white",
                        "alpha": 0.72,
                        "edgecolor": "none",
                        "pad": 2,
                    },
                )
            lines_avg, labels_avg = ax_avg.get_legend_handles_labels()
            lines_phase, labels_phase = ax_phase.get_legend_handles_labels()
            ax_avg.legend(
                lines_avg + lines_phase,
                labels_avg + labels_phase,
                fontsize=6.5 if n_steps >= 8 else 8,
                ncol=2 if n_steps >= 8 else 1,
                loc="lower right",
            )
            fig.tight_layout()
            delay_label = str(int(delay_fs)).replace("-", "m")
            path = (
                output_dir
                / f"N{n_steps}"
                / "figures"
                / "preview"
                / f"biased_overlay_lineout_{delay_label}_fs.png"
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, dpi=dpi)
            plt.close(fig)
            scheme_paths.append(path)
        paths_by_scheme[f"N{n_steps}"] = scheme_paths
    return paths_by_scheme


def run_comparison(args: argparse.Namespace) -> dict[str, Any]:
    manifest = build_run_manifest()
    if args.dry_run:
        print(json.dumps(json_safe(manifest), indent=2, ensure_ascii=False))
        return manifest

    baseline, runner, legacy = _load_baseline_context()
    output_dir = args.output_dir.resolve()
    subruns: dict[int, dict[str, Any]] = {}
    arrays_by_steps: dict[int, dict[str, np.ndarray]] = {}

    for n_steps in PHASE_STEP_COUNTS:
        scheme_output_dir = output_dir / f"N{n_steps}"
        print(f"[phase-step-comparison] running N={n_steps}: {scheme_output_dir}")
        result = runner.run_v2_legacy_output(
            _runner_args(args, scheme_output_dir),
            base_params_builder=baseline._build_system_maker_base_params,
            config_transform=_config_transform(n_steps),
            example_name=f"ta_three_level_phase_step_comparison_N{n_steps}",
            workflow_extra={
                "phase_step_comparison": True,
                "n_phase_steps": n_steps,
                "baseline_example": BASELINE_EXAMPLE_PATH,
                "delay_convention": "pump_center_fs = probe_center_fs - delay_fs",
            },
            include_previews=False,
        )
        subruns[n_steps] = result
        arrays_by_steps[n_steps] = _load_scheme_arrays(
            Path(result["data_npz"]), n_steps=n_steps
        )

    _assert_shared_axes(arrays_by_steps)
    data_dir = output_dir / "data"
    plot_dir = output_dir / "figures" / "plot"
    preview_dir = output_dir / "figures" / "preview"
    comparison_csv = _write_rows(
        data_dir / "phase_step_comparison.csv",
        _comparison_rows(arrays_by_steps),
    )
    stats_rows = _stats_rows(arrays_by_steps)
    stats_csv = _write_rows(data_dir / "phase_step_stats.csv", stats_rows)
    stats_json = write_json(data_dir / "phase_step_stats.json", {"stats": stats_rows})
    comparison_npz = _save_comparison_npz(
        data_dir / "phase_step_comparison.npz",
        arrays_by_steps,
    )

    base_config = build_scheme_config(legacy.DemoConfig(), n_steps=4)
    map_figure = _plot_map_comparison(
        plot_dir / "phase_step_map_comparison.png",
        arrays_by_steps,
        energy_xlim_eV=tuple(base_config.ta_map_xlim_eV),
        dpi=int(base_config.figure_dpi),
    )
    lineout_figures = _plot_delay_lineouts(
        preview_dir,
        arrays_by_steps,
        energy_xlim_eV=tuple(base_config.plot_energy_range_eV),
        dpi=int(base_config.figure_dpi),
    )
    biased_overlay_figures = _plot_biased_overlays(
        output_dir,
        arrays_by_steps,
        energy_xlim_eV=tuple(base_config.ta_map_xlim_eV),
        dpi=int(base_config.figure_dpi),
    )

    meta = {
        **manifest,
        "output_dir": output_dir,
        "subruns": {f"N{key}": value for key, value in subruns.items()},
        "outputs": {
            "comparison_csv": comparison_csv,
            "stats_csv": stats_csv,
            "stats_json": stats_json,
            "comparison_npz": comparison_npz,
            "map_figure": map_figure,
            "lineout_figures": lineout_figures,
            "biased_overlay_figures": biased_overlay_figures,
        },
    }
    meta_path = write_json(output_dir / "meta.json", meta)
    print("TA phase-step comparison finished.")
    print(f"output_dir: {output_dir}")
    print(f"comparison_npz: {comparison_npz}")
    print(f"meta_json: {meta_path}")
    return {**meta, "meta_json": meta_path}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--force-run", action="store_true")
    parser.add_argument("--force", action="store_true", help="Alias for --force-run.")
    parser.add_argument("--no-checkpoints", action="store_true")
    parser.add_argument("--wavelength", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the exact 3 x 3 run manifest without executing simulations.",
    )
    return parser.parse_args()


def main() -> None:
    run_comparison(parse_args())


if __name__ == "__main__":
    main()
