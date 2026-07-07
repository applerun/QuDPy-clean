#!/usr/bin/env python3
"""Add-on outputs for the TA phase-cycling demo without modifying the base file.

This add-on is still demo/report-level code. It is not a generic
pulse-sequence framework and does not connect the experimental TA recipe v1
prototype under ``qudpy_sjh/experiments/ta/``.

Place this file next to:

    ta_three_level_intrinsic_response_phase_cycling_demo.py

Then run this file instead. It calls the base demo first, then adds:

1. pure-probe IO preview using qudpy_sjh.utils.io.save_result_case;
2. pure-probe absorption spectrum figure and CSV;
3. selected-delay IO previews using qudpy_sjh.utils.io.save_result_case;
4. selected-energy TA kinetics figure and CSV.

Note: ``ta_diff_spectra_delay_*.png`` files are spectral-analysis figures from the
base demo, not density-matrix IO previews. IO preview only refers to trajectory
preview figures produced by ``qudpy_sjh.utils.io.save_result_case``.

This file intentionally does not change the base checkpoint naming rules. Existing
checkpoints such as ``phase_0_delay_80_fs.ckp`` should still be reused by the base
demo.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import argparse
import csv
import importlib.util
import json
import shutil
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


BASE_SCRIPT_NAME = "ta_three_level_intrinsic_response_phase_cycling_demo.py"
DEFAULT_SELECTED_KINETIC_ENERGIES_EV = (1.50, 1.55, 1.62, 1.70)


def load_base_module():
    """Load the original demo from the same directory as this add-on file."""

    base_path = Path(__file__).resolve().with_name(BASE_SCRIPT_NAME)
    if not base_path.exists():
        raise FileNotFoundError(
            f"Cannot find base script next to this file: {base_path}\n"
            f"Put this file in the same folder as {BASE_SCRIPT_NAME}."
        )

    spec = importlib.util.spec_from_file_location("ta_phase_base_demo", base_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load base script: {base_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_energy_list(text: str | None) -> tuple[float, ...]:
    if text is None or str(text).strip() == "":
        return DEFAULT_SELECTED_KINETIC_ENERGIES_EV
    values = []
    for item in str(text).replace(";", ",").split(","):
        item = item.strip()
        if not item:
            continue
        values.append(float(item))
    if not values:
        raise ValueError("selected energy list is empty.")
    return tuple(values)


def json_safe(value: Any) -> Any:
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return json_safe(value.to_dict())
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, complex):
        return {"real": float(value.real), "imag": float(value.imag)}
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), indent=2, ensure_ascii=False), encoding="utf-8")
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


def build_config(base, args: argparse.Namespace):
    base_config = base.DemoConfig()
    payload = {
        **asdict(base_config),
        "force_run": bool(args.force_run),
        "use_checkpoints": not bool(args.no_checkpoints),
        "plot_use_wavelength": bool(args.wavelength),
    }
    return base.DemoConfig(**payload)


def to_plot_x(base, energy_eV: np.ndarray, *, use_wavelength: bool) -> tuple[np.ndarray, str]:
    if use_wavelength:
        return base.HC_EV_NM / energy_eV, "Probe wavelength (nm)"
    return energy_eV, "Probe photon energy (eV)"


def prepare_plot_arrays(base, energy_eV: np.ndarray, values: np.ndarray, config) -> tuple[np.ndarray, np.ndarray, str]:
    """Use the same plotting convention as the base demo."""

    if hasattr(base, "prepare_plot_arrays"):
        return base.prepare_plot_arrays(energy_eV, values, config)

    x, xlabel = to_plot_x(base, energy_eV, use_wavelength=config.plot_use_wavelength)
    plot_values = np.asarray(values, dtype=float)

    if config.plot_use_wavelength:
        order = np.argsort(x)
        x = x[order]
        plot_values = plot_values[:, order]

    if config.plot_energy_range_eV is not None:
        e_min, e_max = config.plot_energy_range_eV
        if config.plot_use_wavelength:
            x_min = base.HC_EV_NM / e_max
            x_max = base.HC_EV_NM / e_min
            mask = (x >= min(x_min, x_max)) & (x <= max(x_min, x_max))
        else:
            mask = (x >= e_min) & (x <= e_max)
        if np.any(mask):
            x = x[mask]
            plot_values = plot_values[:, mask]

    return x, plot_values, xlabel


def run_probe_reference(base, config, *, output_dir: Path, normalizer):
    """Recreate the probe-only reference through the base code path.

    This reuses the base checkpoint key ``probe_only``.
    """

    probe_field = base.make_probe_reference_field(config)
    probe_params = base.make_physical_params(
        config,
        probe_field,
        case_name="probe_only",
        description="Probe-only reference shared by all TA maps.",
    )
    probe_result = base.run_with_checkpoint(
        probe_params,
        normalizer=normalizer,
        output_dir=output_dir,
        case_key="probe_only",
        config=config,
    )
    probe_response = base.response_from_result(probe_result, probe_field, config)
    return probe_field, probe_result, probe_response


def save_probe_io_preview(
    probe_result,
    *,
    output_dir: Path,
    preview_dir: Path,
    example_name: str,
    preview_dpi: int,
) -> dict[str, str]:
    """Save pure-probe preview through utils.io.save_result_case."""

    from qudpy_sjh.utils.io import save_result_case

    io_root = preview_dir / "io_preview"
    written = save_result_case(
        probe_result,
        io_root,
        output_data=False,
        output_preview=True,
        save_json=True,
        append_results_csv=False,
        preview_dpi=int(preview_dpi),
        example_name=example_name,
        condition_name="probe_only_reference",
        case_name="probe_only_reference",
    )

    result: dict[str, str] = {key: str(value) for key, value in written.items()}

    # Convenience copy so the preview is easy to find in figures/preview/.
    preview_path = written.get("preview")
    if preview_path is not None and Path(preview_path).exists():
        easy_path = preview_dir / "probe_only_io_preview.png"
        shutil.copy2(preview_path, easy_path)
        result["preview_easy_copy"] = str(easy_path)

    return result


def run_selected_delay_phase0_case(
    base,
    config,
    *,
    delay_fs: float,
    output_dir: Path,
    normalizer,
):
    """Load or run the phase-0 pump+probe case using the base map checkpoint key.

    This intentionally reuses the base map checkpoint name:
        phase_0_delay_<delay>_fs.ckp

    It avoids creating separate trace_delay_* checkpoints just for IO preview.
    """

    delay_label = base.safe_delay_label(delay_fs)
    map_name = "phase_0"
    case_key = f"{map_name}_delay_{delay_label}_fs"

    field = base.make_ta_field(
        config,
        delay_fs=float(delay_fs),
        pump_phase_rad=0.0,
        name=case_key,
    )
    params = base.make_physical_params(
        config,
        field,
        case_name=case_key,
        description=f"Selected-delay IO preview case, delay={delay_fs:g} fs, phase=0.",
    )
    return base.run_with_checkpoint(
        params,
        normalizer=normalizer,
        output_dir=output_dir,
        case_key=case_key,
        config=config,
    )


def _copy_io_preview_or_raise(written: dict[str, Path], *, expected_preview: Path, easy_path: Path) -> Path:
    """Copy IO preview to an easy top-level path and fail loudly if it is missing."""

    preview_path = written.get("preview")
    if preview_path is None:
        preview_path = expected_preview
    preview_path = Path(preview_path)

    if not preview_path.exists():
        raise FileNotFoundError(
            "IO preview was not generated. Expected: "
            f"{preview_path}\n"
            "Check save_result_case(output_preview=True) and the returned paths."
        )

    easy_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(preview_path, easy_path)
    return easy_path


def save_selected_delay_io_previews(
    base,
    config,
    *,
    output_dir: Path,
    preview_dir: Path,
    normalizer,
    example_name: str,
    preview_dpi: int,
) -> dict[str, str]:
    """Save selected-delay TA previews through utils.io.save_result_case.

    The top-level rho_preview_delay_*.png files are convenience copies of the
    IO-generated preview.png files. They intentionally overwrite the base demo's
    legacy hand-drawn rho_preview_delay_*.png outputs, if those files exist.
    """

    from qudpy_sjh.utils.io import save_result_case

    io_root = preview_dir / "io_preview"
    outputs: dict[str, str] = {}

    for target_delay in tuple(float(x) for x in config.preview_delays_fs):
        delay_label = base.safe_delay_label(target_delay)
        case_name = f"ta_delay_{delay_label}_fs_phase_0"

        print(f"[extra] selected-delay IO preview: delay={target_delay:g} fs -> {case_name}")
        result = run_selected_delay_phase0_case(
            base,
            config,
            delay_fs=target_delay,
            output_dir=output_dir,
            normalizer=normalizer,
        )

        written = save_result_case(
            result,
            io_root,
            output_data=False,
            output_preview=True,
            save_json=True,
            append_results_csv=False,
            preview_dpi=int(preview_dpi),
            example_name=example_name,
            condition_name="selected_delay_phase_0",
            case_name=case_name,
        )

        for key, value in written.items():
            outputs[f"{case_name}_{key}"] = str(value)

        expected_preview = io_root / case_name / "figs" / "preview.png"
        easy_path = preview_dir / f"rho_preview_delay_{delay_label}.png"
        copied = _copy_io_preview_or_raise(
            written,
            expected_preview=expected_preview,
            easy_path=easy_path,
        )
        outputs[f"{case_name}_preview_easy_copy"] = str(copied)

    return outputs


def write_probe_absorption_csv(base, *, path: Path, probe_response: dict[str, np.ndarray]) -> Path:
    energy_eV = np.asarray(probe_response["energy_eV"], dtype=float)
    omega_fs_inv = np.asarray(probe_response["omega_fs_inv"], dtype=float)
    absorption = np.asarray(probe_response["absorption"], dtype=float)

    rows = []
    for energy, omega, value in zip(energy_eV, omega_fs_inv, absorption):
        rows.append(
            {
                "energy_eV": float(energy),
                "wavelength_nm": float(base.HC_EV_NM / energy),
                "omega_fs_inv": float(omega),
                "probe_only_absorption": float(value),
            }
        )
    return write_csv_rows(path, rows)


def plot_probe_absorption_spectrum(
    base,
    *,
    path: Path,
    probe_response: dict[str, np.ndarray],
    config,
    dpi: int,
) -> Path:
    energy_eV = np.asarray(probe_response["energy_eV"], dtype=float)
    absorption = np.asarray(probe_response["absorption"], dtype=float)

    x_plot, y2d, xlabel = prepare_plot_arrays(base, energy_eV, absorption[None, :], config)
    y_plot = np.asarray(y2d[0], dtype=float)

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    ax.plot(x_plot, y_plot, linewidth=2.0, color="black", label="probe-only absorption")
    ax.axhline(0.0, linewidth=0.8, linestyle="--", color="black", alpha=0.5)
    ax.set_title("Probe-only absorption spectrum")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("omega * Im[P_probe(omega) / E_probe(omega)]")
    ax.legend(fontsize=8, loc="best")

    finite = y_plot[np.isfinite(y_plot)]
    if finite.size:
        y_abs = float(np.max(np.abs(finite)))
        if y_abs > 0:
            ax.set_ylim(-1.08 * y_abs, 1.08 * y_abs)
        ax.text(
            0.02,
            0.98,
            f"min={np.min(finite):.2e}\nmax={np.max(finite):.2e}\nrms={np.sqrt(np.mean(finite ** 2)):.2e}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            bbox={"facecolor": "white", "alpha": 0.72, "edgecolor": "none", "pad": 2},
        )

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=int(dpi))
    plt.close(fig)
    return path


def nearest_energy_index(energy_eV: np.ndarray, target_eV: float) -> int:
    return int(np.argmin(np.abs(np.asarray(energy_eV, dtype=float) - float(target_eV))))


def write_selected_energy_kinetics_csv(
    base,
    *,
    path: Path,
    delays_fs: np.ndarray,
    energy_eV: np.ndarray,
    phase_stack: np.ndarray,
    phase_avg: np.ndarray,
    phase_labels: list[str],
    selected_energies_eV: tuple[float, ...],
) -> Path:
    rows = []
    for target_eV in selected_energies_eV:
        idx = nearest_energy_index(energy_eV, target_eV)
        actual_eV = float(energy_eV[idx])
        row_common = {
            "target_energy_eV": float(target_eV),
            "actual_energy_eV": actual_eV,
            "actual_wavelength_nm": float(base.HC_EV_NM / actual_eV),
            "energy_index": int(idx),
        }
        for delay_index, delay_fs in enumerate(delays_fs):
            row = {
                **row_common,
                "delay_fs": float(delay_fs),
                "phase_average": float(phase_avg[delay_index, idx]),
            }
            for phase_i, label in enumerate(phase_labels):
                row[f"phase_{label}"] = float(phase_stack[phase_i, delay_index, idx])
            rows.append(row)
    return write_csv_rows(path, rows)


def plot_selected_energy_kinetics(
    base,
    *,
    path: Path,
    delays_fs: np.ndarray,
    energy_eV: np.ndarray,
    phase_stack: np.ndarray,
    phase_avg: np.ndarray,
    phase_labels: list[str],
    selected_energies_eV: tuple[float, ...],
    dpi: int,
) -> Path:
    n = len(selected_energies_eV)
    ncols = 2 if n > 1 else 1
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.4 * ncols, 4.4 * nrows), squeeze=False)
    axes_flat = axes.ravel()

    phase_title_map = {
        "0": "phase 0",
        "pi2": "phase π/2",
        "pi": "phase π",
        "3pi2": "phase 3π/2",
    }
    line_styles = {
        "0": "-",
        "pi2": "--",
        "pi": "-.",
        "3pi2": ":",
    }

    for ax_left, target_eV in zip(axes_flat, selected_energies_eV):
        idx = nearest_energy_index(energy_eV, target_eV)
        actual_eV = float(energy_eV[idx])
        actual_nm = float(base.HC_EV_NM / actual_eV)
        ax_right = ax_left.twinx()

        avg_y = np.asarray(phase_avg[:, idx], dtype=float)
        ax_left.plot(
            delays_fs,
            avg_y,
            color="black",
            linewidth=2.2,
            label="phase average",
        )
        ax_left.axhline(0.0, color="black", linewidth=0.8, linestyle="--", alpha=0.45)

        phase_values = []
        for phase_i, label in enumerate(phase_labels):
            y = np.asarray(phase_stack[phase_i, :, idx], dtype=float)
            phase_values.append(y)
            ax_right.plot(
                delays_fs,
                y,
                color="red",
                linestyle=line_styles.get(label, "-"),
                linewidth=1.2,
                alpha=0.75,
                label=phase_title_map.get(label, f"phase {label}"),
            )
        ax_right.axhline(0.0, color="red", linewidth=0.8, linestyle="--", alpha=0.25)

        avg_finite = avg_y[np.isfinite(avg_y)]
        phase_arr = np.vstack(phase_values)
        phase_finite = phase_arr[np.isfinite(phase_arr)]
        if avg_finite.size:
            avg_abs = float(np.max(np.abs(avg_finite)))
            if avg_abs > 0:
                ax_left.set_ylim(-1.08 * avg_abs, 1.08 * avg_abs)
        if phase_finite.size:
            phase_abs = float(np.max(np.abs(phase_finite)))
            if phase_abs > 0:
                ax_right.set_ylim(-1.08 * phase_abs, 1.08 * phase_abs)

        ax_left.set_title(f"TA kinetics at {actual_eV:.4f} eV ({actual_nm:.1f} nm)")
        ax_left.set_xlabel("Pump-probe delay (fs)")
        ax_left.set_ylabel("Phase-averaged S_TA", color="black")
        ax_right.set_ylabel("Single-phase S_TA", color="red")
        ax_left.tick_params(axis="y", labelcolor="black")
        ax_right.tick_params(axis="y", labelcolor="red")

        lines_left, labels_left = ax_left.get_legend_handles_labels()
        lines_right, labels_right = ax_right.get_legend_handles_labels()
        ax_left.legend(lines_left + lines_right, labels_left + labels_right, fontsize=8, loc="best")

    for ax in axes_flat[n:]:
        ax.axis("off")

    fig.suptitle("Selected-energy TA kinetics", y=0.995)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=int(dpi))
    plt.close(fig)
    return path


def add_extra_outputs(
    base,
    config,
    *,
    output_dir: Path,
    selected_energies_eV: tuple[float, ...],
    preview_dpi: int,
) -> dict[str, str]:
    data_dir = output_dir / "data"
    preview_dir = output_dir / "figures" / "preview"
    data_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)

    normalizer = base.ParaNormalizer(auto_scale=True)
    _probe_field, probe_result, probe_response = run_probe_reference(
        base,
        config,
        output_dir=output_dir,
        normalizer=normalizer,
    )

    outputs: dict[str, str] = {}

    probe_preview = save_probe_io_preview(
        probe_result,
        output_dir=output_dir,
        preview_dir=preview_dir,
        example_name=config.example_name,
        preview_dpi=preview_dpi,
    )
    outputs.update({f"probe_io_{key}": value for key, value in probe_preview.items()})

    selected_delay_previews = save_selected_delay_io_previews(
        base,
        config,
        output_dir=output_dir,
        preview_dir=preview_dir,
        normalizer=normalizer,
        example_name=config.example_name,
        preview_dpi=preview_dpi,
    )
    outputs.update({f"selected_delay_io_{key}": value for key, value in selected_delay_previews.items()})

    probe_absorption_csv = write_probe_absorption_csv(
        base,
        path=data_dir / "probe_only_absorption_spectrum.csv",
        probe_response=probe_response,
    )
    outputs["probe_absorption_csv"] = str(probe_absorption_csv)

    probe_absorption_fig = plot_probe_absorption_spectrum(
        base,
        path=preview_dir / "probe_only_absorption_spectrum.png",
        probe_response=probe_response,
        config=config,
        dpi=preview_dpi,
    )
    outputs["probe_absorption_figure"] = str(probe_absorption_fig)

    npz_path = data_dir / "ta_phase_cycling_comparison.npz"
    if not npz_path.exists():
        raise FileNotFoundError(
            f"Missing base demo NPZ output: {npz_path}\n"
            "Run without --skip-base-run, or run the base demo first."
        )

    with np.load(npz_path, allow_pickle=False) as data:
        delays_fs = np.asarray(data["delays_fs"], dtype=float)
        energy_eV = np.asarray(data["energy_eV"], dtype=float)
        phase_stack = np.asarray(data["TA_phase_cases"], dtype=float)
        phase_avg = np.asarray(data["TA_phase_avg"], dtype=float)
        phase_labels = [str(x) for x in np.asarray(data["phase_labels"]).tolist()]

    kinetics_csv = write_selected_energy_kinetics_csv(
        base,
        path=data_dir / "ta_selected_energy_kinetics.csv",
        delays_fs=delays_fs,
        energy_eV=energy_eV,
        phase_stack=phase_stack,
        phase_avg=phase_avg,
        phase_labels=phase_labels,
        selected_energies_eV=selected_energies_eV,
    )
    outputs["selected_energy_kinetics_csv"] = str(kinetics_csv)

    kinetics_fig = plot_selected_energy_kinetics(
        base,
        path=preview_dir / "ta_selected_energy_kinetics.png",
        delays_fs=delays_fs,
        energy_eV=energy_eV,
        phase_stack=phase_stack,
        phase_avg=phase_avg,
        phase_labels=phase_labels,
        selected_energies_eV=selected_energies_eV,
        dpi=preview_dpi,
    )
    outputs["selected_energy_kinetics_figure"] = str(kinetics_fig)

    summary = write_json(
        output_dir / "extra_outputs_meta.json",
        {
            "source": Path(__file__).name,
            "base_script": BASE_SCRIPT_NAME,
            "selected_kinetic_energies_eV": selected_energies_eV,
            "outputs": outputs,
            "notes": {
                "pure_probe_preview": "Generated with qudpy_sjh.utils.io.save_result_case, then copied to figures/preview/probe_only_io_preview.png for convenience.",
                "selected_delay_previews": "Generated with qudpy_sjh.utils.io.save_result_case; top-level rho_preview_delay_*.png files are convenience copies of IO preview.png files.",
                "checkpoint_policy": "Uses the base demo run_with_checkpoint and preserves the base case keys, including *_fs suffixes.",
            },
        },
    )
    outputs["extra_outputs_meta"] = str(summary)

    return outputs


def parse_args(base) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=base.DEFAULT_OUTPUT_DIR,
        help="Output directory shared with the base demo.",
    )
    parser.add_argument(
        "--force-run",
        action="store_true",
        help="Forwarded to the base demo. Recompute instead of reusing checkpoints.",
    )
    parser.add_argument(
        "--no-checkpoints",
        action="store_true",
        help="Forwarded to the base demo. Run without checkpoint load/save.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Forwarded to the base demo. Use the base quick delay grid.",
    )
    parser.add_argument(
        "--wavelength",
        action="store_true",
        help="Forwarded to the base demo plotting convention.",
    )
    parser.add_argument(
        "--skip-base-run",
        action="store_true",
        help="Do not call base.run_demo; use existing NPZ/checkpoints and only add extra outputs.",
    )
    parser.add_argument(
        "--selected-energies",
        type=str,
        default=",".join(f"{x:g}" for x in DEFAULT_SELECTED_KINETIC_ENERGIES_EV),
        help="Comma-separated selected probe photon energies in eV for kinetics plots.",
    )
    parser.add_argument(
        "--preview-dpi",
        type=int,
        default=180,
        help="DPI for added preview figures.",
    )
    return parser.parse_args()


def main() -> None:
    base = load_base_module()
    args = parse_args(base)
    config = build_config(base, args)
    output_dir = Path(args.output_dir)
    selected_energies = parse_energy_list(args.selected_energies)

    if not args.skip_base_run:
        base.run_demo(config, output_dir=output_dir, quick=bool(args.quick))

    outputs = add_extra_outputs(
        base,
        config,
        output_dir=output_dir,
        selected_energies_eV=selected_energies,
        preview_dpi=int(args.preview_dpi),
    )

    print("\nExtra TA outputs finished.")
    for key, value in outputs.items():
        print(f"{key:36s}: {value}")


if __name__ == "__main__":
    main()
