#!/usr/bin/env python3
"""Canonical three-level TA phase-step convergence example.

The physical parameters match the historical three-level validation baseline.
The detector uses a fixed phase-zero probe local oscillator.  Pump and probe
interaction phases are independently cycled; the local-oscillator phase is not.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
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

DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "bin"
    / "optical_bloch_plots"
    / "ta_three_level_canonical_phase_step_convergence"
)
DELAYS_FS = (-100.0, 0.0, 100.0)
PHASE_STEP_COUNTS = (2, 3, 4)
TARGET = {"pump": 0, "probe": 1}
ENERGY_WINDOW_EV = (1.4, 1.8)
REFERENCE_N = 4

from qudpy_sjh.experiments import (  # noqa: E402
    PhaseGrid,
    PulseSpec,
    ReadoutPlan,
    SingleRunCheckpointSettings,
    load_projected_result,
    project_phase_orders,
    save_projected_result,
)
from qudpy_sjh.experiments.ta import TAPrePCObservable, TAPrePCRecipe  # noqa: E402
from qudpy_sjh.systems import (  # noqa: E402
    make_base_physical_params_from_system,
    make_three_level_ladder_system,
)
from qudpy_sjh.utils.core import PureDephasingChannel  # noqa: E402
from qudpy_sjh.utils.fields.carrier_envelope import (  # noqa: E402
    make_gaussian_carrier_envelope_field,
)
from qudpy_sjh.utils.serialization import write_json  # noqa: E402


def uniform_phases(n_steps: int) -> tuple[float, ...]:
    n = int(n_steps)
    if n < 1:
        raise ValueError("n_steps must be positive.")
    return tuple(2.0 * math.pi * index / n for index in range(n))


def build_system():
    return make_three_level_ladder_system(
        energy_01_eV=1.55,
        energy_12_eV=1.70,
        mu01_D=5.0,
        mu12_D=9.0,
        labels=("level_0", "level_1", "level_2"),
        initial_state="ground",
        name="canonical_three_level_ta_validation",
        metadata={
            "baseline": "M6 frozen three-level validation parameters",
            "scope": "M6 phase-step convergence",
        },
    )


def build_pulses() -> tuple[PulseSpec, PulseSpec]:
    pump_field = make_gaussian_carrier_envelope_field(
        E0_MV_per_cm=0.30,
        laser_energy_eV=1.55,
        center_fs=0.0,
        sigma_fs=12.0,
        phase_rad=0.0,
        name="pump_template",
    )
    probe_field = make_gaussian_carrier_envelope_field(
        E0_MV_per_cm=0.008,
        laser_energy_eV=1.62,
        center_fs=0.0,
        sigma_fs=7.0,
        phase_rad=0.0,
        name="probe_template",
    )
    return (
        PulseSpec(
            name="pump",
            field_template=pump_field,
            template_center_fs=0.0,
            phase_tag="pump",
            independent_phase=True,
        ),
        PulseSpec(
            name="probe",
            field_template=probe_field,
            template_center_fs=0.0,
            phase_tag="probe",
            independent_phase=True,
        ),
    )


def build_base_params(system: Any, probe: PulseSpec):
    return make_base_physical_params_from_system(
        system,
        field=probe.field_template,
        t_start_fs=-1500.0,
        t_end_fs=1500.0,
        dt_fs=0.2,
        solver_mode="lab_exact",
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
        input_description="Canonical M6 three-level TA phase-step validation.",
        input_metadata={
            "transition_dephasing_note": (
                "Explicit level pure-dephasing channels preserve the historical baseline; "
                "the separate system-adapter transition mapping issue is not changed in M6."
            ),
        },
    )


def build_readout(probe: PulseSpec, *, mode: str = "full") -> ReadoutPlan:
    return ReadoutPlan(
        mode=mode,
        readout_field=probe.field_template,
        window="hann",
        subtract_mean=True,
        rel_threshold=1.0e-6,
        zero_padding_factor=4,
        emitted_field_scale=1.0,
        return_intermediates=False,
        metadata={
            "detector_policy": "fixed phase-zero external probe LO",
            "lo_in_phase_grid": False,
        },
    )


def build_recipe(n_steps: int, *, readout_mode: str = "full") -> TAPrePCRecipe:
    pump, probe = build_pulses()
    system = build_system()
    phases = uniform_phases(n_steps)
    return TAPrePCRecipe(
        base_params=build_base_params(system, probe),
        pump=pump,
        probe=probe,
        delays_fs=DELAYS_FS,
        phase_grid=PhaseGrid({"pump": phases, "probe": phases}),
        readout_plan=build_readout(probe, mode=readout_mode),
        observable="delta_T_over_T",
        number_density_m3=1.0e24,
        probe_center_fs=0.0,
        denominator_rel_threshold=1.0e-8,
        denominator_abs_threshold=0.0,
        target_phase_vector=dict(TARGET),
        case_name=f"canonical_ta_N{int(n_steps)}",
        metadata={
            "example": "ta_three_level_canonical_phase_step_convergence",
            "energy_comparison_window_eV": ENERGY_WINDOW_EV,
        },
    )


def _checkpoint_executor(output_dir: Path, n_steps: int, *, force_run: bool):
    checkpoint_dir = output_dir / f"N{n_steps}xN{n_steps}" / "checkpoints"

    def execute(plan):
        checkpoint = checkpoint_dir / f"{plan.case_name}.ckp"
        configured = replace(
            plan,
            checkpoint=SingleRunCheckpointSettings(
                enabled=True,
                checkpoint_path=checkpoint,
                force_run=force_run,
            ),
        )
        return configured.execute()

    return execute


def _complete_valid_mask(observable: TAPrePCObservable) -> np.ndarray:
    phase_axes = tuple(
        index
        for index, name in enumerate(observable.axis_names)
        if name.startswith("phase:")
    )
    return np.all(observable.valid_reference_mask, axis=phase_axes)


def project_observable(
    observable: TAPrePCObservable,
    recipe: TAPrePCRecipe,
    *,
    targets: Mapping[str, Mapping[str, int]],
) -> dict[str, Any]:
    result = project_phase_orders(
        observable.data,
        axis_names=observable.axis_names,
        axis_values=observable.axis_values,
        phase_grid=recipe.phase_grid,
        targets=targets,
    )
    result["metadata"].update(
        {
            "observable": observable.quantity,
            "condition_formula": observable.metadata["condition_formula"],
            "readout_mode": observable.metadata["readout_mode"],
            "valid_reference_mask_after_phase_reduction": _complete_valid_mask(observable),
            "denominator_policy": observable.metadata["denominator_policy"],
            "recipe": recipe.to_dict(),
        }
    )
    return result


def run_phase_grid(
    n_steps: int,
    *,
    output_dir: Path,
    force_run: bool,
) -> dict[str, Any]:
    recipe = build_recipe(n_steps)
    start = time.perf_counter()
    dynamics = recipe.execute_dynamics(
        executor=_checkpoint_executor(output_dir, n_steps, force_run=force_run)
    )
    readouts = recipe.apply_readout(dynamics)
    observable = recipe.postprocess(readouts)
    projected = project_observable(
        observable,
        recipe,
        targets={"S_0_1": TARGET},
    )
    base = output_dir / f"N{n_steps}xN{n_steps}" / "projected"
    save_projected_result(projected, base)
    loaded = load_projected_result(base)
    np.testing.assert_allclose(
        loaded["projected"]["S_0_1"],
        projected["projected"]["S_0_1"],
        equal_nan=True,
    )
    return {
        "recipe": recipe,
        "dynamics": dynamics,
        "observable": observable,
        "projected": projected,
        "elapsed_s": time.perf_counter() - start,
    }


def _window_mask(energy_eV: np.ndarray) -> np.ndarray:
    energy = np.asarray(energy_eV, dtype=float)
    return (energy >= ENERGY_WINDOW_EV[0]) & (energy <= ENERGY_WINDOW_EV[1])


def _finite_mask(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    return np.isfinite(array.real) & np.isfinite(array.imag)


def _relative_l2(values: np.ndarray, reference: np.ndarray) -> float | None:
    denominator = float(np.linalg.norm(reference))
    if denominator <= np.finfo(float).tiny:
        return None
    return float(np.linalg.norm(values - reference) / denominator)


def build_summary_rows(results: Mapping[int, Mapping[str, Any]]) -> list[dict[str, Any]]:
    reference = results[REFERENCE_N]["projected"]
    energy = np.asarray(reference["axis_values"]["energy_eV"], dtype=float)
    window = _window_mask(energy)
    spectra = {
        n_steps: np.asarray(payload["projected"]["projected"]["S_0_1"])
        for n_steps, payload in results.items()
    }
    for n_steps, payload in results.items():
        local_energy = np.asarray(
            payload["projected"]["axis_values"]["energy_eV"], dtype=float
        )
        if local_energy.shape != energy.shape or not np.allclose(
            local_energy, energy, rtol=0.0, atol=1.0e-12
        ):
            raise ValueError(f"N={n_steps} energy axis differs from N={REFERENCE_N}.")

    rows: list[dict[str, Any]] = []
    for delay_index, delay_fs in enumerate(DELAYS_FS):
        row: dict[str, Any] = {
            "delay_fs": delay_fs,
            "reference_n": REFERENCE_N,
        }
        for n_steps in PHASE_STEP_COUNTS:
            values = spectra[n_steps][delay_index]
            valid = window & _finite_mask(values)
            row[f"max_abs_S{n_steps}"] = (
                None if not np.any(valid) else float(np.max(np.abs(values[valid])))
            )
            row[f"max_abs_imag_S{n_steps}"] = (
                None if not np.any(valid) else float(np.max(np.abs(values[valid].imag)))
            )
            row[f"valid_bins_N{n_steps}"] = int(np.count_nonzero(valid))
        reference_values = spectra[REFERENCE_N][delay_index]
        for n_steps in (2, 3):
            values = spectra[n_steps][delay_index]
            joint = window & _finite_mask(values) & _finite_mask(reference_values)
            count = int(np.count_nonzero(joint))
            row[f"joint_valid_bins_N{n_steps}_N4"] = count
            row[f"joint_valid_fraction_N{n_steps}_N4"] = count / int(np.count_nonzero(window))
            if count == 0:
                row[f"max_abs_error_N{n_steps}_N4"] = None
                row[f"relative_L2_error_N{n_steps}_N4"] = None
            else:
                selected = values[joint]
                selected_reference = reference_values[joint]
                row[f"max_abs_error_N{n_steps}_N4"] = float(
                    np.max(np.abs(selected - selected_reference))
                )
                row[f"relative_L2_error_N{n_steps}_N4"] = _relative_l2(
                    selected, selected_reference
                )
        rows.append(row)
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def _delay_label(delay_fs: float) -> str:
    if delay_fs < 0:
        return f"minus{abs(int(delay_fs))}fs"
    if delay_fs > 0:
        return f"plus{int(delay_fs)}fs"
    return "0fs"


def write_overlays(output_dir: Path, results: Mapping[int, Mapping[str, Any]]) -> list[Path]:
    energy = np.asarray(
        results[REFERENCE_N]["projected"]["axis_values"]["energy_eV"], dtype=float
    )
    window = _window_mask(energy)
    colors = {2: "#0072B2", 3: "#D55E00", 4: "#009E73"}
    paths = []
    for delay_index, delay_fs in enumerate(DELAYS_FS):
        fig, axes = plt.subplots(3, 1, figsize=(8.0, 8.0), sharex=True)
        for n_steps in PHASE_STEP_COUNTS:
            values = np.asarray(
                results[n_steps]["projected"]["projected"]["S_0_1"]
            )[delay_index]
            axes[0].plot(energy[window], values.real[window], color=colors[n_steps], label=f"N={n_steps}")
            axes[1].plot(energy[window], values.imag[window], color=colors[n_steps], label=f"N={n_steps}")
            axes[2].plot(energy[window], np.abs(values[window]), color=colors[n_steps], label=f"N={n_steps}")
        axes[0].set_ylabel("Re S(0,1)")
        axes[1].set_ylabel("Im S(0,1)")
        axes[2].set_ylabel("|S(0,1)|")
        axes[2].set_xlabel("Energy (eV)")
        axes[0].legend()
        axes[0].set_title(f"Canonical fixed-LO TA, T={delay_fs:g} fs")
        fig.tight_layout()
        path = output_dir / f"delay_{_delay_label(delay_fs)}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(path)
    return paths


def run_weak_check(reference_payload: Mapping[str, Any]) -> dict[str, Any]:
    recipe = reference_payload["recipe"]
    weak_plan = build_readout(recipe.probe, mode="weak")
    weak_readouts = recipe.apply_readout(
        reference_payload["dynamics"],
        readout_plan=weak_plan,
    )
    weak_observable = recipe.postprocess(weak_readouts)
    weak_projected = project_observable(
        weak_observable,
        recipe,
        targets={"S_0_1": TARGET},
    )
    full = np.asarray(reference_payload["projected"]["projected"]["S_0_1"])
    weak = np.asarray(weak_projected["projected"]["S_0_1"])
    energy = np.asarray(weak_projected["axis_values"]["energy_eV"], dtype=float)
    joint = _window_mask(energy)[None, :] & _finite_mask(full) & _finite_mask(weak)
    return {
        "readout_modes": ["full", "weak"],
        "same_recipe": True,
        "same_axes": bool(full.shape == weak.shape),
        "joint_valid_bins": int(np.count_nonzero(joint)),
        "max_abs_full": float(np.max(np.abs(full[joint]))),
        "max_abs_weak": float(np.max(np.abs(weak[joint]))),
        "max_abs_difference": float(np.max(np.abs(full[joint] - weak[joint]))),
        "relative_L2_full_vs_weak": _relative_l2(weak[joint], full[joint]),
        "projected": weak_projected,
    }


def run_fourier_content(reference_payload: Mapping[str, Any]) -> dict[str, Any]:
    recipe = reference_payload["recipe"]
    observable = reference_payload["observable"]
    targets = {
        f"S_0_{order}": {"pump": 0, "probe": order}
        for order in range(REFERENCE_N)
    }
    projected = project_observable(observable, recipe, targets=targets)
    energy = np.asarray(projected["axis_values"]["energy_eV"], dtype=float)
    window = _window_mask(energy)
    rows = []
    for name, values in projected["projected"].items():
        array = np.asarray(values)
        finite = window[None, :] & _finite_mask(array)
        rows.append(
            {
                "channel": name,
                "max_abs_over_delays_and_window": float(np.max(np.abs(array[finite]))),
            }
        )
    return {"projected": projected, "rows": rows}


def save_validation_outputs(
    output_dir: Path,
    results: Mapping[int, Mapping[str, Any]],
    *,
    summary_rows: list[dict[str, Any]],
    weak_check: Mapping[str, Any],
    fourier_content: Mapping[str, Any],
    plot_paths: list[Path],
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    energy = np.asarray(
        results[REFERENCE_N]["projected"]["axis_values"]["energy_eV"], dtype=float
    )
    spectra_path = output_dir / "spectra.npz"
    np.savez_compressed(
        spectra_path,
        energy_eV=energy,
        delays_fs=np.asarray(DELAYS_FS),
        **{
            f"S_0_1_N{n_steps}": np.asarray(
                results[n_steps]["projected"]["projected"]["S_0_1"]
            )
            for n_steps in PHASE_STEP_COUNTS
        },
        **{
            f"valid_mask_N{n_steps}": np.asarray(
                results[n_steps]["projected"]["metadata"][
                    "valid_reference_mask_after_phase_reduction"
                ],
                dtype=bool,
            )
            for n_steps in PHASE_STEP_COUNTS
        },
        S_0_1_N4_weak=np.asarray(
            weak_check["projected"]["projected"]["S_0_1"]
        ),
        **{
            f"N4_{name}": np.asarray(values)
            for name, values in fourier_content["projected"]["projected"].items()
        },
    )
    summary_path = _write_csv(output_dir / "summary.csv", summary_rows)
    content_path = _write_csv(
        output_dir / "N4_fourier_content.csv",
        list(fourier_content["rows"]),
    )
    reference_metadata = results[REFERENCE_N]["projected"]["metadata"]
    projection_metadata = {
        key: reference_metadata[key]
        for key in (
            "phase_projection_convention",
            "phase_projection_convention_version",
            "target_phase_vector_semantics",
            "normalization",
            "phase_grid",
            "phase_axes",
            "targets",
            "remaining_axis_names",
        )
    }
    metadata_path = write_json(
        output_dir / "metadata.json",
        {
            "example": "ta_three_level_canonical_phase_step_convergence",
            "baseline": "M6 frozen three-level validation parameters",
            "system": build_system().to_dict(include_arrays=True),
            "pulse_parameters": {
                "pump": {"E0_MV_per_cm": 0.30, "energy_eV": 1.55, "sigma_fs": 12.0},
                "probe": {"E0_MV_per_cm": 0.008, "energy_eV": 1.62, "sigma_fs": 7.0},
            },
            "propagation": {
                "solver_mode": "lab_exact",
                "t_start_fs": -1500.0,
                "t_end_fs": 1500.0,
                "dt_fs": 0.2,
                "pure_dephasing_Tphi_fs": {"level_1": 120.0, "level_2": 100.0},
            },
            "delays_fs": DELAYS_FS,
            "delay_convention": "pump_center_fs = probe_center_fs - T; positive T means pump before probe",
            "phase_dimensions": {
                "pump": "pump interaction pulse",
                "probe": "probe interaction pulse",
                "fixed_probe_LO": "excluded from PhaseGrid",
            },
            "phase_grids": {
                f"N{n_steps}": {
                    "pump": uniform_phases(n_steps),
                    "probe": uniform_phases(n_steps),
                }
                for n_steps in PHASE_STEP_COUNTS
            },
            "target_phase_vector": TARGET,
            "readout_plan": build_readout(build_pulses()[1]).to_dict(),
            "observable": "delta_T_over_T = (I_on - I_off) / I_off",
            "denominator_relative_threshold": 1.0e-8,
            "energy_comparison_window_eV": ENERGY_WINDOW_EV,
            "reference_n": REFERENCE_N,
            "reference_wording": "N=4 numerical reference / highest-N baseline, not ground truth",
            "phase_projection_metadata": projection_metadata,
            "solver_case_counts": {
                f"N{n_steps}": results[n_steps]["dynamics"]["solver_case_counts"]
                for n_steps in PHASE_STEP_COUNTS
            },
            "elapsed_s": {
                f"N{n_steps}": results[n_steps]["elapsed_s"]
                for n_steps in PHASE_STEP_COUNTS
            },
            "weak_detector_check": {
                key: value for key, value in weak_check.items() if key != "projected"
            },
            "fourier_content": fourier_content["rows"],
            "interpretation_scope": (
                "Uniform N-step projection distinguishes orders modulo N. Conclusions apply "
                "only to this system, field strength, pulse shape, detector, and delay set."
            ),
            "code_path": str(Path(__file__).resolve()),
            "artifacts": {
                "summary_csv": summary_path,
                "spectra_npz": spectra_path,
                "fourier_content_csv": content_path,
                "plots": plot_paths,
            },
        },
    )
    return {
        "summary_csv": summary_path,
        "metadata_json": metadata_path,
        "spectra_npz": spectra_path,
        "fourier_content_csv": content_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--force-run", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def main() -> dict[str, Any]:
    args = parse_args()
    results = {
        n_steps: run_phase_grid(
            n_steps,
            output_dir=args.output_dir,
            force_run=bool(args.force_run),
        )
        for n_steps in PHASE_STEP_COUNTS
    }
    rows = build_summary_rows(results)
    weak = run_weak_check(results[REFERENCE_N])
    content = run_fourier_content(results[REFERENCE_N])
    plots = [] if args.no_plots else write_overlays(args.output_dir, results)
    paths = save_validation_outputs(
        args.output_dir,
        results,
        summary_rows=rows,
        weak_check=weak,
        fourier_content=content,
        plot_paths=plots,
    )
    for row in rows:
        print(row)
    for name, path in paths.items():
        print(f"{name}: {path}")
    return {"results": results, "summary": rows, "paths": paths}


if __name__ == "__main__":
    main()
