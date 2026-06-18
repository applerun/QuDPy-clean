#!/usr/bin/env python3
"""Three-level intrinsic TA response workflow.

This is still an example-level / scratch-level implementation, but it is written
in the shape of the future `experiments/ta` module:

    TaExpConfig
    TaDelayResult
    TaResult
    TaExp
    TaResultIO

Changes in this revision
------------------------
1. Time-domain diagnostic figures are saved per delay under:
       figures/selected time traces/
   with names like:
       time traces of delaytime 40 fs.png

2. The delay grid is no longer uniformly sampled. It uses a broader, manually
   defined set of delay times so the response can be seen before pump arrival,
   around pulse overlap, after pump excitation, and after relaxation decay.

3. A kinetic trace at a selected energy is saved:
       figures/kinetic_trace_at_1p55_eV.png
       data/kinetic_trace_at_1p55_eV.csv

4. Every pump+probe delay case is checkpointed and saved under `cases/`.
   `run_case(..., load_ckp=..., save_ckp=...)` is used so re-running after
   plotting-only changes does not repeat the expensive propagation.

5. Existing plotting helpers are reused where they match the task. In
   particular, the electric-field panel in the selected time traces calls
   `plot_field(...)`. TA-specific map/lineout/kinetic/polarization panels
   remain experiment-level plots because plotting.py has no dedicated TA
   response plot helper yet.

6. All heavy numerical operations still go through existing QuDPy APIs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field as dataclass_field
from pathlib import Path
import csv
import json
import math
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

if __package__ is None or __package__ == "":
    # Intended path:
    #   QuDPy/sjh_learn/examples/ta/ta_three_level_intrinsic_response.py
    # so parents[3] is the repository root.
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from sjh_learn.utils.core import (
    NLevelPhysicalParams,
    ParaNormalizer,
    PureDephasingChannel,
    RelaxationChannel,
    run_case,
)
from sjh_learn.utils.fields import make_ta_gaussian_field
from sjh_learn.utils.spectroscopy import lab_frame_fft_response_legacy, polarization_C_per_m2

try:
    from sjh_learn.utils.plotting import plot_field
except Exception:  # pragma: no cover - optional compatibility guard
    plot_field = None

try:
    from sjh_learn.utils.io import save_result_case
except Exception:  # pragma: no cover - optional compatibility guard
    save_result_case = None


EXAMPLE_NAME = "ta_three_level_intrinsic_response"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / EXAMPLE_NAME


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
        return {"real": float(np.real(value)), "imag": float(np.imag(value))}

    if isinstance(value, Path):
        return str(value)

    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(payload), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _safe_delay_label(delay_fs: float) -> str:
    if abs(delay_fs) < 1e-12:
        delay_fs = 0.0
    text = f"{delay_fs:.3f}".rstrip("0").rstrip(".")
    return text.replace("-", "m").replace(".", "p")


def _pretty_delay_text(delay_fs: float) -> str:
    if abs(delay_fs) < 1e-12:
        delay_fs = 0.0
    return f"{delay_fs:.3f}".rstrip("0").rstrip(".")


def _nearest_delay(delays: np.ndarray, target: float) -> float:
    idx = int(np.argmin(np.abs(delays - target)))
    return float(delays[idx])


def _complex_columns(prefix: str, value: complex) -> dict[str, float]:
    return {
        f"Re_{prefix}": float(np.real(value)),
        f"Im_{prefix}": float(np.imag(value)),
        f"abs_{prefix}": float(np.abs(value)),
    }


@dataclass(frozen=True)
class TaExpConfig:
    """Configuration for the three-level intrinsic TA demonstration."""

    example_name: str = EXAMPLE_NAME

    # Broad non-uniform delay grid:
    #   - negative delays: probe before pump, response should be small away from overlap
    #   - dense around 0: pulse overlap / early-time response
    #   - sparse positive tail: population/response decay
    probe_delays_fs: tuple[float, ...] = (
        -300.0, -220.0, -160.0, -110.0, -80.0, -60.0, -45.0, -30.0,
        -20.0, -10.0, 0.0, 10.0, 20.0, 30.0, 45.0, 60.0, 80.0,
        110.0, 150.0, 220.0, 320.0, 460.0, 650.0, 900.0, 1200.0,1500,2000,2500,3000,4000,5000
    )

    # Probe-anchored convention.
    probe_center_fs: float = 0.0

    # Optical fields.
    pump_E0_MV_per_cm: float = 0.18
    probe_E0_MV_per_cm: float = 0.012
    pump_laser_energy_eV: float = 1.55
    probe_laser_energy_eV: float = 1.65
    pump_sigma_fs: float = 12.0
    probe_sigma_fs: float = 7.0
    pump_phase_rad: float = 0.0
    probe_phase_rad: float = 0.0

    # Time grid. By default, the actual bounds are computed from the full delay
    # range and pulse widths so the entire pump/probe sequence is covered.
    auto_time_grid: bool = True
    t_start_fs: float = -6000.0
    t_end_fs: float = 500
    dt_fs: float = 0.2
    time_padding_sigma_factor: float = 10.0
    post_probe_padding_fs: float = 260.0

    # Three-level ladder system.
    basis: tuple[str, ...] = ("g", "e", "f")
    energies_eV: tuple[float, ...] = (0.0, 1.55, 3.30)
    dipole_matrix_D: tuple[tuple[float, ...], ...] = (
        (0.0, 5.0, 0.0),
        (5.0, 0.0, 4.0),
        (0.0, 4.0, 0.0),
    )

    # Demonstration dissipation values. Shorter than the previous draft so the
    # kinetic trace can visibly rise and decay within the broad delay range.
    T1_2_to_1_fs: float = 150.0
    T1_1_to_0_fs: float = 350.0
    Tphi_1_fs: float = 90.0
    Tphi_2_fs: float = 80.0

    # Spectroscopy settings.
    number_density_m3: float = 1.0e24
    window: str | None = "hann"
    subtract_mean: bool = True
    rel_threshold: float = 1.0e-6
    zero_padding_factor: int = 4

    # Plot settings.
    plot_energy_range_eV: tuple[float, float] = (1.25, 1.95)
    cmap: str = "plasma"

    # Kinetic trace.
    kinetic_energy_eV: float = 1.55

    # Per-delay time-domain diagnostic figures.
    selected_time_trace_delays_fs: tuple[float, ...] = (-220.0, -60.0, 0.0, 45.0, 220.0, 650.0, 1200.0)

    # Raw DynamicsResult saving and checkpointing.
    # Every delay is saved by default. Checkpoints are also used by run_case so
    # plot-only changes do not force recomputation.
    save_probe_only_result: bool = True
    save_case_delays_fs: tuple[float, ...] = (0.0, 45.0, 220.0, 650.0, 1200.0)
    save_all_delay_results: bool = True
    save_case_previews: bool = False
    use_checkpoints: bool = True
    force_run: bool = False

    # Map construction.
    allow_energy_axis_interpolation: bool = True

    @classmethod
    def default_three_level_demo(cls) -> "TaExpConfig":
        return cls()


@dataclass
class TaDelayResult:
    delay_fs: float
    field_metadata: dict[str, Any]
    pump_probe_result: Any
    energy_eV: np.ndarray
    omega_fs_inv: np.ndarray
    s_pump_probe: np.ndarray
    s_probe_only: np.ndarray
    s_ta: np.ndarray
    p_over_e_pump_probe: np.ndarray
    p_over_e_probe_only: np.ndarray
    pump_probe_response: dict[str, np.ndarray]


@dataclass
class TaResult:
    config: TaExpConfig
    probe_field_metadata: dict[str, Any]
    probe_only_result: Any
    probe_only_response: dict[str, np.ndarray]
    delay_results: list[TaDelayResult]
    common_energy_eV: np.ndarray
    common_omega_fs_inv: np.ndarray
    delays_fs: np.ndarray
    ta_map: np.ndarray
    pump_probe_map: np.ndarray
    probe_only_spectrum: np.ndarray
    selected_case_results: dict[str, Any] = dataclass_field(default_factory=dict)
    metadata: dict[str, Any] = dataclass_field(default_factory=dict)


class TaExp:
    """Example-level TA experiment runner."""

    def __init__(
        self,
        config: TaExpConfig | None = None,
        *,
        normalizer: ParaNormalizer | None = None,
        output_dir: Path | None = None,
    ) -> None:
        self.config = TaExpConfig.default_three_level_demo() if config is None else config
        self.normalizer = ParaNormalizer(auto_scale=True) if normalizer is None else normalizer
        self.output_dir = DEFAULT_OUTPUT_DIR if output_dir is None else Path(output_dir)
        self._time_bounds_cache: tuple[float, float] | None = None

    def time_bounds_fs(self) -> tuple[float, float]:
        c = self.config
        if not c.auto_time_grid:
            return float(c.t_start_fs), float(c.t_end_fs)

        if self._time_bounds_cache is not None:
            return self._time_bounds_cache

        delays = np.asarray(c.probe_delays_fs, dtype=float)
        pump_centers = float(c.probe_center_fs) - delays
        all_centers = np.concatenate([pump_centers, np.asarray([float(c.probe_center_fs)])])
        sigma_pad = float(c.time_padding_sigma_factor) * max(float(c.pump_sigma_fs), float(c.probe_sigma_fs))

        t_start = float(np.min(all_centers) - sigma_pad)
        t_end = float(max(np.max(all_centers) + sigma_pad, float(c.probe_center_fs) + c.post_probe_padding_fs))

        # Align to dt for cleaner identical time grids.
        dt = float(c.dt_fs)
        t_start = math.floor(t_start / dt) * dt
        t_end = math.ceil(t_end / dt) * dt

        self._time_bounds_cache = (t_start, t_end)
        return self._time_bounds_cache

    def checkpoint_path(self, case_key: str) -> Path | None:
        if not self.config.use_checkpoints:
            return None
        path = self.output_dir / "checkpoints" / f"{case_key}.ckp"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def run_case_with_checkpoint(self, params: NLevelPhysicalParams, *, case_key: str):
        ckp = self.checkpoint_path(case_key)
        if ckp is None:
            return self.run_case_with_checkpoint(params, case_key="probe_only")

        return run_case(
            params,
            normalizer=self.normalizer,
            load_ckp=ckp,
            save_ckp=ckp,
            force_run=bool(self.config.force_run),
        )

    def make_ta_field(self, delay_fs: float):
        """Construct a probe-anchored TAField using existing helper semantics."""

        c = self.config
        pump_center_fs = float(c.probe_center_fs) - float(delay_fs)

        return make_ta_gaussian_field(
            probe_delay_fs=float(delay_fs),
            pump_E0_MV_per_cm=float(c.pump_E0_MV_per_cm),
            probe_E0_MV_per_cm=float(c.probe_E0_MV_per_cm),
            pump_laser_energy_eV=float(c.pump_laser_energy_eV),
            probe_laser_energy_eV=float(c.probe_laser_energy_eV),
            pump_center_fs=pump_center_fs,
            pump_sigma_fs=float(c.pump_sigma_fs),
            probe_sigma_fs=float(c.probe_sigma_fs),
            pump_phase_rad=float(c.pump_phase_rad),
            probe_phase_rad=float(c.probe_phase_rad),
            name=f"ta_delay_{delay_fs:g}_fs",
            metadata={
                "example_name": c.example_name,
                "probe_delay_fs": float(delay_fs),
                "time_anchor": "probe",
                "probe_center_fs": float(c.probe_center_fs),
                "pump_center_fs": pump_center_fs,
            },
        )

    def make_probe_reference_field(self):
        return self.make_ta_field(0.0)["probe"]

    def make_physical_params(self, field, *, case_name: str, input_description: str) -> NLevelPhysicalParams:
        c = self.config
        t_start_fs, t_end_fs = self.time_bounds_fs()

        return NLevelPhysicalParams(
            energies_eV=tuple(float(x) for x in c.energies_eV),
            dipole_matrix_D=tuple(tuple(float(v) for v in row) for row in c.dipole_matrix_D),
            t_start_fs=float(t_start_fs),
            t_end_fs=float(t_end_fs),
            dt_fs=float(c.dt_fs),
            field=field,
            basis=tuple(c.basis),
            relaxation_channels=(
                RelaxationChannel(
                    name="relaxation_2_to_1",
                    from_level=2,
                    to_level=1,
                    T1_fs=float(c.T1_2_to_1_fs),
                ),
                RelaxationChannel(
                    name="relaxation_1_to_0",
                    from_level=1,
                    to_level=0,
                    T1_fs=float(c.T1_1_to_0_fs),
                ),
            ),
            pure_dephasing_channels=(
                PureDephasingChannel(
                    name="pure_dephasing_level_1",
                    level=1,
                    Tphi_fs=float(c.Tphi_1_fs),
                ),
                PureDephasingChannel(
                    name="pure_dephasing_level_2",
                    level=2,
                    Tphi_fs=float(c.Tphi_2_fs),
                ),
            ),
            solver_mode="lab_exact",
            input_description=input_description,
            input_metadata={
                "example_name": c.example_name,
                "case_name": case_name,
                "response_definition": (
                    "S_TA = omega*Im[P_pump_probe/E_probe] "
                    "- omega*Im[P_probe_only/E_probe]"
                ),
                "number_density_m3": float(c.number_density_m3),
                "model_note": (
                    "Three-level ladder demonstration. Parameters are chosen for "
                    "diagnostic visibility, not fitted to a specific material."
                ),
            },
        )

    def run_probe_only(self, probe_field):
        params = self.make_physical_params(
            probe_field,
            case_name="probe_only",
            input_description="Probe-only TA reference shared by all delays.",
        )
        return run_case(params, normalizer=self.normalizer)

    def run_pump_probe(self, delay_fs: float):
        ta_field = self.make_ta_field(delay_fs)
        params = self.make_physical_params(
            ta_field,
            case_name=f"delay_{delay_fs:g}_fs_pump_probe",
            input_description=f"Pump+probe TA run, probe_delay_fs={delay_fs:g}.",
        )
        case_key = f"delay_{_safe_delay_label(delay_fs)}_fs_pump_probe"
        result = self.run_case_with_checkpoint(params, case_key=case_key)
        return ta_field, result

    def polarization_from_result(self, result) -> np.ndarray:
        physical = result.physical_params
        if physical is None:
            raise ValueError("DynamicsResult.physical_params is required.")
        return polarization_C_per_m2(
            result.density_array(),
            physical.dipole_matrix_D,
            float(self.config.number_density_m3),
        )

    def response_from_result(self, result, probe_field) -> dict[str, np.ndarray]:
        c = self.config

        t_fs = np.asarray(result.times_fs, dtype=float)
        if t_fs.ndim != 1 or t_fs.size < 2:
            raise ValueError("result.times_fs must be a 1D array with at least two points.")

        E_probe = np.asarray(probe_field(t_fs), dtype=float)
        P_t = self.polarization_from_result(result)

        return lab_frame_fft_response_legacy(
            t_fs=t_fs,
            E_MV_per_cm=E_probe,
            P_C_per_m2=P_t,
            rhoij=result.matrix_element(0, 1),
            window=c.window,
            subtract_mean=bool(c.subtract_mean),
            rel_threshold=float(c.rel_threshold),
            zero_padding_factor=int(c.zero_padding_factor),
        )

    @staticmethod
    def omega_im_p_over_e(response: dict[str, np.ndarray]) -> np.ndarray:
        return np.asarray(response["omega_fs_inv"], dtype=float) * np.imag(response["P_over_E"])

    def run_delay(
        self,
        delay_fs: float,
        *,
        probe_field,
        probe_only_response: dict[str, np.ndarray],
    ) -> TaDelayResult:
        ta_field, pump_probe_result = self.run_pump_probe(delay_fs)
        pump_probe_response = self.response_from_result(pump_probe_result, probe_field)

        energy = np.asarray(pump_probe_response["energy_eV"], dtype=float)
        omega = np.asarray(pump_probe_response["omega_fs_inv"], dtype=float)

        s_pump_probe = self.omega_im_p_over_e(pump_probe_response)
        s_probe_only = self.omega_im_p_over_e(probe_only_response)

        probe_energy = np.asarray(probe_only_response["energy_eV"], dtype=float)
        if energy.shape == probe_energy.shape and np.allclose(energy, probe_energy):
            s_probe_on_axis = s_probe_only
            p_over_e_probe_on_axis = np.asarray(probe_only_response["P_over_E"])
        elif self.config.allow_energy_axis_interpolation:
            overlap_min = max(float(np.min(energy)), float(np.min(probe_energy)))
            overlap_max = min(float(np.max(energy)), float(np.max(probe_energy)))
            mask = (energy >= overlap_min) & (energy <= overlap_max)
            if not np.any(mask):
                raise ValueError(f"No shared energy range at delay {delay_fs:g} fs.")

            energy = energy[mask]
            omega = omega[mask]
            s_pump_probe = s_pump_probe[mask]
            p_probe = np.asarray(probe_only_response["P_over_E"])
            s_probe_on_axis = np.interp(energy, probe_energy, s_probe_only)
            p_over_e_probe_on_axis = (
                np.interp(energy, probe_energy, np.real(p_probe))
                + 1j * np.interp(energy, probe_energy, np.imag(p_probe))
            )
            pump_probe_response = {
                key: np.asarray(value)[mask] if isinstance(value, np.ndarray) and value.shape == mask.shape else value
                for key, value in pump_probe_response.items()
            }
        else:
            raise ValueError(
                "Pump+probe and probe-only energy axes differ. "
                "Set allow_energy_axis_interpolation=True or fix the response grid."
            )

        s_ta = s_pump_probe - s_probe_on_axis

        return TaDelayResult(
            delay_fs=float(delay_fs),
            field_metadata=ta_field.to_dict(),
            pump_probe_result=pump_probe_result,
            energy_eV=energy,
            omega_fs_inv=omega,
            s_pump_probe=s_pump_probe,
            s_probe_only=s_probe_on_axis,
            s_ta=s_ta,
            p_over_e_pump_probe=np.asarray(pump_probe_response["P_over_E"]),
            p_over_e_probe_only=np.asarray(p_over_e_probe_on_axis),
            pump_probe_response=pump_probe_response,
        )

    def build_common_map(self, delay_results: list[TaDelayResult]):
        if not delay_results:
            raise ValueError("delay_results must not be empty.")

        energy_axes = [np.asarray(item.energy_eV, dtype=float) for item in delay_results]
        overlap_min = max(float(np.min(axis)) for axis in energy_axes)
        overlap_max = min(float(np.max(axis)) for axis in energy_axes)
        if overlap_max <= overlap_min:
            raise ValueError("Delay spectra do not share an overlapping energy range.")

        reference_axis = energy_axes[0]
        mask = (reference_axis >= overlap_min) & (reference_axis <= overlap_max)
        common_energy = reference_axis[mask]
        if common_energy.size < 2:
            raise ValueError("Common energy grid has fewer than two points.")

        ta_rows = []
        pp_rows = []
        pr_rows = []
        omega_rows = []
        for item in delay_results:
            energy = np.asarray(item.energy_eV, dtype=float)
            ta_rows.append(np.interp(common_energy, energy, np.asarray(item.s_ta, dtype=float)))
            pp_rows.append(np.interp(common_energy, energy, np.asarray(item.s_pump_probe, dtype=float)))
            pr_rows.append(np.interp(common_energy, energy, np.asarray(item.s_probe_only, dtype=float)))
            omega_rows.append(np.interp(common_energy, energy, np.asarray(item.omega_fs_inv, dtype=float)))

        common_omega = np.mean(np.vstack(omega_rows), axis=0)
        return (
            common_energy,
            common_omega,
            np.vstack(ta_rows),
            np.vstack(pp_rows),
            np.vstack(pr_rows),
        )

    def select_case_results(
        self,
        *,
        probe_only_result,
        delay_results: list[TaDelayResult],
        ta_map: np.ndarray,
    ) -> dict[str, Any]:
        c = self.config
        selected: dict[str, Any] = {}

        if c.save_probe_only_result:
            selected["probe_only"] = probe_only_result

        if c.save_all_delay_results:
            for item in delay_results:
                selected[f"delay_{_safe_delay_label(item.delay_fs)}_fs_pump_probe"] = item.pump_probe_result
            return selected

        delays = np.asarray([item.delay_fs for item in delay_results], dtype=float)
        requested = {_nearest_delay(delays, target) for target in c.save_case_delays_fs}
        requested.update(_nearest_delay(delays, target) for target in c.selected_time_trace_delays_fs)

        # Also save the delay with maximal absolute TA signal.
        if ta_map.size:
            row_idx = int(np.unravel_index(np.argmax(np.abs(ta_map)), ta_map.shape)[0])
            requested.add(float(delays[row_idx]))

        for item in delay_results:
            if any(math.isclose(item.delay_fs, target, abs_tol=1e-9) for target in requested):
                selected[f"delay_{_safe_delay_label(item.delay_fs)}_fs_pump_probe"] = item.pump_probe_result

        return selected

    def run(self) -> TaResult:
        c = self.config

        probe_field = self.make_probe_reference_field()
        probe_only_result = self.run_probe_only(probe_field)
        probe_only_response = self.response_from_result(probe_only_result, probe_field)

        delay_results: list[TaDelayResult] = []
        for delay_fs in c.probe_delays_fs:
            print(f"Running delay {delay_fs:g} fs...")
            delay_results.append(
                self.run_delay(
                    float(delay_fs),
                    probe_field=probe_field,
                    probe_only_response=probe_only_response,
                )
            )

        (
            common_energy,
            common_omega,
            ta_map,
            pump_probe_map,
            probe_only_map,
        ) = self.build_common_map(delay_results)

        delays = np.asarray([item.delay_fs for item in delay_results], dtype=float)
        selected_case_results = self.select_case_results(
            probe_only_result=probe_only_result,
            delay_results=delay_results,
            ta_map=ta_map,
        )

        t_start_fs, t_end_fs = self.time_bounds_fs()
        kinetic_idx = int(np.argmin(np.abs(common_energy - c.kinetic_energy_eV)))
        kinetic_energy_actual = float(common_energy[kinetic_idx])

        metadata = {
            "example_name": c.example_name,
            "description": (
                "Three-level intrinsic TA response demo. This is an example-level "
                "workflow shaped for a future experiments/ta module."
            ),
            "response_definition": {
                "S_pump_probe": "omega * Im[P_pump_probe(omega, delay) / E_probe(omega)]",
                "S_probe_only": "omega * Im[P_probe_only(omega) / E_probe(omega)]",
                "S_TA": "S_pump_probe - S_probe_only",
            },
            "model_note": (
                "Three-level ladder demonstration; parameters are chosen to make "
                "a visible diagnostic response, not to model a fitted material."
            ),
            "system": {
                "basis": list(c.basis),
                "dimension": len(c.energies_eV),
                "energies_eV": list(c.energies_eV),
                "dipole_matrix_D": c.dipole_matrix_D,
                "transition_notes": [
                    "g<->e near 1.55 eV is pump-resonant.",
                    "e<->f near 1.75 eV can produce an excited-state-absorption-like feature after pump excitation.",
                ],
            },
            "field": {
                "time_anchor": "probe",
                "probe_center_fs": c.probe_center_fs,
                "pump_center_rule": "pump_center_fs = probe_center_fs - probe_delay_fs",
                "pump_E0_MV_per_cm": c.pump_E0_MV_per_cm,
                "probe_E0_MV_per_cm": c.probe_E0_MV_per_cm,
                "pump_laser_energy_eV": c.pump_laser_energy_eV,
                "probe_laser_energy_eV": c.probe_laser_energy_eV,
                "pump_sigma_fs": c.pump_sigma_fs,
                "probe_sigma_fs": c.probe_sigma_fs,
            },
            "time_grid": {
                "auto_time_grid": c.auto_time_grid,
                "t_start_fs": t_start_fs,
                "t_end_fs": t_end_fs,
                "dt_fs": c.dt_fs,
                "n_time_points": int(np.asarray(probe_only_result.times_fs).size),
            },
            "spectroscopy": {
                "number_density_m3": c.number_density_m3,
                "window": c.window,
                "subtract_mean": c.subtract_mean,
                "rel_threshold": c.rel_threshold,
                "zero_padding_factor": c.zero_padding_factor,
                "kinetic_energy_eV_requested": c.kinetic_energy_eV,
                "kinetic_energy_eV_actual": kinetic_energy_actual,
                "note": "FFT, frequency axis, positive-frequency mask and small-denominator mask are delegated to lab_frame_fft_response.",
            },
            "scan": {
                "probe_delays_fs": list(c.probe_delays_fs),
                "n_delays": len(c.probe_delays_fs),
                "selected_time_trace_delays_fs": list(c.selected_time_trace_delays_fs),
                "selected_case_keys": list(selected_case_results.keys()),
                "checkpoints_enabled": c.use_checkpoints,
                "force_run": c.force_run,
            },
            "sanity": {
                "probe_only": {
                    "max_trace_error": float(probe_only_result.max_trace_error()),
                    "max_hermiticity_error": float(probe_only_result.max_hermiticity_error()),
                },
                "pump_probe_max": {
                    "max_trace_error": float(max(item.pump_probe_result.max_trace_error() for item in delay_results)),
                    "max_hermiticity_error": float(max(item.pump_probe_result.max_hermiticity_error() for item in delay_results)),
                },
            },
        }

        return TaResult(
            config=c,
            probe_field_metadata=probe_field.to_dict(),
            probe_only_result=probe_only_result,
            probe_only_response=probe_only_response,
            delay_results=delay_results,
            common_energy_eV=common_energy,
            common_omega_fs_inv=common_omega,
            delays_fs=delays,
            ta_map=ta_map,
            pump_probe_map=pump_probe_map,
            probe_only_spectrum=probe_only_map[0],
            selected_case_results=selected_case_results,
            metadata=metadata,
        )


class TaResultIO:
    """Structured IO for TaResult."""

    def __init__(self, *, output_dir: Path | None = None) -> None:
        self.output_dir = DEFAULT_OUTPUT_DIR if output_dir is None else Path(output_dir)

    def save(self, result: TaResult) -> dict[str, Any]:
        output_dir = self.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        output_files: dict[str, Any] = {}
        output_files.update(self.save_map_data(result, output_dir))
        output_files["delay_spectra"] = self.save_delay_spectra(result, output_dir)
        output_files["figures"] = self.save_figures(result, output_dir)
        output_files["selected_cases"] = self.save_selected_cases(result, output_dir)

        metadata = dict(result.metadata)
        metadata["output_files"] = output_files
        _write_json(output_dir / "meta.json", metadata)
        output_files["metadata"] = str(output_dir / "meta.json")

        return output_files

    def save_map_data(self, result: TaResult, output_dir: Path) -> dict[str, str]:
        data_dir = output_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)

        rows = []
        for i_delay, delay in enumerate(result.delays_fs):
            for i_energy, energy in enumerate(result.common_energy_eV):
                rows.append(
                    {
                        "delay_fs": float(delay),
                        "energy_eV": float(energy),
                        "omega_fs_inv": float(result.common_omega_fs_inv[i_energy]),
                        "S_TA": float(result.ta_map[i_delay, i_energy]),
                        "S_pump_probe": float(result.pump_probe_map[i_delay, i_energy]),
                        "S_probe_only": float(result.probe_only_spectrum[i_energy]),
                    }
                )

        csv_path = data_dir / "ta_map.csv"
        _write_rows(csv_path, rows)

        npz_path = data_dir / "ta_map.npz"
        np.savez_compressed(
            npz_path,
            delays_fs=result.delays_fs,
            energy_eV=result.common_energy_eV,
            omega_fs_inv=result.common_omega_fs_inv,
            S_TA=result.ta_map,
            S_pump_probe=result.pump_probe_map,
            S_probe_only=result.probe_only_spectrum,
        )

        probe_path = data_dir / "probe_reference_spectrum.csv"
        probe_response = result.probe_only_response
        probe_rows = []
        energy = np.asarray(probe_response["energy_eV"], dtype=float)
        omega = np.asarray(probe_response["omega_fs_inv"], dtype=float)
        p_over_e = np.asarray(probe_response["P_over_E"])
        s_probe = TaExp.omega_im_p_over_e(probe_response)
        abs_e = np.asarray(probe_response["abs_E_fft"], dtype=float)
        for idx in range(energy.size):
            probe_rows.append(
                {
                    "energy_eV": float(energy[idx]),
                    "omega_fs_inv": float(omega[idx]),
                    "S_probe_only": float(s_probe[idx]),
                    "abs_E_fft": float(abs_e[idx]),
                    **_complex_columns("P_over_E_probe_only", p_over_e[idx]),
                }
            )
        _write_rows(probe_path, probe_rows)

        kinetic_path = data_dir / "kinetic_trace_at_1p55_eV.csv"
        kinetic_rows = self._kinetic_rows(result)
        _write_rows(kinetic_path, kinetic_rows)

        return {
            "ta_map_csv": str(csv_path),
            "ta_map_npz": str(npz_path),
            "probe_reference_spectrum_csv": str(probe_path),
            "kinetic_trace_at_1p55_eV_csv": str(kinetic_path),
        }

    def _kinetic_rows(self, result: TaResult) -> list[dict[str, float]]:
        idx = int(np.argmin(np.abs(result.common_energy_eV - result.config.kinetic_energy_eV)))
        energy = float(result.common_energy_eV[idx])
        return [
            {
                "delay_fs": float(delay),
                "energy_eV": energy,
                "S_TA": float(result.ta_map[i, idx]),
                "S_pump_probe": float(result.pump_probe_map[i, idx]),
                "S_probe_only": float(result.probe_only_spectrum[idx]),
            }
            for i, delay in enumerate(result.delays_fs)
        ]

    def save_delay_spectra(self, result: TaResult, output_dir: Path) -> list[str]:
        spectra_dir = output_dir / "data" / "delay_spectra"
        spectra_dir.mkdir(parents=True, exist_ok=True)

        paths: list[str] = []
        for item in result.delay_results:
            rows = []
            for idx, energy in enumerate(item.energy_eV):
                rows.append(
                    {
                        "delay_fs": float(item.delay_fs),
                        "energy_eV": float(energy),
                        "omega_fs_inv": float(item.omega_fs_inv[idx]),
                        "S_TA": float(item.s_ta[idx]),
                        "S_pump_probe": float(item.s_pump_probe[idx]),
                        "S_probe_only": float(item.s_probe_only[idx]),
                        **_complex_columns("P_over_E_pump_probe", item.p_over_e_pump_probe[idx]),
                        **_complex_columns("P_over_E_probe_only", item.p_over_e_probe_only[idx]),
                    }
                )
            path = spectra_dir / f"delay_{_safe_delay_label(item.delay_fs)}_fs.csv"
            _write_rows(path, rows)
            paths.append(str(path))

        return paths

    def save_selected_cases(self, result: TaResult, output_dir: Path) -> dict[str, str]:
        cases_root = output_dir
        cases_root.mkdir(parents = True, exist_ok = True)

        paths: dict[str, str] = {}

        for key, dyn_result in result.selected_case_results.items():
            if save_result_case is not None:
                written = save_result_case(
                    dyn_result,
                    cases_root/"res_per_delay",
                    output_data = True,
                    output_preview = bool(result.config.save_case_previews),
                    case_name = key,
                    example_name = result.config.example_name,
                    condition_name = "ta_delay_scan",
                    append_results_csv = True,
                )

                case_dir = written.get("case_dir", cases_root / key)
                paths[key] = str(case_dir)

            else:
                case_dir = cases_root / key
                case_dir.mkdir(parents = True, exist_ok = True)
                ckp_path = case_dir / "result.ckp"
                dyn_result.save_ckp(ckp_path)
                paths[key] = str(ckp_path)

        return paths

    def save_figures(self, result: TaResult, output_dir: Path) -> dict[str, Any]:
        fig_dir = output_dir / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)

        return {
            "ta_map": str(self._plot_ta_map(result, fig_dir)),
            "selected_delay_lineouts": str(self._plot_selected_delay_lineouts(result, fig_dir)),
            "probe_fft_mask": str(self._plot_probe_fft_mask(result, fig_dir)),
            "kinetic_trace_at_1p55_eV": str(self._plot_kinetic_trace(result, fig_dir)),
            "selected_time_traces": self._plot_selected_time_traces(result, fig_dir),
        }

    def _plot_ta_map(self, result: TaResult, fig_dir: Path) -> Path:
        c = result.config
        energy = result.common_energy_eV
        delays = result.delays_fs
        values = result.ta_map

        e_min, e_max = c.plot_energy_range_eV
        mask = (energy >= e_min) & (energy <= e_max)
        if not np.any(mask):
            mask = np.ones_like(energy, dtype=bool)

        fig, ax = plt.subplots(figsize=(6.8, 4.4))
        mesh = ax.pcolormesh(
            energy[mask],
            delays,
            values[:, mask],
            shading="auto",
            cmap=c.cmap,
        )
        ax.set_xlabel("Energy (eV)")
        ax.set_ylabel("Probe delay (fs)")
        ax.set_title("Intrinsic TA response")
        cbar = fig.colorbar(mesh, ax=ax)
        cbar.set_label("S_TA (arb.)")
        fig.tight_layout()

        path = fig_dir / "ta_map.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        return path

    def _plot_selected_delay_lineouts(self, result: TaResult, fig_dir: Path) -> Path:
        c = result.config
        energy = result.common_energy_eV
        e_min, e_max = c.plot_energy_range_eV
        energy_mask = (energy >= e_min) & (energy <= e_max)
        if not np.any(energy_mask):
            energy_mask = np.ones_like(energy, dtype=bool)

        target_delays = sorted({_nearest_delay(result.delays_fs, x) for x in (-220.0, -60.0, 0.0, 45.0, 220.0, 650.0, 1200.0)})
        colors = plt.get_cmap("plasma")(np.linspace(0.08, 0.92, len(target_delays)))

        fig, ax = plt.subplots(figsize=(6.8, 4.2))

        for color, target in zip(colors, target_delays):
            idx = int(np.argmin(np.abs(result.delays_fs - target)))
            ax.plot(
                energy[energy_mask],
                result.ta_map[idx, energy_mask],
                linewidth=1.0,
                color=color,
                label=f"{result.delays_fs[idx]:g} fs",
            )

        ax.set_title("Selected TA lineouts")
        ax.set_xlabel("Energy (eV)")
        ax.set_ylabel("S_TA (arb.)")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, ncols=2, fontsize=8)
        fig.tight_layout()

        path = fig_dir / "selected_delay_lineouts.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        return path

    def _plot_probe_fft_mask(self, result: TaResult, fig_dir: Path) -> Path:
        c = result.config
        response = result.probe_only_response
        energy = np.asarray(response["energy_eV"], dtype=float)
        abs_e = np.asarray(response["abs_E_fft"], dtype=float)

        e_min, e_max = c.plot_energy_range_eV
        mask = (energy >= e_min) & (energy <= e_max)
        if not np.any(mask):
            mask = np.ones_like(energy, dtype=bool)

        y = abs_e[mask]
        y_max = float(np.max(y)) if y.size else 1.0
        if y_max > 0:
            y = y / y_max

        fig, ax = plt.subplots(figsize=(6.4, 4.0))
        ax.plot(energy[mask], y, linewidth=1.5)
        ax.set_title("Probe FFT support after response mask")
        ax.set_xlabel("Energy (eV)")
        ax.set_ylabel("normalized |E_probe(ω)|")
        ax.grid(alpha=0.25)
        fig.tight_layout()

        path = fig_dir / "probe_fft_mask.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        return path

    def _plot_kinetic_trace(self, result: TaResult, fig_dir: Path) -> Path:
        idx = int(np.argmin(np.abs(result.common_energy_eV - result.config.kinetic_energy_eV)))
        energy = float(result.common_energy_eV[idx])

        fig, ax = plt.subplots(figsize=(6.4, 4.0))
        ax.plot(result.delays_fs, result.ta_map[:, idx], marker="o", linewidth=1.2)
        ax.axhline(0.0, linewidth=0.8)
        ax.set_title(f"TA kinetics at {energy:.3f} eV")
        ax.set_xlabel("Probe delay (fs)")
        ax.set_ylabel("S_TA (arb.)")
        ax.grid(alpha=0.25)
        fig.tight_layout()

        path = fig_dir / "kinetic_trace_at_1p55_eV.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        return path

    def _plot_selected_time_traces(self, result: TaResult, fig_dir: Path) -> dict[str, str]:
        trace_dir = fig_dir / "selected time traces"
        trace_dir.mkdir(parents=True, exist_ok=True)

        selected_paths: dict[str, str] = {}
        targets = sorted({_nearest_delay(result.delays_fs, x) for x in result.config.selected_time_trace_delays_fs})

        for target in targets:
            item = min(result.delay_results, key=lambda x: abs(x.delay_fs - target))
            path = self._plot_time_traces_for_delay(result, item, trace_dir)
            selected_paths[f"delay_{_safe_delay_label(item.delay_fs)}_fs"] = str(path)

        return selected_paths

    def _plot_time_traces_for_delay(self, result: TaResult, item: TaDelayResult, trace_dir: Path) -> Path:
        dyn = item.pump_probe_result
        probe = result.probe_only_result

        t_fs = np.asarray(dyn.times_fs, dtype=float)
        t_probe = np.asarray(probe.times_fs, dtype=float)
        if t_fs.shape != t_probe.shape or not np.allclose(t_fs, t_probe):
            raise ValueError("Pump+probe and probe-only time axes do not match.")

        P_pp = TaExp(result.config, output_dir=self.output_dir).polarization_from_result(dyn)
        P_probe = TaExp(result.config, output_dir=self.output_dir).polarization_from_result(probe)
        delta = np.real(P_pp - P_probe)

        def _norm(y):
            y = np.asarray(y, dtype=float)
            scale = float(np.max(np.abs(y))) if y.size else 1.0
            if scale <= 0:
                scale = 1.0
            return y / scale

        fig, axes = plt.subplots(4, 1, figsize=(7.4, 7.2), sharex=True)
        linewidth = 0.5

        # Reuse existing plotting.py helper for the field panel when available.
        # This panel is intentionally the only one delegated to plotting.py:
        # plotting.py has no dedicated helper for P(t), ΔP(t), or TA response.
        if plot_field is not None and getattr(dyn, "physical_params", None) is not None:
            plot_field(dyn.physical_params.field, t_fs, ax=axes[0])
            for line in axes[0].lines:
                line.set_linewidth(linewidth)
            axes[0].set_ylabel("E(t)")
        else:
            E_total = np.asarray(dyn.field_MV_per_cm_values(), dtype=float)
            axes[0].plot(t_fs, E_total, linewidth=linewidth)
            axes[0].set_ylabel("E(t)")

        axes[0].set_title(f"Time traces of delay time {_pretty_delay_text(item.delay_fs)} fs")

        axes[1].plot(t_fs, _norm(np.real(P_pp)), linewidth=linewidth)
        axes[1].set_ylabel("Re P_pp")

        axes[2].plot(t_fs, _norm(np.real(P_probe)), linewidth=linewidth)
        axes[2].set_ylabel("Re P_probe")

        axes[3].plot(t_fs, _norm(delta), linewidth=linewidth)
        axes[3].set_ylabel("Re ΔP")
        axes[3].set_xlabel("Time (fs)")

        for ax in axes:
            ax.grid(alpha=0.25)

        fig.tight_layout()

        path = trace_dir / f"time traces of delaytime {_pretty_delay_text(item.delay_fs)} fs.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        return path


def main() -> None:
    output_dir = DEFAULT_OUTPUT_DIR
    config = TaExpConfig.default_three_level_demo()

    exp = TaExp(config, output_dir=output_dir)
    result = exp.run()

    output_files = TaResultIO(output_dir=output_dir).save(result)

    print("Three-level intrinsic TA example finished.")
    print(f"n delays          : {len(result.delays_fs)}")
    print(f"time grid fs      : {result.metadata['time_grid']['t_start_fs']} to {result.metadata['time_grid']['t_end_fs']}")
    print(f"energy points     : {result.common_energy_eV.size}")
    print(f"selected cases    : {list(result.selected_case_results.keys())}")
    print(f"output directory  : {output_dir}")
    print(f"TA map            : {output_files['figures']['ta_map']}")
    print(f"TA kinetics       : {output_files['figures']['kinetic_trace_at_1p55_eV']}")


if __name__ == "__main__":
    main()
