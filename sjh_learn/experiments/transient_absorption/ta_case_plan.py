"""Full-window transient absorption case planning and execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from sjh_learn.experiments.transient_absorption.absorption_readout import ReadoutWindow
from sjh_learn.experiments.transient_absorption.pulse_scheduling import compute_pulse_centers
from sjh_learn.experiments.transient_absorption.ta_settings import TASettings
from sjh_learn.utils.core.normalization import ParaNormalizer
from sjh_learn.utils.core.results import DynamicsResult
from sjh_learn.utils.core.solvers import run_case
from sjh_learn.utils.fields import make_ta_gaussian_field


RunCaseFn = Callable[..., DynamicsResult]


@dataclass(frozen=True)
class TAFullWindowCasePlan:
    """One delay case that executes exactly one full-window ``run_case`` call."""

    delay_fs: float
    pump_center_fs: float
    probe_center_fs: float
    simulation_window: tuple[float, float]
    readout_window: ReadoutWindow
    field: object
    physical_params: object
    case_name: str

    def execute(
        self,
        *,
        normalizer: ParaNormalizer | None = None,
        rho0=None,
        run_case_fn: RunCaseFn = run_case,
        load_ckp=None,
        save_ckp=None,
        force_run: bool = False,
    ) -> DynamicsResult:
        result = run_case_fn(
            self.physical_params,
            normalizer=normalizer,
            rho0=rho0,
            load_ckp=load_ckp,
            save_ckp=save_ckp,
            force_run=force_run,
        )
        if not isinstance(result, DynamicsResult):
            raise TypeError("TAFullWindowCasePlan.execute expects run_case to return DynamicsResult.")
        return result


def make_case_plan(
    settings: TASettings,
    delay_fs: float,
    *,
    case_name: str | None = None,
) -> TAFullWindowCasePlan:
    pump_center_fs, probe_center_fs = compute_pulse_centers(
        delay_fs,
        probe_center_fs=settings.probe_center_fs,
    )
    readout_window = ReadoutWindow(*settings.readout_window)
    readout_window.validate_inside(settings.simulation_window)
    field = make_ta_gaussian_field(
        probe_delay_fs=float(delay_fs),
        pump_E0_MV_per_cm=float(settings.pump_E0_MV_per_cm),
        probe_E0_MV_per_cm=float(settings.probe_E0_MV_per_cm),
        pump_laser_energy_eV=float(settings.pump_laser_energy_eV),
        probe_laser_energy_eV=settings.probe_laser_energy_eV,
        pump_center_fs=float(pump_center_fs),
        pump_sigma_fs=float(settings.pump_sigma_fs),
        probe_sigma_fs=float(settings.probe_sigma_fs),
        pump_phase_rad=float(settings.pump_phase_rad),
        probe_phase_rad=float(settings.probe_phase_rad),
        name=case_name or f"ta_delay_{delay_fs:g}_fs",
        metadata={
            "experiment": "transient_absorption",
            "execution": "full_window",
            "delay_fs": float(delay_fs),
            "probe_center_fs": float(probe_center_fs),
            "pump_center_fs": float(pump_center_fs),
            "readout_window_fs": [readout_window.start_fs, readout_window.end_fs],
        },
    )
    resolved_case_name = case_name or f"delay_{delay_fs:g}_fs"
    physical_params = settings.physical_params_for_field(
        field,
        delay_fs=delay_fs,
        case_name=resolved_case_name,
    )
    return TAFullWindowCasePlan(
        delay_fs=float(delay_fs),
        pump_center_fs=float(pump_center_fs),
        probe_center_fs=float(probe_center_fs),
        simulation_window=settings.simulation_window,
        readout_window=readout_window,
        field=field,
        physical_params=physical_params,
        case_name=resolved_case_name,
    )


__all__ = ["TAFullWindowCasePlan", "make_case_plan"]

