"""Settings for a simple full-window transient absorption delay scan."""

from __future__ import annotations

from dataclasses import dataclass

from qudpy_sjh.experiments.transient_absorption.pulse_scheduling import compute_pulse_centers
from qudpy_sjh.utils.core.parameters import NLevelPhysicalParams, PureDephasingChannel, RelaxationChannel


@dataclass(frozen=True)
class TASettings:
    """Configuration for full-window-only transient absorption cases."""

    t_start_fs: float = -300.0
    t_end_fs: float = 300.0
    dt_fs: float = 2.0
    delays_fs: tuple[float, ...] = (-100.0, 0.0, 100.0)

    probe_center_fs: float = 0.0
    readout_start_fs: float = -80.0
    readout_end_fs: float = 160.0

    pump_E0_MV_per_cm: float = 0.02
    probe_E0_MV_per_cm: float = 0.005
    pump_laser_energy_eV: float = 1.55
    probe_laser_energy_eV: float | None = None
    pump_sigma_fs: float = 12.0
    probe_sigma_fs: float = 8.0
    pump_phase_rad: float = 0.0
    probe_phase_rad: float = 0.0

    energies_eV: tuple[float, ...] = (0.0, 1.55)
    dipole_matrix_D: tuple[tuple[complex, ...], ...] = ((0.0, 1.0), (1.0, 0.0))
    basis: tuple[str, ...] | None = ("g", "e")
    relaxation_channels: tuple[RelaxationChannel, ...] = ()
    pure_dephasing_channels: tuple[PureDephasingChannel, ...] = ()
    solver_mode: str = "lab_exact"

    number_density_m3: float = 1.0e24
    fft_window: str | None = "hann"
    subtract_mean: bool = True
    rel_threshold: float = 1.0e-6
    zero_padding_factor: int = 4

    def __post_init__(self) -> None:
        if self.t_end_fs <= self.t_start_fs:
            raise ValueError("t_end_fs must be greater than t_start_fs.")
        if self.dt_fs <= 0.0:
            raise ValueError("dt_fs must be positive.")
        if self.readout_end_fs <= self.readout_start_fs:
            raise ValueError("readout_end_fs must be greater than readout_start_fs.")
        if self.readout_start_fs < self.t_start_fs or self.readout_end_fs > self.t_end_fs:
            raise ValueError("readout window must be inside the full simulation window.")
        if not self.delays_fs:
            raise ValueError("delays_fs must not be empty.")

    @property
    def simulation_window(self) -> tuple[float, float]:
        return float(self.t_start_fs), float(self.t_end_fs)

    @property
    def readout_window(self) -> tuple[float, float]:
        return float(self.readout_start_fs), float(self.readout_end_fs)

    def pump_center_fs_for_delay(self, delay_fs: float) -> float:
        pump_center, _probe_center = compute_pulse_centers(
            delay_fs,
            probe_center_fs=self.probe_center_fs,
        )
        return pump_center

    def physical_params_for_field(
        self,
        field,
        *,
        delay_fs: float,
        case_name: str | None = None,
    ) -> NLevelPhysicalParams:
        return NLevelPhysicalParams(
            energies_eV=tuple(float(value) for value in self.energies_eV),
            dipole_matrix_D=tuple(tuple(complex(item) for item in row) for row in self.dipole_matrix_D),
            t_start_fs=float(self.t_start_fs),
            t_end_fs=float(self.t_end_fs),
            dt_fs=float(self.dt_fs),
            field=field,
            basis=self.basis,
            relaxation_channels=self.relaxation_channels,
            pure_dephasing_channels=self.pure_dephasing_channels,
            solver_mode=self.solver_mode,
            input_description="Full-window transient absorption delay case.",
            input_metadata={
                "experiment": "transient_absorption",
                "case_name": case_name,
                "delay_fs": float(delay_fs),
                "probe_center_fs": float(self.probe_center_fs),
                "pump_center_fs": self.pump_center_fs_for_delay(delay_fs),
                "readout_window_fs": [float(self.readout_start_fs), float(self.readout_end_fs)],
                "execution": "full_window_run_case",
            },
        )


__all__ = ["TASettings"]

