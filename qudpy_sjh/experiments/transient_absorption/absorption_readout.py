"""Readout-window post-processing for ordinary ``DynamicsResult`` objects."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from qudpy_sjh.utils.core.results import DynamicsResult
from qudpy_sjh.utils.spectroscopy import lab_frame_fft_response_legacy, polarization_C_per_m2


@dataclass(frozen=True)
class ReadoutWindow:
    start_fs: float
    end_fs: float

    def __post_init__(self) -> None:
        if self.end_fs <= self.start_fs:
            raise ValueError("readout window end must be greater than start.")

    def validate_inside(self, simulation_window: tuple[float, float]) -> None:
        start, end = simulation_window
        if self.start_fs < float(start) or self.end_fs > float(end):
            raise ValueError("readout window must be inside the full simulation window.")

    def mask(self, times_fs: np.ndarray) -> np.ndarray:
        times = np.asarray(times_fs, dtype=float)
        return (times >= self.start_fs) & (times <= self.end_fs)


def _time_axis_fs(result: DynamicsResult) -> np.ndarray:
    if result.times_fs is not None:
        return np.asarray(result.times_fs, dtype=float)
    return np.asarray(result.times, dtype=float)


def slice_result_to_readout_window(
    result: DynamicsResult,
    readout_window: ReadoutWindow | tuple[float, float],
) -> dict[str, np.ndarray]:
    window = readout_window if isinstance(readout_window, ReadoutWindow) else ReadoutWindow(*readout_window)
    times_fs = _time_axis_fs(result)
    mask = window.mask(times_fs)
    if int(np.count_nonzero(mask)) < 2:
        raise ValueError("readout window must contain at least two result samples.")
    density = result.density_array()
    return {
        "times_fs": times_fs[mask],
        "times": np.asarray(result.times)[mask],
        "density": density[mask],
        "mask": mask,
    }


def compute_absorption_readout(
    result: DynamicsResult,
    readout_window: ReadoutWindow | tuple[float, float],
    *,
    number_density_m3: float = 1.0e24,
    window: str | None = "hann",
    subtract_mean: bool = True,
    rel_threshold: float = 1.0e-6,
    zero_padding_factor: int = 4,
) -> dict[str, np.ndarray]:
    sliced = slice_result_to_readout_window(result, readout_window)
    physical = result.physical_params
    if physical is None:
        raise ValueError("result.physical_params is required to compute polarization readout.")
    times_fs = sliced["times_fs"]
    density = sliced["density"]
    field_values = result.field_MV_per_cm_values(
        times=sliced["times"],
        times_fs=times_fs,
    )
    if field_values is None:
        raise ValueError("result does not expose a lab-frame field for readout.")
    polarization = polarization_C_per_m2(
        density,
        physical.dipole_matrix_D,
        number_density_m3=number_density_m3,
    )
    coherence = density[:, 0, 1] if density.shape[1] >= 2 else np.zeros(times_fs.shape, dtype=np.complex128)
    response = lab_frame_fft_response_legacy(
        t_fs=times_fs,
        E_MV_per_cm=field_values,
        P_C_per_m2=polarization,
        rhoij=coherence,
        window=window,
        subtract_mean=subtract_mean,
        rel_threshold=rel_threshold,
        zero_padding_factor=zero_padding_factor,
    )
    response["readout_times_fs"] = times_fs
    response["readout_field_MV_per_cm"] = field_values
    response["readout_polarization_C_per_m2"] = polarization
    return response


__all__ = ["ReadoutWindow", "compute_absorption_readout", "slice_result_to_readout_window"]

