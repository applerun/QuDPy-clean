"""Pulse timing convention for full-window transient absorption."""

from __future__ import annotations


def compute_pulse_centers(
    delay_fs: float,
    probe_center_fs: float = 0.0,
) -> tuple[float, float]:
    """Return ``(pump_center_fs, probe_center_fs)`` for the fixed-probe convention."""

    probe_center = float(probe_center_fs)
    pump_center = probe_center - float(delay_fs)
    return pump_center, probe_center


__all__ = ["compute_pulse_centers"]

