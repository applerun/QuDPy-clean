"""Full-window-only transient absorption scaffold."""

from .absorption_readout import (
    ReadoutWindow,
    compute_absorption_readout,
    slice_result_to_readout_window,
)
from .delay_scan_runner import TADelayScanCase, run_delay_scan
from .pulse_scheduling import compute_pulse_centers
from .ta_case_plan import TAFullWindowCasePlan, make_case_plan
from .ta_settings import TASettings

__all__ = [
    "ReadoutWindow",
    "TADelayScanCase",
    "TAFullWindowCasePlan",
    "TASettings",
    "compute_absorption_readout",
    "compute_pulse_centers",
    "make_case_plan",
    "run_delay_scan",
    "slice_result_to_readout_window",
]

