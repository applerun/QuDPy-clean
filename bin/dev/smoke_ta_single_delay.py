#!/usr/bin/env python3
"""Milestone 5.2 single-delay TA recipe v2 execute smoke.

这是开发 smoke，不是正式 example。它只验证单 delay TA recipe v2 可以实际
跑通 pump-probe / probe-only 两条 `SingleRunPlan`，提取 absorption-like
bundle，并按固定 convention 计算：

    S_TA = S_pump_probe - S_probe_only

不保存输出文件，不做 delay scan，不做 phase cycling，不迁移旧 TA demo。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from qudpy_sjh.experiments.pulse_sequence import PulseSpec, ReadoutSpec  # noqa: E402
from qudpy_sjh.experiments.ta import TADelayCenters, TASingleDelayPlan  # noqa: E402
from qudpy_sjh.utils.core import NLevelPhysicalParams, ParaNormalizer  # noqa: E402
from qudpy_sjh.utils.fields.carrier_envelope import make_gaussian_carrier_envelope_field  # noqa: E402


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    pump_template = make_gaussian_carrier_envelope_field(
        E0_MV_per_cm=0.02,
        laser_energy_eV=1.55,
        center_fs=0.0,
        sigma_fs=10.0,
        phase_rad=0.0,
        name="pump_template",
    )
    probe_template = make_gaussian_carrier_envelope_field(
        E0_MV_per_cm=0.005,
        laser_energy_eV=1.55,
        center_fs=0.0,
        sigma_fs=8.0,
        phase_rad=0.0,
        name="probe_template",
    )
    pump = PulseSpec(
        name="pump",
        field_template=pump_template,
        template_center_fs=0.0,
        phase_tag="pump",
        independent_phase=True,
    )
    probe = PulseSpec(
        name="probe",
        field_template=probe_template,
        template_center_fs=0.0,
        phase_tag="probe",
        independent_phase=True,
    )
    base_params = NLevelPhysicalParams(
        basis=("g", "e"),
        energies_eV=(0.0, 1.55),
        dipole_matrix_D=((0.0, 5.0), (5.0, 0.0)),
        t_start_fs=-80.0,
        t_end_fs=120.0,
        dt_fs=1.0,
        field=probe_template,
        solver_mode="lab_exact",
        input_description="Milestone 5.2 single-delay TA recipe v2 execute smoke.",
        input_metadata={"script": "bin/dev/smoke_ta_single_delay.py"},
    )
    plan = TASingleDelayPlan(
        base_params=base_params,
        pump=pump,
        probe=probe,
        delay=TADelayCenters(delay_fs=20.0, probe_center_fs=0.0),
        normalizer=ParaNormalizer(),
        readout=ReadoutSpec(
            mode="absorption",
            readout_field_name="probe",
            rel_threshold=1.0e-10,
            zero_padding_factor=4,
            return_intermediates=False,
        ),
        case_name="smoke_ta_single_delay",
    )

    pair = plan.execute_pair()
    contrast = pair.compute_contrast()

    _require(pair.pump_probe is not None, "pump_probe result must exist.")
    _require(pair.probe_only is not None, "probe_only result must exist.")
    _require(pair.pump_probe_bundle is not None, "pump_probe_bundle must exist.")
    _require(pair.probe_only_bundle is not None, "probe_only_bundle must exist.")
    _require(contrast.delta_absorption.size > 0, "delta_absorption must contain points.")
    _require(contrast.energy_eV.size == contrast.delta_absorption.size, "energy and signal sizes must match.")

    pp_trace = pair.pump_probe.dynamics_result.max_trace_error()
    po_trace = pair.probe_only.dynamics_result.max_trace_error()
    pp_herm = pair.pump_probe.dynamics_result.max_hermiticity_error()
    po_herm = pair.probe_only.dynamics_result.max_hermiticity_error()
    for value, label in (
        (pp_trace, "pump_probe max_trace_error"),
        (po_trace, "probe_only max_trace_error"),
        (pp_herm, "pump_probe max_hermiticity_error"),
        (po_herm, "probe_only max_hermiticity_error"),
    ):
        _require(np.isfinite(value), f"{label} must be finite.")
    _require(np.all(np.isfinite(contrast.energy_eV)), "energy axis must be finite.")

    print("smoke_ta_single_delay_ok")
    print(f"case_name: {contrast.case_name}")
    print(f"delay_fs: {contrast.delay_fs:.6f}")
    print(f"n_points: {contrast.energy_eV.size}")
    print(f"energy_range_eV: {float(np.min(contrast.energy_eV)):.6f} -> {float(np.max(contrast.energy_eV)):.6f}")
    print(f"max_abs_delta_absorption: {float(np.max(np.abs(contrast.delta_absorption))):.6e}")
    print(f"pump_probe max_trace_error: {pp_trace:.6e}")
    print(f"probe_only max_trace_error: {po_trace:.6e}")
    print(f"pump_probe max_hermiticity_error: {pp_herm:.6e}")
    print(f"probe_only max_hermiticity_error: {po_herm:.6e}")


if __name__ == "__main__":
    main()
