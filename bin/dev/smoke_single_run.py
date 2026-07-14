#!/usr/bin/env python3
"""Milestone 3 generic SingleRunPlan execute smoke.

这是开发 smoke，不是正式 example。它只验证 generic single-run 层可以实际
构造 field、调用 run_case，并产出 absorption-like readout；不做 TA
subtraction、不做 phase cycling projection、不保存输出文件。
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from qudpy_sjh.experiments.pulse_sequence import (  # noqa: E402
    PulseSequenceSpec,
    PulseSpec,
    ReadoutSpec,
    SingleRunCheckpointSettings,
    SingleRunFieldPlan,
    SingleRunPlan,
)
from qudpy_sjh.utils.core import NLevelPhysicalParams, ParaNormalizer  # noqa: E402
from qudpy_sjh.utils.fields import FieldPhySeries  # noqa: E402
from qudpy_sjh.utils.fields.carrier_envelope import make_gaussian_carrier_envelope_field  # noqa: E402


CASE_NAME = "smoke_single_run_absorption"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _spectrum_points(spectrum: dict[str, np.ndarray]) -> int:
    energy = np.asarray(spectrum.get("energy_eV"))
    absorption = np.asarray(spectrum.get("absorption"))
    _require(energy.size > 0, "spectrum energy_eV must contain at least one point.")
    _require(absorption.size > 0, "spectrum absorption must contain at least one point.")
    _require(energy.shape == absorption.shape, "energy_eV and absorption must have the same shape.")
    return int(energy.size)


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

    sequence = PulseSequenceSpec(
        name="smoke_pump_probe",
        pulses=(
            PulseSpec(
                name="pump",
                field_template=pump_template,
                template_center_fs=0.0,
                phase_tag="pump",
                phase_rad=0.0,
                independent_phase=True,
            ),
            PulseSpec(
                name="probe",
                field_template=probe_template,
                template_center_fs=0.0,
                phase_tag="probe",
                phase_rad=0.0,
                independent_phase=True,
            ),
        ),
    )
    field_plan = SingleRunFieldPlan(
        sequence=sequence,
        centers_fs={"pump": -20.0, "probe": 0.0},
        phase_vector={"pump": math.pi, "probe": 0.0},
        case_name=CASE_NAME,
        metadata={"smoke": "milestone_3_execute"},
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
        input_description="Milestone 3 generic SingleRunPlan execute smoke.",
        input_metadata={"script": "bin/dev/smoke_single_run.py"},
    )
    plan = SingleRunPlan(
        base_params=base_params,
        field_plan=field_plan,
        normalizer=ParaNormalizer(),
        readout=ReadoutSpec(
            mode="absorption",
            readout_field_name="probe",
            rel_threshold=1.0e-10,
            zero_padding_factor=4,
            return_intermediates=False,
        ),
        checkpoint=SingleRunCheckpointSettings(enabled=False),
    )

    params = plan.make_params()
    _require(isinstance(params.field, FieldPhySeries), "make_params() must replace field with FieldPhySeries.")
    _require(params.field.sub_field_names == ("pump", "probe"), "field subfield names must be pump/probe.")
    pump_payload = params.field["pump"].to_dict()
    _require(
        bool(pump_payload["metadata"]["phase_override_applied"]),
        "pump phase override must be applied by CarrierEnvelopeField.with_phase(...).",
    )

    result = plan.execute()
    _require(result.case_name == CASE_NAME, "unexpected case_name.")
    _require(result.dynamics_result is not None, "dynamics_result must not be None.")
    trace_error = result.dynamics_result.max_trace_error()
    hermiticity_error = result.dynamics_result.max_hermiticity_error()
    _require(np.isfinite(trace_error), "max_trace_error must be finite.")
    _require(np.isfinite(hermiticity_error), "max_hermiticity_error must be finite.")
    _require(result.readout is not None, "absorption readout must produce a readout result.")
    assert result.readout is not None
    _require(result.readout.mode == "absorption", "readout mode must be absorption.")
    _require(result.readout.spectrum is not None, "absorption readout must produce a spectrum.")
    assert result.readout.spectrum is not None
    for key in ("energy_eV", "omega_fs_inv", "absorption"):
        _require(key in result.readout.spectrum, f"spectrum is missing key: {key}")

    n_spectrum_points = _spectrum_points(result.readout.spectrum)
    energy = np.asarray(result.readout.spectrum["energy_eV"], dtype=float)
    readout_summary = result.readout.to_dict()
    field_metadata = result.field_metadata
    _require(field_metadata["sub_field_names"] == ["pump", "probe"], "field metadata must contain pump/probe.")
    _require(
        field_metadata["fields"][0]["metadata"]["phase_override_applied"] is True,
        "field metadata must record pump phase_override_applied=True.",
    )

    print("smoke_single_run_ok")
    print(f"case_name: {result.case_name}")
    print(f"n_time_points: {readout_summary['n_time_points']}")
    print(f"max_trace_error: {trace_error:.6e}")
    print(f"max_hermiticity_error: {hermiticity_error:.6e}")
    print(f"readout_mode: {result.readout.mode}")
    print(f"n_spectrum_points: {n_spectrum_points}")
    print(f"energy_range_eV: {float(np.min(energy)):.6f} -> {float(np.max(energy)):.6f}")
    print(f"readout_field_source: {readout_summary['readout_field_source']}")


if __name__ == "__main__":
    main()
