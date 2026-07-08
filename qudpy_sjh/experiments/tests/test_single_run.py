from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
from qutip import Qobj

from qudpy_sjh.experiments.pulse_sequence import (
    PulseSequenceSpec,
    PulseSpec,
    ReadoutSpec,
    SingleRunCheckpointSettings,
    SingleRunFieldPlan,
    SingleRunPlan,
    compute_single_run_readout,
)
from qudpy_sjh.experiments.pulse_sequence.single_run import select_readout_field
from qudpy_sjh.utils.core import DynamicsResult, NLevelPhysicalParams
from qudpy_sjh.utils.fields import FieldPhySeries
from qudpy_sjh.utils.fields.carrier_envelope import CarrierEnvelopeField, make_constant_carrier_envelope_field


def _template(name: str, *, energy_eV: float = 1.55) -> CarrierEnvelopeField:
    return make_constant_carrier_envelope_field(
        E0_MV_per_cm=0.01,
        laser_energy_eV=energy_eV,
        name=name,
    )


def _base_params():
    return NLevelPhysicalParams(
        energies_eV=(0.0, 1.55),
        dipole_matrix_D=((0.0, 1.0), (1.0, 0.0)),
        t_start_fs=-2.0,
        t_end_fs=2.0,
        dt_fs=1.0,
        field=_template("base"),
        input_description="base",
        input_metadata={"user_note": "keep_me"},
    )


def _field_plan(*, phase_vector=None) -> tuple[SingleRunFieldPlan, CarrierEnvelopeField, CarrierEnvelopeField]:
    pump_template = _template("pump_template")
    probe_template = _template("probe_template", energy_eV=1.70)
    sequence = PulseSequenceSpec(
        name="pump_probe",
        pulses=(
            PulseSpec(
                name="pump",
                field_template=pump_template,
                phase_tag="pump",
                independent_phase=True,
            ),
            PulseSpec(
                name="probe",
                field_template=probe_template,
                phase_tag="probe",
                independent_phase=True,
            ),
        ),
    )
    plan = SingleRunFieldPlan(
        sequence=sequence,
        centers_fs={"pump": -10.0, "probe": 0.0},
        phase_vector={"pump": 0.0, "probe": 0.0} if phase_vector is None else phase_vector,
        case_name="case_a",
    )
    return plan, pump_template, probe_template


class ReadoutSpecTests(unittest.TestCase):
    def test_readout_spec_validation_accepts_known_modes(self):
        self.assertEqual(ReadoutSpec(mode="none").mode, "none")
        self.assertEqual(ReadoutSpec(mode="polarization").mode, "polarization")
        self.assertEqual(ReadoutSpec(mode="absorption").mode, "absorption")

    def test_readout_spec_validation_rejects_invalid_values(self):
        with self.assertRaises(ValueError):
            ReadoutSpec(mode="ta")
        with self.assertRaises(ValueError):
            ReadoutSpec(number_density_m3=0.0)
        with self.assertRaises(ValueError):
            ReadoutSpec(zero_padding_factor=0)
        with self.assertRaises(ValueError):
            ReadoutSpec(rel_threshold=0.0)


class SingleRunPlanTests(unittest.TestCase):
    def test_make_params_replaces_field_without_mutating_base_system(self):
        base = _base_params()
        field_plan, _, _ = _field_plan()
        plan = SingleRunPlan(
            base_params=base,
            field_plan=field_plan,
            readout=ReadoutSpec(mode="none"),
            input_metadata={"operator": "single_run_test"},
        )

        params = plan.make_params()

        self.assertIsInstance(params, NLevelPhysicalParams)
        self.assertIsInstance(params.field, FieldPhySeries)
        self.assertIsNot(params.field, base.field)
        self.assertIs(base.field, plan.base_params.field)
        self.assertEqual(params.energies_eV, base.energies_eV)
        self.assertEqual(params.dipole_matrix_D, base.dipole_matrix_D)
        self.assertEqual(params.t_start_fs, base.t_start_fs)
        self.assertEqual(params.t_end_fs, base.t_end_fs)
        self.assertEqual(params.dt_fs, base.dt_fs)
        self.assertEqual(params.solver_mode, base.solver_mode)
        self.assertEqual(params.input_metadata["user_note"], "keep_me")
        self.assertEqual(params.input_metadata["operator"], "single_run_test")
        self.assertIn("single_run_workflow", params.input_metadata)
        self.assertEqual(params.input_metadata["single_run_workflow"]["case_name"], "case_a")

    def test_readout_field_selection(self):
        field_plan, _, _ = _field_plan()
        field = field_plan.build_field()

        self.assertIs(select_readout_field(field, None), field)
        self.assertIs(select_readout_field(field, "probe"), field["probe"])
        with self.assertRaises(KeyError):
            select_readout_field(field, "missing")

    def test_phase_override_is_applied_in_make_params(self):
        base = _base_params()
        field_plan, _, _ = _field_plan(phase_vector={"pump": math.pi, "probe": 0.0})
        reference_plan, _, _ = _field_plan(phase_vector={"pump": 0.0, "probe": 0.0})
        plan = SingleRunPlan(base_params=base, field_plan=field_plan)
        reference = SingleRunPlan(base_params=base, field_plan=reference_plan)

        params = plan.make_params()
        reference_params = reference.make_params()
        pump = params.field["pump"]
        reference_pump = reference_params.field["pump"]
        payload = pump.to_dict()

        self.assertTrue(payload["metadata"]["phase_override_applied"])
        self.assertAlmostEqual(payload["phase_rad"], math.pi)
        np.testing.assert_allclose(
            pump.positive_frequency_E_MV_per_cm(np.array([0.0, 1.0])),
            -reference_pump.positive_frequency_E_MV_per_cm(np.array([0.0, 1.0])),
            rtol=1e-12,
            atol=1e-12,
        )

    def test_checkpoint_validation(self):
        with self.assertRaises(ValueError):
            SingleRunCheckpointSettings(enabled=True)

        settings = SingleRunCheckpointSettings(enabled=False)
        self.assertFalse(settings.enabled)

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "case.ckp"
            settings = SingleRunCheckpointSettings(enabled=True, checkpoint_path=path)
            self.assertEqual(settings.checkpoint_path, path)


class SingleRunReadoutTests(unittest.TestCase):
    def test_compute_polarization_readout_without_running_solver(self):
        physical = _base_params()
        result = DynamicsResult(
            mode="lab_exact",
            times=np.array([0.0, 1.0]),
            times_fs=np.array([0.0, 1.0]),
            states=[
                Qobj(np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128)),
                Qobj(np.array([[0.5, 0.1], [0.1, 0.5]], dtype=np.complex128)),
            ],
            parameters=None,
            physical_params=physical,
        )

        readout = compute_single_run_readout(result, readout=ReadoutSpec(mode="polarization"))

        self.assertIsNotNone(readout)
        assert readout is not None
        self.assertEqual(readout.mode, "polarization")
        self.assertEqual(readout.to_dict()["n_time_points"], 2)
        self.assertGreater(readout.to_dict()["max_abs_polarization"], 0.0)


if __name__ == "__main__":
    unittest.main()
