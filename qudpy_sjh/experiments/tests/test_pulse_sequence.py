from __future__ import annotations

import math
import unittest

import numpy as np

from qudpy_sjh.experiments.pulse_sequence import (
    FieldGroupSpec,
    PulseSequenceSpec,
    PulseSpec,
    SingleRunFieldPlan,
    is_supported_phase_backend,
    normalize_phase_vector,
    supports_phase_override,
)
from qudpy_sjh.utils.fields import FieldPhyRoot, FieldPhySeries
from qudpy_sjh.utils.fields.carrier_envelope import CarrierEnvelopeField, make_constant_carrier_envelope_field


def _template(name: str) -> CarrierEnvelopeField:
    return make_constant_carrier_envelope_field(
        E0_MV_per_cm=0.01,
        laser_energy_eV=1.55,
        name=name,
    )


class _UnsupportedField(FieldPhyRoot):
    def physical_E_MV_per_cm(self, t_fs):
        return np.ones_like(np.asarray(t_fs, dtype=float))

    def __repr__(self) -> str:
        return "_UnsupportedField()"

    def to_dict(self):
        return {"class": self.__class__.__name__, "rebuildable": False}


class PulseSequenceTests(unittest.TestCase):
    def test_pulse_spec_validation(self):
        with self.assertRaises(ValueError):
            PulseSpec(name="  ", field_template=_template("bad_name"))
        with self.assertRaises(ValueError):
            PulseSpec(name="pump", field_template=_template("bad_tag"), phase_tag=" ")

        pulse = PulseSpec(name="probe", field_template=_template("probe_template"), phase_tag=None)
        self.assertEqual(pulse.phase_tags(), ())

        tagged = PulseSpec(
            name="pump",
            field_template=_template("pump_template"),
            phase_tag="pump",
            independent_phase=True,
        )
        self.assertEqual(tagged.phase_tags(), ("pump",))
        self.assertEqual(tagged.phase_tags(independent_only=True), ("pump",))

    def test_phase_vector_normalization(self):
        self.assertEqual(
            normalize_phase_vector({"pump": math.pi}, known_tags=("pump", "probe")),
            {"pump": math.pi, "probe": 0.0},
        )
        with self.assertRaises(ValueError):
            normalize_phase_vector({"unknown": 0.0}, known_tags=("pump",))

    def test_carrier_envelope_phase_backend_detection(self):
        field = _template("phase_backend")

        self.assertTrue(is_supported_phase_backend(field))
        self.assertTrue(supports_phase_override(field))

    def test_carrier_envelope_with_phase_changes_positive_frequency_field(self):
        field = _template("phase_template")
        times = np.linspace(-2.0, 2.0, 9)

        phase0 = field.with_phase(0.0)
        phase_pi = field.with_phase(math.pi)
        phase_half_pi = field.with_phase(0.5 * math.pi)

        np.testing.assert_allclose(
            phase0.positive_frequency_E_MV_per_cm(times),
            field.positive_frequency_E_MV_per_cm(times),
            rtol=1e-12,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            phase_pi.positive_frequency_E_MV_per_cm(times),
            -field.positive_frequency_E_MV_per_cm(times),
            rtol=1e-12,
            atol=1e-12,
        )
        self.assertFalse(np.allclose(field.physical_E_MV_per_cm(times), phase_half_pi.physical_E_MV_per_cm(times)))
        self.assertEqual(phase_pi.to_dict()["phase_rad"], math.pi)

    def test_sequence_phase_tags_are_stable_and_unique(self):
        pump = PulseSpec(
            name="pump",
            field_template=_template("pump_template"),
            phase_tag="pump",
            independent_phase=True,
        )
        probe = PulseSpec(
            name="probe",
            field_template=_template("probe_template"),
            phase_tag="probe",
            independent_phase=False,
        )
        duplicate = PulseSpec(
            name="readout",
            field_template=_template("readout_template"),
            phase_tag="pump",
            independent_phase=True,
        )
        sequence = PulseSequenceSpec(name="seq", pulses=(pump, probe, duplicate))

        self.assertEqual(sequence.phase_tags(), ("pump", "probe"))
        self.assertEqual(sequence.phase_tags(independent_only=True), ("pump",))

    def test_single_field_construction_applies_phase_and_records_metadata(self):
        pump = PulseSpec(
            name="pump",
            field_template=_template("pump_template"),
            template_center_fs=-20.0,
            phase_tag="pump",
            phase_rad=0.1,
            independent_phase=True,
        )
        probe = PulseSpec(
            name="probe",
            field_template=_template("probe_template"),
            template_center_fs=0.0,
            phase_tag="probe",
            phase_rad=0.2,
            independent_phase=True,
        )
        sequence = PulseSequenceSpec(name="pump_probe", pulses=(pump, probe))
        field = sequence.build_field(
            centers_fs={"pump": -30.0, "probe": 0.0},
            phase_vector={"pump": 0.5, "probe": 0.0},
        )

        self.assertIsInstance(field, FieldPhySeries)
        self.assertEqual(field.sub_field_names, ("pump", "probe"))
        payload = field.to_dict()
        self.assertEqual(payload["metadata"]["phase_tags"], ["pump", "probe"])
        pump_payload = payload["fields"][0]
        pump_metadata = pump_payload["metadata"]
        self.assertEqual(pump_payload["phase_rad"], 0.6)
        self.assertEqual(pump_metadata["phase_tag"], "pump")
        self.assertAlmostEqual(pump_metadata["phase_rad"], 0.6)
        self.assertTrue(pump_metadata["phase_override_applied"])
        self.assertEqual(pump_metadata["requested_center_fs"], -30.0)
        self.assertEqual(pump_metadata["template_center_fs"], -20.0)
        self.assertEqual(pump_metadata["time_shift_fs"], -10.0)

    def test_group_level_phase_tag_takes_priority_and_rejects_internal_tags(self):
        band_a = PulseSpec(name="band_a", field_template=_template("band_a_template"))
        band_b = PulseSpec(name="band_b", field_template=_template("band_b_template"))
        group = FieldGroupSpec(
            name="pump_group",
            pulses=(band_a, band_b),
            phase_tag="pump",
            independent_phase=True,
        )
        sequence = PulseSequenceSpec(name="seq", field_groups=(group,))

        self.assertEqual(group.phase_tags(), ("pump",))
        self.assertEqual(sequence.phase_tags(), ("pump",))
        field = sequence.build_field(
            centers_fs={"band_a": 0.0, "band_b": 0.0},
            phase_vector={"pump": 1.0},
        )
        self.assertEqual(field.sub_field_names, ("pump_group",))
        group_payload = field.to_dict()["fields"][0]
        self.assertEqual(group_payload["metadata"]["phase_tag"], "pump")

        tagged_band = PulseSpec(
            name="tagged_band",
            field_template=_template("tagged_band_template"),
            phase_tag="band",
        )
        with self.assertRaises(ValueError):
            FieldGroupSpec(name="bad_group", pulses=(tagged_band,), phase_tag="group")

    def test_missing_center_raises(self):
        pump = PulseSpec(name="pump", field_template=_template("pump_template"))
        sequence = PulseSequenceSpec(name="seq", pulses=(pump,))
        with self.assertRaises(ValueError):
            sequence.build_field(centers_fs={}, phase_vector=None)

    def test_unknown_phase_tag_raises(self):
        pump = PulseSpec(name="pump", field_template=_template("pump_template"), phase_tag="pump")
        sequence = PulseSequenceSpec(name="seq", pulses=(pump,))
        with self.assertRaises(ValueError):
            sequence.build_field(centers_fs={"pump": 0.0}, phase_vector={"probe": 0.0})

    def test_unsupported_field_strict_phase_override_raises(self):
        pulse = PulseSpec(
            name="custom",
            field_template=_UnsupportedField(),
            phase_tag="custom",
            independent_phase=True,
        )

        with self.assertRaises(TypeError):
            pulse.shifted(center_fs=0.0, phase_vector={"custom": 0.1})

        field = pulse.shifted(center_fs=0.0, phase_vector={"custom": 0.0})
        payload = field.to_dict()
        self.assertFalse(payload["metadata"]["phase_override_applied"])

    def test_single_run_field_plan_does_not_run_solver(self):
        pump = PulseSpec(name="pump", field_template=_template("pump_template"), phase_tag="pump")
        sequence = PulseSequenceSpec(name="seq", pulses=(pump,))
        plan = SingleRunFieldPlan(
            sequence=sequence,
            centers_fs={"pump": 5.0},
            phase_vector={"pump": 0.3},
            case_name="single_run",
        )
        field = plan.build_field()

        self.assertIsInstance(field, FieldPhySeries)
        self.assertEqual(field.name, "single_run")
        self.assertEqual(plan.to_dict()["phase_vector"], {"pump": 0.3})


if __name__ == "__main__":
    unittest.main()
