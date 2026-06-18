from pathlib import Path
import unittest

import numpy as np

from qudpy_sjh.utils.fields import (
    TAField,
    make_default_gaussian_carrier_field,
    make_pump_probe_field_from_templates,
    make_ta_field_from_templates,
)


def _template(name: str, amplitude: float):
    return make_default_gaussian_carrier_field(
        E0_MV_per_cm=amplitude,
        laser_energy_eV=0.0,
        pulse_center_fs=0.0,
        pulse_sigma_fs=10.0,
        name=name,
    )


class PumpProbeFieldTemplateTests(unittest.TestCase):
    def test_delay_places_pump_before_fixed_probe(self):
        field = make_pump_probe_field_from_templates(
            pump_template=_template("pump_template", 2.0),
            probe_template=_template("probe_template", 1.0),
            delay_fs=200.0,
            probe_center_fs=0.0,
            name="ta_templates",
        )

        self.assertIsInstance(field, TAField)
        self.assertEqual(field.sub_field_names, ("pump", "probe"))
        self.assertAlmostEqual(field["pump"].to_dict()["time_shift_fs"], -200.0)
        self.assertAlmostEqual(field["probe"].to_dict()["time_shift_fs"], 0.0)
        payload = field.to_dict()["metadata"]
        self.assertAlmostEqual(payload["delay_fs"], 200.0)
        self.assertAlmostEqual(payload["probe_center_fs"], 0.0)
        self.assertAlmostEqual(payload["pump_center_fs"], -200.0)
        self.assertEqual(payload["center_rule"], "pump_center_fs = probe_center_fs - delay_fs")

    def test_combined_field_uses_shifted_template_values(self):
        field = make_pump_probe_field_from_templates(
            pump_template=_template("pump_template", 2.0),
            probe_template=_template("probe_template", 1.0),
            delay_fs=200.0,
            probe_center_fs=0.0,
        )

        t_fs = np.array([-200.0, 0.0])
        total = field(t_fs)
        self.assertGreater(total[0], 1.9)
        self.assertGreater(total[1], 0.9)

    def test_templates_are_not_mutated(self):
        pump_template = _template("pump_template", 2.0)
        probe_template = _template("probe_template", 1.0)
        make_pump_probe_field_from_templates(
            pump_template=pump_template,
            probe_template=probe_template,
            delay_fs=50.0,
        )

        self.assertAlmostEqual(pump_template.to_dict()["center_fs"], 0.0)
        self.assertAlmostEqual(probe_template.to_dict()["center_fs"], 0.0)

    def test_metadata_can_be_extended(self):
        field = make_pump_probe_field_from_templates(
            pump_template=_template("pump_template", 2.0),
            probe_template=_template("probe_template", 1.0),
            delay_fs=25.0,
            probe_center_fs=5.0,
            metadata={"case": "smoke"},
        )

        payload = field.to_dict()["metadata"]
        self.assertEqual(payload["case"], "smoke")
        self.assertEqual(payload["experiment"], "TA")
        self.assertEqual(
            payload["template_convention"],
            "pump/probe templates are expected to be centered at 0 fs.",
        )
        self.assertAlmostEqual(payload["pump_center_fs"], -20.0)
        self.assertAlmostEqual(payload["probe_center_fs"], 5.0)

    def test_ta_alias_accepts_probe_delay_name(self):
        field = make_ta_field_from_templates(
            pump_template=_template("pump_template", 2.0),
            probe_template=_template("probe_template", 1.0),
            probe_delay_fs=100.0,
        )

        self.assertAlmostEqual(field.to_dict()["metadata"]["delay_fs"], 100.0)
        self.assertAlmostEqual(field["pump"].to_dict()["time_shift_fs"], -100.0)

    def test_field_modules_do_not_depend_on_ta_workflow_or_piecewise(self):
        field_root = Path(__file__).parents[1] / "fields"
        source = "\n".join(path.read_text(encoding="utf-8") for path in field_root.rglob("*.py"))
        forbidden = (
            "qudpy_sjh.experiments.transient_absorption",
            "piecewise_propagation",
            "dark_propagation",
            "PieceDynamicsResultSeries",
            "execute_piece_sequence",
            "materialize_full",
            "piecewise=",
        )
        for token in forbidden:
            self.assertNotIn(token, source)


if __name__ == "__main__":
    unittest.main()
