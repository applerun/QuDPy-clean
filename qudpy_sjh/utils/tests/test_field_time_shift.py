from __future__ import annotations

import unittest

import numpy as np

from qudpy_sjh.utils.fields import FieldPhySeries
from qudpy_sjh.utils.fields.carrier_envelope import (
    CarrierEnvelopeField,
    make_gaussian_carrier_envelope_field,
)


def _gaussian(center_fs: float = 0.0):
    return make_gaussian_carrier_envelope_field(
        E0_MV_per_cm=1.0,
        laser_energy_eV=0.0,
        center_fs=center_fs,
        sigma_fs=10.0,
        name="template",
    )


def _peak_time(field, center_guess: float) -> float:
    grid = np.linspace(center_guess - 5.0, center_guess + 5.0, 1001)
    values = field(grid)
    return float(grid[int(np.argmax(values))])


class FieldTimeShiftTests(unittest.TestCase):
    def test_zero_shift_matches_original_values(self):
        field = _gaussian()
        shifted = field.time_shifted(0.0)
        grid = np.linspace(-50.0, 50.0, 101)

        np.testing.assert_allclose(shifted(grid), field(grid))
        self.assertIsInstance(shifted, CarrierEnvelopeField)

    def test_negative_shift_moves_gaussian_peak_earlier(self):
        shifted = _gaussian().time_shifted(-200.0)

        self.assertAlmostEqual(_peak_time(shifted, -200.0), -200.0, places=6)

    def test_positive_shift_moves_gaussian_peak_later(self):
        shifted = _gaussian().time_shifted(100.0)

        self.assertAlmostEqual(_peak_time(shifted, 100.0), 100.0, places=6)

    def test_original_field_is_not_mutated(self):
        field = _gaussian()
        shifted = field.time_shifted(-200.0)

        self.assertAlmostEqual(field.to_dict()["center_fs"], 0.0)
        self.assertAlmostEqual(shifted.to_dict()["center_fs"], -200.0)
        self.assertIsNot(shifted, field)
        self.assertIs(shifted.carrier, field.carrier)

    def test_metadata_records_carrier_envelope_shift_semantics(self):
        shifted = _gaussian().time_shifted(-25.0, metadata={"purpose": "test"})
        payload = shifted.to_dict()

        self.assertEqual(payload["metadata"]["time_shift_fs"], -25.0)
        self.assertEqual(payload["metadata"]["purpose"], "test")
        self.assertEqual(
            payload["metadata"]["time_shift_semantics"],
            "envelope center shift under carrier-envelope convention",
        )

    def test_shifted_field_works_inside_field_series(self):
        pump = _gaussian().time_shifted(-20.0, name="pump")
        probe = _gaussian().time_shifted(20.0, name="probe")
        series = FieldPhySeries(fields=(pump, probe), sub_field_names=("pump", "probe"))

        values = series(np.array([-20.0, 20.0]))

        self.assertEqual(values.shape, (2,))
        self.assertGreater(values[0], 1.0)
        self.assertGreater(values[1], 1.0)

    def test_no_ta_piecewise_dark_imports(self):
        import qudpy_sjh.utils.fields.lab_fields as lab_fields

        names = set(vars(lab_fields))
        self.assertNotIn("transient_absorption", names)
        self.assertNotIn("PieceDynamicsResultSeries", names)
        self.assertNotIn("piecewise", names)
        self.assertNotIn("dark_propagation", names)


if __name__ == "__main__":
    unittest.main()
