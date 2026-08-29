from pathlib import Path
import unittest


class FieldsSplitImportTests(unittest.TestCase):
    def test_fields_public_api_is_generic(self):
        import qudpy_sjh.utils.fields as fields

        self.assertSetEqual(
            set(fields.__all__),
            {
                "FieldPhyRoot",
                "FieldPhyCustomed",
                "TimeShiftedField",
                "make_code_field_adapter",
                "FieldPhySeries",
                "iter_scan_params",
            },
        )
        for name in fields.__all__:
            self.assertIsNotNone(getattr(fields, name))

    def test_carrier_envelope_public_api_imports(self):
        from qudpy_sjh.utils.fields.carrier_envelope import (
            CarrierEnvelopeField,
            CarrierSpec,
            GaussianEnvelopeSpec,
            MultiCarrierEnvelopeField,
            make_gaussian_carrier_envelope_field,
            make_multi_carrier_field_from_spectrum,
            make_pump_probe_field_series,
        )

        self.assertIsNotNone(CarrierEnvelopeField)
        self.assertIsNotNone(CarrierSpec)
        self.assertIsNotNone(GaussianEnvelopeSpec)
        self.assertIsNotNone(MultiCarrierEnvelopeField)
        self.assertIsNotNone(make_gaussian_carrier_envelope_field)
        self.assertIsNotNone(make_multi_carrier_field_from_spectrum)
        self.assertIsNotNone(make_pump_probe_field_series)

    def test_legacy_field_api_is_not_reexported(self):
        import qudpy_sjh.utils.fields as fields

        legacy_names = (
            "CarrierFieldPhysical",
            "GaussianCarrierFieldPhysical",
            "TAField",
            "TwoDESField",
            "make_default_gaussian_carrier_field",
            "make_pump_probe_field_from_templates",
            "make_ta_field_from_templates",
            "make_ta_gaussian_field",
            "make_twodes_gaussian_field",
        )
        for name in legacy_names:
            self.assertFalse(hasattr(fields, name), name)

    def test_split_module_imports(self):
        from qudpy_sjh.utils.fields.field_series import FieldPhySeries
        from qudpy_sjh.utils.fields.lab_fields import FieldPhyRoot, TimeShiftedField

        self.assertIsNotNone(FieldPhyRoot)
        self.assertIsNotNone(TimeShiftedField)
        self.assertIsNotNone(FieldPhySeries)

    def test_field_modules_do_not_depend_on_experiment_workflows(self):
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
