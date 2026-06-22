import unittest


class FieldsSplitImportTests(unittest.TestCase):
	def test_fields_public_api_imports(self):
		from qudpy_sjh.utils.fields import (
			FieldPhyRoot,
			FieldPhySeries,
			GaussianCarrierFieldPhysical,
			TimeShiftedField,
			make_default_gaussian_carrier_field,
			make_pump_probe_field_from_templates,
			make_ta_gaussian_field,
			make_twodes_gaussian_field,
		)

		self.assertIsNotNone(FieldPhyRoot)
		self.assertIsNotNone(FieldPhySeries)
		self.assertIsNotNone(GaussianCarrierFieldPhysical)
		self.assertIsNotNone(TimeShiftedField)
		self.assertIsNotNone(make_default_gaussian_carrier_field)
		self.assertIsNotNone(make_ta_gaussian_field)
		self.assertIsNotNone(make_pump_probe_field_from_templates)
		self.assertIsNotNone(make_twodes_gaussian_field)

	def test_split_module_imports(self):
		from qudpy_sjh.utils.fields.field_series import FieldPhySeries
		from qudpy_sjh.utils.fields.lab_fields import FieldPhyRoot, TimeShiftedField


		self.assertIsNotNone(FieldPhyRoot)
		self.assertIsNotNone(TimeShiftedField)
		self.assertIsNotNone(FieldPhySeries)


	def test_basic_fields_are_not_reexported_from_lab_fields(self):
		import qudpy_sjh.utils.fields.lab_fields as lab_fields

		self.assertFalse(hasattr(lab_fields, "CarrierFieldPhysical"))
		self.assertFalse(hasattr(lab_fields, "GaussianCarrierFieldPhysical"))
		self.assertFalse(hasattr(lab_fields, "make_default_gaussian_carrier_field"))
		self.assertFalse(hasattr(lab_fields, "rebuild_physical_field"))

	def test_ta_helpers_are_not_reexported_from_field_series(self):
		import qudpy_sjh.utils.fields.field_series as field_series

		self.assertFalse(hasattr(field_series, "TAField"))
		self.assertFalse(hasattr(field_series, "TwoDESField"))
		self.assertFalse(hasattr(field_series, "make_ta_gaussian_field"))
		self.assertFalse(hasattr(field_series, "make_twodes_gaussian_field"))


if __name__ == "__main__":
	unittest.main()
