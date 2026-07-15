from __future__ import annotations

import unittest

import qudpy_sjh.experiments.ta as ta_package
from qudpy_sjh.experiments.ta import (
    LegacyTADelayScanPlan,
    LegacyTAPlan,
    LegacyTAResult,
    LegacyTAResultIO,
    LegacyTASettings,
    TADelayScanPlan,
    TADelayScanPlanV2,
)
from qudpy_sjh.experiments.ta.ta_case_plan import TADelayScanPlan as LegacyModuleDelayScanPlan
from qudpy_sjh.experiments.ta.ta_case_plan import TAPlan as LegacyModuleTAPlan
from qudpy_sjh.experiments.ta.ta_recipe_v2 import TADelayScanPlan as RecipeV2DelayScanPlan
from qudpy_sjh.experiments.ta.ta_result import TAResult as LegacyModuleTAResult
from qudpy_sjh.experiments.ta.ta_result import TAResultIO as LegacyModuleTAResultIO
from qudpy_sjh.experiments.ta.ta_settings import TASettings as LegacyModuleTASettings


class TAExportTests(unittest.TestCase):
    def test_legacy_aliases_point_to_v1_modules(self):
        self.assertIs(LegacyTASettings, LegacyModuleTASettings)
        self.assertIs(LegacyTAPlan, LegacyModuleTAPlan)
        self.assertIs(LegacyTADelayScanPlan, LegacyModuleDelayScanPlan)
        self.assertIs(LegacyTAResult, LegacyModuleTAResult)
        self.assertIs(LegacyTAResultIO, LegacyModuleTAResultIO)

    def test_v2_scan_alias_points_to_recipe_v2_module(self):
        self.assertIs(TADelayScanPlanV2, RecipeV2DelayScanPlan)

    def test_bare_delay_scan_plan_keeps_legacy_v1_meaning(self):
        self.assertIs(TADelayScanPlan, LegacyModuleDelayScanPlan)
        self.assertIsNot(TADelayScanPlan, RecipeV2DelayScanPlan)

    def test_direct_ta_recipe_v2_import_still_exposes_original_class_name(self):
        from qudpy_sjh.experiments.ta.ta_recipe_v2 import TADelayScanPlan as DirectDelayScanPlan

        self.assertIs(DirectDelayScanPlan, RecipeV2DelayScanPlan)

    def test_all_exports_are_unique_and_explicit(self):
        exported = list(ta_package.__all__)

        self.assertEqual(len(exported), len(set(exported)))
        self.assertIn("TADelayScanPlan", exported)
        self.assertIn("LegacyTADelayScanPlan", exported)
        self.assertIn("TADelayScanPlanV2", exported)
        self.assertIn("TADelayScanMapV2", exported)
        self.assertIn("TADelayScanResultV2", exported)
        self.assertNotIn("TADelayScanMap", exported)
        self.assertNotIn("TADelayScanResult", exported)


if __name__ == "__main__":
    unittest.main()
