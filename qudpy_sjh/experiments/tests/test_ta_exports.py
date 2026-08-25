from __future__ import annotations

import unittest

import qudpy_sjh.experiments.ta as ta_package
from qudpy_sjh.experiments.ta import (
    TADelayCenters,
    TAPrePCObservable,
    TAPrePCRecipe,
    build_ta_pre_pc_observable,
)


class CanonicalTAExportTests(unittest.TestCase):
    def test_public_surface_contains_only_recipe_first_api(self):
        self.assertEqual(
            set(ta_package.__all__),
            {
                "TADelayCenters",
                "TAPrePCObservable",
                "TAPrePCRecipe",
                "build_ta_pre_pc_observable",
            },
        )
        for value in (
            TADelayCenters,
            TAPrePCObservable,
            TAPrePCRecipe,
            build_ta_pre_pc_observable,
        ):
            self.assertIsNotNone(value)

    def test_delay_convention(self):
        centers = TADelayCenters(delay_fs=100.0, probe_center_fs=10.0)

        self.assertEqual(centers.pump_center_fs, -90.0)
        self.assertEqual(centers.to_dict()["delay_fs"], 100.0)


if __name__ == "__main__":
    unittest.main()
