from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
EXAMPLE_PATH = (
    REPO_ROOT
    / "bin"
    / "examples"
    / "ta"
    / "ta_three_level_canonical_phase_step_convergence.py"
)


def _load_example():
    spec = importlib.util.spec_from_file_location("ta_canonical_m6_example", EXAMPLE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(EXAMPLE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CanonicalTAExampleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.example = _load_example()

    def test_n_changes_only_phase_sampling_identity(self):
        recipes = {n_steps: self.example.build_recipe(n_steps) for n_steps in (2, 3, 4)}
        reference = recipes[4]

        for n_steps, recipe in recipes.items():
            expected = np.asarray(self.example.uniform_phases(n_steps))
            self.assertEqual(recipe.phase_grid.tags, ("pump", "probe"))
            np.testing.assert_allclose(recipe.phase_grid.phases_by_tag["pump"], expected)
            np.testing.assert_allclose(recipe.phase_grid.phases_by_tag["probe"], expected)
            self.assertEqual(recipe.target_phase_vector, {"pump": 0, "probe": 1})
            self.assertEqual(recipe.delays_fs, (-100.0, 0.0, 100.0))
            self.assertEqual(recipe.readout_plan.mode, "full")
            self.assertEqual(recipe.readout_plan.metadata["lo_in_phase_grid"], False)
            self.assertEqual(recipe.base_params.energies_eV, reference.base_params.energies_eV)
            self.assertEqual(
                recipe.base_params.dipole_matrix_D,
                reference.base_params.dipole_matrix_D,
            )
            self.assertEqual(recipe.base_params.t_start_fs, -1500.0)
            self.assertEqual(recipe.base_params.t_end_fs, 1500.0)
            self.assertEqual(recipe.base_params.dt_fs, 0.2)

    def test_canonical_example_does_not_name_heavy_runner_api(self):
        source = EXAMPLE_PATH.read_text(encoding="utf-8")

        for legacy_name in (
            "PhaseCyclingPlan",
            "ProjectedReadoutBundle",
            "TAPhaseCycledPumpProbeResult",
            "ReadoutSpec",
            "execute_with_legacy_readout",
            "relative_response",
        ):
            self.assertNotIn(legacy_name, source)


if __name__ == "__main__":
    unittest.main()
