from __future__ import annotations

import math
import unittest

import numpy as np

from qudpy_sjh.experiments import (
    PHASE_PROJECTION_CONVENTION,
    PHASE_PROJECTION_CONVENTION_VERSION,
    TARGET_PHASE_VECTOR_SEMANTICS,
    PhaseGrid,
    build_uniform_phase_grid,
    project_phase_orders,
)
from qudpy_sjh.experiments.pulse_sequence import phase_projection as pure_module


class UniformPhaseGridTests(unittest.TestCase):
    def test_uniform_helper_accepts_shared_or_per_tag_step_counts(self):
        shared = build_uniform_phase_grid(("pump", "probe"), n_steps=4)
        unequal = build_uniform_phase_grid(
            ("pump", "probe"),
            n_steps={"pump": 4, "probe": 3},
        )

        self.assertEqual(tuple(len(shared.phases_by_tag[tag]) for tag in shared.tags), (4, 4))
        self.assertEqual(tuple(len(unequal.phases_by_tag[tag]) for tag in unequal.tags), (4, 3))
        self.assertEqual(len(unequal), 12)

    def test_uniform_helper_rejects_ambiguous_step_mappings(self):
        with self.assertRaisesRegex(ValueError, "exactly match"):
            build_uniform_phase_grid(("pump", "probe"), n_steps={"pump": 4})
        with self.assertRaisesRegex(ValueError, "positive integer"):
            build_uniform_phase_grid(("pump",), n_steps={"pump": 2.5})


class PureNamedAxisProjectionTests(unittest.TestCase):
    def test_pure_module_has_no_runner_readout_or_recipe_dependency(self):
        namespace = vars(pure_module)

        self.assertNotIn("SingleRunPlan", namespace)
        self.assertNotIn("ReadoutPlan", namespace)
        self.assertNotIn("TAPrePCRecipe", namespace)

    def test_one_dimensional_projection_preserves_payload_axis(self):
        grid = build_uniform_phase_grid(("pump",), n_steps=8)
        phases = np.asarray(grid.phases_by_tag["pump"])
        energy = np.asarray([1.4, 1.5, 1.6])
        payload = np.asarray([2.0, -1.0, 0.5])
        data = np.exp(-2j * phases)[:, None] * payload[None, :]

        result = project_phase_orders(
            data,
            axis_names=("phase:pump", "energy_eV"),
            axis_values={"phase:pump": phases, "energy_eV": energy},
            phase_grid=grid,
            targets={"order_2": {"pump": 2}},
        )

        self.assertEqual(result["axis_names"], ("energy_eV",))
        np.testing.assert_allclose(result["axis_values"]["energy_eV"], energy)
        np.testing.assert_allclose(result["projected"]["order_2"], payload, atol=1.0e-12)

    def test_two_dimensional_unequal_grid_preserves_nonphase_order(self):
        grid = build_uniform_phase_grid(
            ("pump", "probe"),
            n_steps={"pump": 4, "probe": 3},
        )
        pump = np.asarray(grid.phases_by_tag["pump"])
        probe = np.asarray(grid.phases_by_tag["probe"])
        delay = np.asarray([-10.0, 20.0])
        energy = np.asarray([1.4, 1.5, 1.6])
        payload = np.asarray([[1.0, 2.0, 3.0], [-0.5, 0.25, 1.5]])
        harmonic = np.exp(-1j * (pump[:, None] - probe[None, :]))
        data = payload[:, None, None, :] * harmonic[None, :, :, None]

        result = project_phase_orders(
            data,
            axis_names=("T", "phase:pump", "phase:probe", "energy_eV"),
            axis_values={
                "T": delay,
                "phase:pump": pump,
                "phase:probe": probe,
                "energy_eV": energy,
            },
            phase_grid=grid,
            targets={"S_1_minus1": {"pump": 1, "probe": -1}},
        )

        self.assertEqual(result["axis_names"], ("T", "energy_eV"))
        np.testing.assert_allclose(result["projected"]["S_1_minus1"], payload, atol=1.0e-12)
        self.assertEqual(result["metadata"]["normalization"]["divisor"], 12)

    def test_noncontiguous_phase_axes_are_found_by_name(self):
        grid = build_uniform_phase_grid(
            ("pump", "probe"),
            n_steps={"pump": 4, "probe": 3},
        )
        pump = np.asarray(grid.phases_by_tag["pump"])
        probe = np.asarray(grid.phases_by_tag["probe"])
        energy = np.asarray([1.4, 1.5, 1.6])
        delay = np.asarray([-10.0, 20.0])
        payload = np.asarray([[1.0, 2.0, 3.0], [-0.5, 0.25, 1.5]])
        canonical = (
            payload[:, None, None, :]
            * np.exp(-1j * (pump[:, None] - probe[None, :]))[None, :, :, None]
        )
        data = np.transpose(canonical, (3, 1, 0, 2))

        result = project_phase_orders(
            data,
            axis_names=("energy_eV", "phase:pump", "T", "phase:probe"),
            axis_values={
                "energy_eV": energy,
                "phase:pump": pump,
                "T": delay,
                "phase:probe": probe,
            },
            phase_grid=grid,
            targets={"channel": {"pump": 1, "probe": -1}},
        )

        self.assertEqual(result["axis_names"], ("energy_eV", "T"))
        np.testing.assert_allclose(result["projected"]["channel"], payload.T, atol=1.0e-12)

    def test_explicit_phase_axis_mapping_supports_noncanonical_names(self):
        grid = build_uniform_phase_grid(("pump",), n_steps=4)
        phases = np.asarray(grid.phases_by_tag["pump"])
        data = np.exp(-1j * phases)

        result = project_phase_orders(
            data,
            axis_names=("cycled_pump",),
            axis_values={"cycled_pump": phases},
            phase_grid=grid,
            phase_axes={"pump": "cycled_pump"},
            targets={"target": {"pump": 1}},
        )

        self.assertEqual(result["axis_names"], ())
        np.testing.assert_allclose(result["projected"]["target"], 1.0, atol=1.0e-12)

    def test_multiple_targets_are_extracted_from_one_array(self):
        grid = build_uniform_phase_grid(
            ("pump", "probe"),
            n_steps={"pump": 5, "probe": 4},
        )
        pump = np.asarray(grid.phases_by_tag["pump"])
        probe = np.asarray(grid.phases_by_tag["probe"])
        amplitude_a = 2.0 + 0.5j
        amplitude_b = -0.25 + 0.75j
        data = (
            amplitude_a * np.exp(-1j * (pump[:, None] + probe[None, :]))
            + amplitude_b * np.exp(-1j * (-pump[:, None] + probe[None, :]))
        )

        result = project_phase_orders(
            data,
            axis_names=("phase:pump", "phase:probe"),
            axis_values={"phase:pump": pump, "phase:probe": probe},
            phase_grid=grid,
            targets={
                "A": {"pump": 1, "probe": 1},
                "B": {"pump": -1, "probe": 1},
            },
        )

        np.testing.assert_allclose(result["projected"]["A"], amplitude_a, atol=1.0e-12)
        np.testing.assert_allclose(result["projected"]["B"], amplitude_b, atol=1.0e-12)

    def test_alias_equivalent_targets_are_allowed_and_equal(self):
        grid = build_uniform_phase_grid(("pump",), n_steps=4)
        phases = np.asarray(grid.phases_by_tag["pump"])
        data = np.exp(-1j * phases)

        result = project_phase_orders(
            data,
            axis_names=("phase:pump",),
            axis_values={"phase:pump": phases},
            phase_grid=grid,
            targets={"m1": {"pump": 1}, "m5": {"pump": 5}},
        )

        np.testing.assert_allclose(result["projected"]["m1"], result["projected"]["m5"])
        np.testing.assert_allclose(result["projected"]["m1"], 1.0, atol=1.0e-12)

    def test_normalize_false_returns_unnormalized_phase_sum(self):
        grid = build_uniform_phase_grid(("pump",), n_steps=4)
        phases = np.asarray(grid.phases_by_tag["pump"])
        result = project_phase_orders(
            np.exp(-1j * phases),
            axis_names=("phase:pump",),
            axis_values={"phase:pump": phases},
            phase_grid=grid,
            targets={"m1": {"pump": 1}},
            normalize=False,
        )

        np.testing.assert_allclose(result["projected"]["m1"], 4.0, atol=1.0e-12)
        self.assertEqual(result["metadata"]["normalization"]["divisor"], 1)

    def test_nonuniform_values_execute_documented_equal_weight_sum(self):
        phases = np.asarray([0.0, 0.71, 2.13])
        grid = PhaseGrid({"probe": tuple(phases)})
        data = np.asarray([1.0, 2.0, 4.0])

        result = project_phase_orders(
            data,
            axis_names=("phase:probe",),
            axis_values={"phase:probe": phases},
            phase_grid=grid,
            targets={"m1": {"probe": 1}},
        )

        expected = np.mean(data * np.exp(1j * phases))
        np.testing.assert_allclose(result["projected"]["m1"], expected)
        self.assertIn("equal-weight", result["metadata"]["nonuniform_phase_note"])

    def test_metadata_uses_frozen_m1_convention(self):
        grid = build_uniform_phase_grid(("pump",), n_steps=2)
        phases = np.asarray(grid.phases_by_tag["pump"])
        result = project_phase_orders(
            np.ones(2),
            axis_names=("phase:pump",),
            axis_values={"phase:pump": phases},
            phase_grid=grid,
            targets={"population": {}},
        )

        metadata = result["metadata"]
        self.assertEqual(metadata["phase_projection_convention"], PHASE_PROJECTION_CONVENTION)
        self.assertEqual(
            metadata["phase_projection_convention_version"],
            PHASE_PROJECTION_CONVENTION_VERSION,
        )
        self.assertEqual(metadata["target_phase_vector_semantics"], TARGET_PHASE_VECTOR_SEMANTICS)
        self.assertEqual(result["targets"]["population"], {"pump": 0})


class ProjectionValidationTests(unittest.TestCase):
    def setUp(self):
        self.grid = build_uniform_phase_grid(("pump",), n_steps=2)
        self.phases = np.asarray(self.grid.phases_by_tag["pump"])

    def _project(self, **overrides):
        arguments = {
            "data": np.ones((2, 3)),
            "axis_names": ("phase:pump", "energy_eV"),
            "axis_values": {
                "phase:pump": self.phases,
                "energy_eV": np.asarray([1.4, 1.5, 1.6]),
            },
            "phase_grid": self.grid,
            "targets": {"target": {"pump": 0}},
        }
        arguments.update(overrides)
        return project_phase_orders(**arguments)

    def test_axis_contract_failures(self):
        with self.assertRaisesRegex(ValueError, "unique"):
            self._project(axis_names=("phase:pump", "phase:pump"))
        with self.assertRaisesRegex(ValueError, "missing from axis_names"):
            self._project(
                axis_names=("pump_axis", "energy_eV"),
                axis_values={
                    "pump_axis": self.phases,
                    "energy_eV": np.asarray([1.4, 1.5, 1.6]),
                },
            )
        with self.assertRaisesRegex(ValueError, "one-dimensional with length 3"):
            self._project(
                axis_values={"phase:pump": self.phases, "energy_eV": np.asarray([1.4, 1.5])}
            )
        with self.assertRaisesRegex(ValueError, "does not match PhaseGrid"):
            self._project(
                axis_values={
                    "phase:pump": np.asarray([0.0, 1.0]),
                    "energy_eV": np.asarray([1.4, 1.5, 1.6]),
                }
            )
        with self.assertRaisesRegex(ValueError, "must contain phase axis"):
            self._project(axis_values={"energy_eV": np.asarray([1.4, 1.5, 1.6])})

    def test_target_validation_failures(self):
        with self.assertRaisesRegex(ValueError, "unknown phase tags"):
            self._project(targets={"target": {"probe": 1}})
        with self.assertRaisesRegex(ValueError, "must be an integer"):
            self._project(targets={"target": {"pump": 1.5}})
        with self.assertRaisesRegex(ValueError, "not bool"):
            self._project(targets={"target": {"pump": True}})
        with self.assertRaisesRegex(ValueError, "non-empty mapping"):
            self._project(targets={})

    def test_phase_grid_rejects_nonfinite_values(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            PhaseGrid({"pump": (0.0, math.nan)})


if __name__ == "__main__":
    unittest.main()
