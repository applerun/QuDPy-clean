from __future__ import annotations

import unittest

import numpy as np

from qudpy_sjh.systems import (
    NLevelSystem,
    make_single_exciton_ladder_system,
    make_three_level_ladder_system,
    make_two_level_system,
)


class NLevelSystemValidationTests(unittest.TestCase):
    def _system(self) -> NLevelSystem:
        return NLevelSystem(
            name="test",
            basis=("0", "X"),
            energies_eV=np.asarray([0.0, 1.5]),
            dipole_matrix_D=np.asarray([[0.0, 2.0], [2.0, 0.0]]),
        )

    def test_basis_must_be_non_empty(self) -> None:
        with self.assertRaisesRegex(ValueError, "basis"):
            NLevelSystem(name="bad", basis=(), energies_eV=np.asarray([]), dipole_matrix_D=np.zeros((0, 0)))

    def test_basis_must_be_unique(self) -> None:
        with self.assertRaisesRegex(ValueError, "unique"):
            NLevelSystem(
                name="bad",
                basis=("0", "0"),
                energies_eV=np.asarray([0.0, 1.0]),
                dipole_matrix_D=np.eye(2),
            )

    def test_energies_length_matches_dimension(self) -> None:
        with self.assertRaisesRegex(ValueError, "length"):
            NLevelSystem(
                name="bad",
                basis=("0", "X"),
                energies_eV=np.asarray([0.0]),
                dipole_matrix_D=np.eye(2),
            )

    def test_dipole_shape_matches_dimension(self) -> None:
        with self.assertRaisesRegex(ValueError, "shape"):
            NLevelSystem(
                name="bad",
                basis=("0", "X"),
                energies_eV=np.asarray([0.0, 1.0]),
                dipole_matrix_D=np.zeros((2, 3)),
            )

    def test_transition_dephasing_labels_must_exist(self) -> None:
        with self.assertRaisesRegex(ValueError, "basis"):
            NLevelSystem(
                name="bad",
                basis=("0", "X"),
                energies_eV=np.asarray([0.0, 1.0]),
                dipole_matrix_D=np.eye(2),
                transition_dephasing_fs_inv={("0", "missing"): 0.01},
            )

    def test_transition_dephasing_rate_must_not_be_negative(self) -> None:
        with self.assertRaisesRegex(ValueError, ">= 0"):
            NLevelSystem(
                name="bad",
                basis=("0", "X"),
                energies_eV=np.asarray([0.0, 1.0]),
                dipole_matrix_D=np.eye(2),
                transition_dephasing_fs_inv={("0", "X"): -0.01},
            )

    def test_dimension_and_summary_dict(self) -> None:
        system = self._system()
        self.assertEqual(system.dimension, 2)
        payload = system.to_dict(include_arrays=False)
        self.assertEqual(payload["dimension"], 2)
        self.assertIn("energy_range_eV", payload)
        self.assertNotIn("energies_eV", payload)
        self.assertNotIn("dipole_matrix_D", payload)


class NLevelSystemDissipationTests(unittest.TestCase):
    def test_with_and_append_dissipation_return_new_objects(self) -> None:
        system = make_two_level_system(energy_eV=1.5, mu_D=2.0)
        replaced = system.with_dissipation(("relax_1",))
        appended = replaced.append_dissipation("pure_1", "pure_2")

        self.assertEqual(system.dissipation, ())
        self.assertEqual(replaced.dissipation, ("relax_1",))
        self.assertEqual(appended.dissipation, ("relax_1", "pure_1", "pure_2"))
        self.assertIsNot(system, replaced)
        self.assertIsNot(replaced, appended)


class BasicMakerTests(unittest.TestCase):
    def test_make_two_level_system(self) -> None:
        system = make_two_level_system(energy_eV=1.55, mu_D=5.0, gamma_fs_inv=0.01)

        self.assertEqual(system.basis, ("0", "X"))
        np.testing.assert_allclose(system.energies_eV, [0.0, 1.55])
        self.assertEqual(system.dipole_matrix_D[0, 1], 5.0)
        self.assertEqual(system.dipole_matrix_D[1, 0], 5.0)
        self.assertEqual(system.initial_state.shape, (2, 2))
        np.testing.assert_allclose(system.initial_state, [[1.0, 0.0], [0.0, 0.0]])
        self.assertEqual(system.transition_dephasing_fs_inv[("0", "X")], 0.01)
        self.assertEqual(system.dissipation, ())

    def test_make_three_level_ladder_system_defaults(self) -> None:
        system = make_three_level_ladder_system(
            energy_01_eV=1.55,
            mu01_D=5.0,
            gamma01_fs_inv=0.01,
            gamma12_fs_inv=0.02,
        )

        self.assertEqual(system.basis, ("0", "X", "XX"))
        np.testing.assert_allclose(system.energies_eV, [0.0, 1.55, 3.10])
        self.assertEqual(system.dipole_matrix_D[0, 1], 5.0)
        self.assertEqual(system.dipole_matrix_D[1, 0], 5.0)
        self.assertAlmostEqual(system.dipole_matrix_D[1, 2], np.sqrt(2.0) * 5.0)
        self.assertAlmostEqual(system.dipole_matrix_D[2, 1], np.sqrt(2.0) * 5.0)
        self.assertEqual(system.dipole_matrix_D[0, 2], 0.0)
        self.assertEqual(system.dipole_matrix_D[2, 0], 0.0)
        self.assertEqual(system.transition_dephasing_fs_inv[("0", "X")], 0.01)
        self.assertEqual(system.transition_dephasing_fs_inv[("X", "XX")], 0.02)


class SingleExcitonLadderTests(unittest.TestCase):
    def test_default_n2(self) -> None:
        system = make_single_exciton_ladder_system(
            n_quantum=2,
            energy_1q_eV=1.6,
            mu_1q_D=4.0,
            gamma_1q_fs_inv=0.01,
        )

        self.assertEqual(system.basis, ("0", "X", "XX"))
        np.testing.assert_allclose(system.energies_eV, [0.0, 1.6, 3.2])
        self.assertEqual(system.dipole_matrix_D[0, 1], 4.0)
        self.assertAlmostEqual(system.dipole_matrix_D[1, 2], np.sqrt(2.0) * 4.0)
        self.assertEqual(system.transition_dephasing_fs_inv[("0", "X")], 0.01)
        self.assertEqual(system.transition_dephasing_fs_inv[("X", "XX")], 0.01)
        self.assertEqual(system.metadata["normalized_eis_eV"], [0.0])
        self.assertEqual(system.metadata["normalized_pb"], [1.0])
        self.assertEqual(system.metadata["normalized_eid"], [1.0])

    def test_n3_with_corrections(self) -> None:
        system = make_single_exciton_ladder_system(
            n_quantum=3,
            energy_1q_eV=1.6,
            mu_1q_D=4.0,
            gamma_1q_fs_inv=0.01,
            eis_eV=[-0.02, 0.01],
            pb=[0.8, 0.6],
            eid=[1.5, 2.0],
        )

        transition_energies = [1.6, 1.58, 1.61]
        np.testing.assert_allclose(system.energies_eV, [0.0, 1.6, 3.18, 4.79])
        self.assertEqual(system.basis, ("0", "X", "XX", "XXX"))
        self.assertEqual(system.dipole_matrix_D[0, 1], 4.0)
        self.assertAlmostEqual(system.dipole_matrix_D[1, 2], 0.8 * np.sqrt(2.0) * 4.0)
        self.assertAlmostEqual(system.dipole_matrix_D[2, 3], 0.6 * np.sqrt(3.0) * 4.0)
        self.assertEqual(system.transition_dephasing_fs_inv[("0", "X")], 0.01)
        self.assertEqual(system.transition_dephasing_fs_inv[("X", "XX")], 0.015)
        self.assertEqual(system.transition_dephasing_fs_inv[("XX", "XXX")], 0.02)
        self.assertEqual(
            [item["energy_eV"] for item in system.metadata["transition_energies_eV"]],
            transition_energies,
        )

    def test_n1_accepts_default_higher_order_values(self) -> None:
        system = make_single_exciton_ladder_system(
            n_quantum=1,
            energy_1q_eV=1.6,
            mu_1q_D=4.0,
            gamma_1q_fs_inv=0.01,
        )

        self.assertEqual(system.dimension, 2)
        self.assertEqual(system.basis, ("0", "X"))
        np.testing.assert_allclose(system.energies_eV, [0.0, 1.6])

    def test_n1_rejects_non_default_higher_order_values(self) -> None:
        kwargs = {
            "n_quantum": 1,
            "energy_1q_eV": 1.6,
            "mu_1q_D": 4.0,
        }
        with self.assertRaisesRegex(ValueError, "eis_eV"):
            make_single_exciton_ladder_system(**kwargs, eis_eV=0.1)
        with self.assertRaisesRegex(ValueError, "pb"):
            make_single_exciton_ladder_system(**kwargs, pb=0.9)
        with self.assertRaisesRegex(ValueError, "eid"):
            make_single_exciton_ladder_system(**kwargs, eid=1.2)

    def test_length_rules(self) -> None:
        make_single_exciton_ladder_system(
            n_quantum=2,
            energy_1q_eV=1.6,
            mu_1q_D=4.0,
            eis_eV=0.01,
            pb=0.9,
            eid=1.2,
        )
        make_single_exciton_ladder_system(
            n_quantum=2,
            energy_1q_eV=1.6,
            mu_1q_D=4.0,
            eis_eV=[0.01],
            pb=[0.9],
            eid=[1.2],
        )
        with self.assertRaisesRegex(ValueError, "eis_eV.*scalar"):
            make_single_exciton_ladder_system(n_quantum=3, energy_1q_eV=1.6, mu_1q_D=4.0, eis_eV=0.01)
        with self.assertRaisesRegex(ValueError, "pb.*length"):
            make_single_exciton_ladder_system(
                n_quantum=3,
                energy_1q_eV=1.6,
                mu_1q_D=4.0,
                eis_eV=[0.0, 0.0],
                pb=[1.0],
            )

    def test_no_gamma_generates_no_transition_dephasing_but_keeps_eid_metadata(self) -> None:
        system = make_single_exciton_ladder_system(
            n_quantum=2,
            energy_1q_eV=1.6,
            mu_1q_D=4.0,
            gamma_1q_fs_inv=None,
            eid=1.5,
        )

        self.assertEqual(system.transition_dephasing_fs_inv, {})
        self.assertEqual(system.metadata["normalized_eid"], [1.5])
        self.assertTrue(system.metadata["no_population_relaxation_generated"])
        self.assertEqual(system.dissipation, ())


if __name__ == "__main__":
    unittest.main()
