from __future__ import annotations

from pathlib import Path
import unittest
from uuid import uuid4

import numpy as np

from qudpy_sjh.systems import NLevelSystem, make_base_physical_params_from_system, make_two_level_system
from qudpy_sjh.utils.checks import n2_mainline_equivalence_check
from qudpy_sjh.utils.core import ParaNormalizer, PureDephasingChannel, RelaxationChannel, run_case
from qudpy_sjh.utils.core.model import build_c_ops, build_static_hamiltonian
from qudpy_sjh.utils.fields.carrier_envelope import make_constant_carrier_envelope_field


GROUND = np.asarray([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128)
EXCITED = np.asarray([[0.0, 0.0], [0.0, 1.0]], dtype=np.complex128)


def _field():
    return make_constant_carrier_envelope_field(
        E0_MV_per_cm=0.01,
        laser_energy_eV=1.5,
        envelope_amplitude=0.0,
        name="zero_envelope_field",
    )


def _params(system: NLevelSystem):
    return make_base_physical_params_from_system(
        system,
        field=_field(),
        t_start_fs=0.0,
        t_end_fs=1.0,
        dt_fs=0.5,
    )


def _run(system: NLevelSystem):
    return run_case(
        _params(system),
        normalizer=ParaNormalizer(time_scale_fs=1.0, auto_scale=False),
    )


class InitialStateAdapterTests(unittest.TestCase):
    def test_explicit_ground_state_reaches_dynamics_t0(self) -> None:
        result = _run(make_two_level_system(energy_eV=1.5, mu_D=1.0, initial_state="ground"))

        np.testing.assert_allclose(result.states[0].full(), GROUND)
        np.testing.assert_allclose(result.parameters.rho0, GROUND)

    def test_excited_state_vector_reaches_dynamics_t0(self) -> None:
        system = make_two_level_system(
            energy_eV=1.5,
            mu_D=1.0,
            initial_state=np.asarray([0.0, 1.0]),
        )
        params = _params(system)
        result = run_case(
            params,
            normalizer=ParaNormalizer(time_scale_fs=1.0, auto_scale=False),
        )

        np.testing.assert_allclose(params.initial_density_matrix, EXCITED)
        np.testing.assert_allclose(result.states[0].full(), EXCITED)
        self.assertFalse(np.allclose(result.states[0].full(), GROUND))
        self.assertLess(
            n2_mainline_equivalence_check(
                params,
                normalizer=ParaNormalizer(time_scale_fs=1.0, auto_scale=False),
            )["overall_max_difference"],
            1.0e-12,
        )

    def test_default_initial_state_matches_explicit_ground(self) -> None:
        default = _run(make_two_level_system(energy_eV=1.5, mu_D=1.0, initial_state=None))
        explicit = _run(make_two_level_system(energy_eV=1.5, mu_D=1.0, initial_state="ground"))

        np.testing.assert_allclose(default.parameters.rho0, explicit.parameters.rho0)
        np.testing.assert_allclose(default.states[0].full(), explicit.states[0].full())

    def test_invalid_initial_states_fail_without_ground_fallback(self) -> None:
        common = {
            "name": "invalid_initial",
            "basis": ("g", "e"),
            "energies_eV": np.asarray([0.0, 1.5]),
            "dipole_matrix_D": np.asarray([[0.0, 1.0], [1.0, 0.0]]),
        }
        invalid = (
            (np.asarray([0.0, 2.0]), "unit norm"),
            (np.asarray([[1.0, 1.0], [0.0, 0.0]]), "Hermitian"),
            (np.asarray([[0.5, 0.0], [0.0, 0.0]]), "trace 1"),
            (np.asarray([1.0, 0.0, 0.0]), "shape"),
        )
        for state, message in invalid:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                NLevelSystem(**common, initial_state=state)
        with self.assertRaisesRegex(ValueError, "must be 'ground'"):
            make_two_level_system(energy_eV=1.5, mu_D=1.0, initial_state="excited")


class SystemAdapterPersistenceTests(unittest.TestCase):
    def test_non_ground_json_round_trip_reaches_same_rho0_and_c_ops(self) -> None:
        system = NLevelSystem(
            name="persistent_excited",
            basis=("ground", "excited"),
            energies_eV=np.asarray([0.0, 1.5]),
            dipole_matrix_D=np.asarray([[0.0, 2.0], [2.0, 0.0]]),
            initial_state=EXCITED,
            transition_dephasing_fs_inv={("ground", "excited"): 0.03},
            dissipation=(
                RelaxationChannel(name="decay", from_level=1, to_level=0, rate_fs_inv=0.1),
                PureDephasingChannel(name="phase", level=1, rate_fs_inv=0.2),
            ),
        )
        path = Path.cwd() / f".tmp_m8_system_{uuid4().hex}.json"
        self.addCleanup(path.unlink, missing_ok=True)

        system.save_json(path)
        loaded = NLevelSystem.load_json(path)
        result = _run(loaded)

        self.assertEqual(loaded.basis, system.basis)
        np.testing.assert_allclose(loaded.dipole_matrix_D, system.dipole_matrix_D)
        np.testing.assert_allclose(loaded.initial_state, system.initial_state)
        self.assertEqual(loaded.transition_dephasing_fs_inv, system.transition_dephasing_fs_inv)
        self.assertEqual(loaded.dissipation, system.dissipation)
        np.testing.assert_allclose(result.states[0].full(), EXCITED)
        self.assertEqual(len(build_c_ops(result.parameters)), 2)


class HamiltonianAndCollapseMappingTests(unittest.TestCase):
    def test_manual_and_maker_systems_produce_equivalent_h_rho0_and_c_ops(self) -> None:
        channels = (
            RelaxationChannel(name="decay", from_level=1, to_level=0, rate_fs_inv=0.1),
            PureDephasingChannel(name="phase", level=1, rate_fs_inv=0.2),
        )
        manual = NLevelSystem(
            name="manual",
            basis=("g", "e"),
            energies_eV=np.asarray([0.0, 1.5]),
            dipole_matrix_D=np.asarray([[0.0, 2.0], [2.0, 0.0]]),
            initial_state=None,
            dissipation=channels,
        )
        made = make_two_level_system(
            energy_eV=1.5,
            mu_D=2.0,
            ground_label="g",
            excited_label="e",
            initial_state=None,
        ).with_dissipation(channels)

        manual_result = _run(manual)
        made_result = _run(made)

        np.testing.assert_allclose(
            build_static_hamiltonian(manual_result.parameters).full(),
            build_static_hamiltonian(made_result.parameters).full(),
        )
        np.testing.assert_allclose(manual_result.parameters.rho0, made_result.parameters.rho0)
        self.assertEqual(manual_result.parameters.basis, made_result.parameters.basis)
        manual_c_ops = build_c_ops(manual_result.parameters)
        made_c_ops = build_c_ops(made_result.parameters)
        self.assertEqual(len(manual_c_ops), 2)
        self.assertEqual(len(made_c_ops), 2)
        for left, right in zip(manual_c_ops, made_c_ops):
            np.testing.assert_allclose(left.full(), right.full())

    def test_transition_dephasing_remains_explicitly_metadata_only(self) -> None:
        system = make_two_level_system(
            energy_eV=1.5,
            mu_D=1.0,
            gamma_fs_inv=0.03,
        )
        params = _params(system)

        self.assertEqual(params.pure_dephasing_channels, ())
        self.assertIn(
            "no unique mapping",
            params.input_metadata["system_adapter"]["transition_dephasing_policy"],
        )


if __name__ == "__main__":
    unittest.main()
