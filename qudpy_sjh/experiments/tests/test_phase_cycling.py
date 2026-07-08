from __future__ import annotations

import math
import unittest

import numpy as np

from qudpy_sjh.experiments.pulse_sequence import (
    PhaseCyclingPlan,
    PhaseGrid,
    PhaseProjectionSpec,
    PulseSequenceSpec,
    PulseSpec,
    ReadoutSpec,
    SingleRunCheckpointSettings,
    SingleRunFieldPlan,
    SingleRunPlan,
    SingleRunReadoutResult,
    SingleRunResult,
    build_uniform_phase_grid,
    extract_single_run_quantity,
    fourier_project_phase_cases,
    normalize_target_phase_vector,
    phase_projection_weight,
)
from qudpy_sjh.utils.core import NLevelPhysicalParams
from qudpy_sjh.utils.fields.carrier_envelope import make_constant_carrier_envelope_field


def _template(name: str):
    return make_constant_carrier_envelope_field(
        E0_MV_per_cm=0.01,
        laser_energy_eV=1.55,
        name=name,
    )


def _base_plan(*, checkpoint_enabled: bool = False) -> SingleRunPlan:
    sequence = PulseSequenceSpec(
        name="seq",
        pulses=(
            PulseSpec(
                name="pump",
                field_template=_template("pump_template"),
                phase_tag="pump",
                independent_phase=True,
            ),
            PulseSpec(
                name="probe",
                field_template=_template("probe_template"),
                phase_tag="probe",
                independent_phase=True,
            ),
        ),
    )
    field_plan = SingleRunFieldPlan(
        sequence=sequence,
        centers_fs={"pump": -5.0, "probe": 0.0},
        phase_vector={"pump": 0.0, "probe": 0.0},
        case_name="base_case",
    )
    params = NLevelPhysicalParams(
        energies_eV=(0.0, 1.55),
        dipole_matrix_D=((0.0, 1.0), (1.0, 0.0)),
        t_start_fs=-2.0,
        t_end_fs=2.0,
        dt_fs=1.0,
        field=_template("base_field"),
    )
    return SingleRunPlan(
        base_params=params,
        field_plan=field_plan,
        readout=ReadoutSpec(mode="polarization"),
        checkpoint=SingleRunCheckpointSettings(
            enabled=checkpoint_enabled,
            checkpoint_path="/tmp/phase_cycling_test.ckp" if checkpoint_enabled else None,
        ),
    )


def _fake_result(
    *,
    case_name: str = "fake",
    polarization=None,
    spectrum=None,
) -> SingleRunResult:
    readout = SingleRunReadoutResult(
        mode="absorption" if spectrum is not None else "polarization",
        polarization_C_per_m2=np.asarray([1.0, 2.0]) if polarization is None else np.asarray(polarization),
        spectrum=spectrum,
    )
    return SingleRunResult(
        case_name=case_name,
        params=None,
        dynamics_result=None,
        field_metadata={},
        readout=readout,
    )


class PhaseGridTests(unittest.TestCase):
    def test_uniform_phase_grid_single_tag(self):
        grid = build_uniform_phase_grid(["pump"], n_steps=4)

        self.assertEqual(grid.tags, ("pump",))
        self.assertEqual(len(grid), 4)
        self.assertEqual(len(list(grid.iter_phase_vectors())), 4)

    def test_phase_grid_two_tags_length_and_order(self):
        grid = PhaseGrid({"pump": (0.0, math.pi), "probe": (0.0, 1.0, 2.0)})

        self.assertEqual(grid.tags, ("pump", "probe"))
        self.assertEqual(len(grid), 6)
        self.assertEqual(
            list(grid.iter_phase_vectors())[0],
            {"pump": 0.0, "probe": 0.0},
        )

    def test_phase_grid_validation(self):
        with self.assertRaises(ValueError):
            build_uniform_phase_grid([""], n_steps=4)
        with self.assertRaises(ValueError):
            build_uniform_phase_grid(["pump"], n_steps=0)


class PhaseMathTests(unittest.TestCase):
    def test_phase_projection_weight_single_tag(self):
        phi = 0.37
        weight = phase_projection_weight({"probe": phi}, {"probe": 1}, sign=-1)

        self.assertAlmostEqual(weight, np.exp(-1j * phi))
        with self.assertRaises(ValueError):
            phase_projection_weight({}, {"probe": 1})
        with self.assertRaises(ValueError):
            phase_projection_weight({"probe": phi}, {"probe": 1}, sign=0)

    def test_fourier_project_single_tag(self):
        phases = [0.0, 0.5 * math.pi, math.pi, 1.5 * math.pi]
        phase_vectors = [{"probe": phase} for phase in phases]
        signal = np.asarray([np.exp(1j * phase) for phase in phases])

        projected = fourier_project_phase_cases(signal, phase_vectors, {"probe": 1}, sign=-1)
        rejected = fourier_project_phase_cases(signal, phase_vectors, {"probe": 0}, sign=-1)

        self.assertAlmostEqual(projected, 1.0 + 0.0j)
        self.assertAlmostEqual(rejected, 0.0 + 0.0j)

    def test_fourier_project_preserves_non_phase_shape(self):
        phases = [0.0, 0.5 * math.pi, math.pi, 1.5 * math.pi]
        phase_vectors = [{"probe": phase} for phase in phases]
        signal = np.asarray([np.exp(1j * phase) for phase in phases])[:, None, None]
        values = signal * np.ones((1, 3, 5), dtype=np.complex128)

        projected = fourier_project_phase_cases(values, phase_vectors, {"probe": 1}, sign=-1)

        self.assertEqual(projected.shape, (3, 5))
        np.testing.assert_allclose(projected, np.ones((3, 5)), rtol=1e-12, atol=1e-12)

    def test_fourier_project_two_tags(self):
        phases = [0.0, 0.5 * math.pi, math.pi, 1.5 * math.pi]
        phase_vectors = [
            {"pulse1": phase1, "pulse2": phase2}
            for phase1 in phases
            for phase2 in phases
        ]
        signal = np.asarray(
            [np.exp(1j * (-item["pulse1"] + item["pulse2"])) for item in phase_vectors]
        )

        projected = fourier_project_phase_cases(
            signal,
            phase_vectors,
            {"pulse1": -1, "pulse2": 1},
            sign=-1,
        )

        self.assertAlmostEqual(projected, 1.0 + 0.0j)

    def test_normalize_target_phase_vector(self):
        self.assertEqual(
            normalize_target_phase_vector({"pump": 1.0}, known_tags=("pump", "probe")),
            {"pump": 1, "probe": 0},
        )
        with self.assertRaises(ValueError):
            normalize_target_phase_vector({"pump": 1.2})
        with self.assertRaises(ValueError):
            normalize_target_phase_vector({"unknown": 1}, known_tags=("pump",))


class QuantitySelectorTests(unittest.TestCase):
    def test_extract_readout_arrays(self):
        result = _fake_result(
            polarization=np.asarray([1.0, 2.0]),
            spectrum={"absorption": np.asarray([3.0, 4.0])},
        )

        np.testing.assert_allclose(
            extract_single_run_quantity(result, "readout.polarization_C_per_m2"),
            np.asarray([1.0, 2.0]),
        )
        np.testing.assert_allclose(
            extract_single_run_quantity(result, "readout.spectrum.absorption"),
            np.asarray([3.0, 4.0]),
        )

    def test_extract_readout_errors(self):
        result = _fake_result(spectrum={"absorption": np.asarray([1.0])})

        with self.assertRaisesRegex(KeyError, "Available keys"):
            extract_single_run_quantity(result, "readout.spectrum.energy_eV")
        with self.assertRaisesRegex(ValueError, "Unsupported quantity"):
            extract_single_run_quantity(result, "density")


class PhaseCyclingPlanTests(unittest.TestCase):
    def test_make_case_plan_does_not_mutate_base_plan(self):
        base = _base_plan()
        plan = PhaseCyclingPlan(
            base_plan=base,
            phase_grid=build_uniform_phase_grid(["probe"], n_steps=4),
            target_phase_vector={"probe": 1},
            projection=PhaseProjectionSpec(quantity="readout.polarization_C_per_m2"),
        )

        case = plan.make_case_plan({"probe": math.pi}, index=2)

        self.assertEqual(case.field_plan.phase_vector["probe"], math.pi)
        self.assertEqual(case.field_plan.phase_vector["pump"], 0.0)
        self.assertEqual(case.case_name, "base_case_phase_0002")
        self.assertEqual(base.field_plan.phase_vector["probe"], 0.0)
        self.assertIsNot(case.field_plan, base.field_plan)

    def test_execute_with_fake_executor_projects_signal(self):
        base = _base_plan()
        plan = PhaseCyclingPlan(
            base_plan=base,
            phase_grid=build_uniform_phase_grid(["probe"], n_steps=4),
            target_phase_vector={"probe": 1},
            projection=PhaseProjectionSpec(quantity="readout.polarization_C_per_m2"),
        )
        calls: list[str] = []

        def fake_executor(case_plan: SingleRunPlan) -> SingleRunResult:
            calls.append(case_plan.case_name)
            phase = case_plan.field_plan.phase_vector["probe"]
            return _fake_result(
                case_name=case_plan.case_name,
                polarization=np.asarray([np.exp(1j * phase)]),
            )

        result = plan.execute(executor=fake_executor)

        self.assertEqual(len(calls), 4)
        self.assertEqual(len(result.case_records), 4)
        self.assertEqual(result.values.shape, (4, 1))
        np.testing.assert_allclose(result.projected, np.asarray([1.0 + 0.0j]), rtol=1e-12, atol=1e-12)

    def test_execute_rejects_shape_mismatch(self):
        base = _base_plan()
        plan = PhaseCyclingPlan(
            base_plan=base,
            phase_grid=build_uniform_phase_grid(["probe"], n_steps=2),
            target_phase_vector={"probe": 1},
            projection=PhaseProjectionSpec(quantity="readout.polarization_C_per_m2"),
        )

        def fake_executor(case_plan: SingleRunPlan) -> SingleRunResult:
            index = int(case_plan.case_name.rsplit("_", 1)[-1])
            if index == 0:
                return _fake_result(case_name=case_plan.case_name, polarization=np.asarray([1.0]))
            return _fake_result(case_name=case_plan.case_name, polarization=np.asarray([1.0, 2.0]))

        with self.assertRaises(ValueError):
            plan.execute(executor=fake_executor)

    def test_checkpoint_guard(self):
        with self.assertRaises(ValueError):
            PhaseCyclingPlan(
                base_plan=_base_plan(checkpoint_enabled=True),
                phase_grid=build_uniform_phase_grid(["probe"], n_steps=4),
                target_phase_vector={"probe": 1},
                projection=PhaseProjectionSpec(quantity="readout.polarization_C_per_m2"),
            )


if __name__ == "__main__":
    unittest.main()
