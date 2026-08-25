from __future__ import annotations

import json
import math
import unittest
from pathlib import Path
from uuid import uuid4

import numpy as np
from qutip import Qobj

from qudpy_sjh.experiments import (
    ReadoutPlan,
    ReadoutResult,
    load_projected_result,
    project_phase_orders,
    save_projected_result,
)
from qudpy_sjh.experiments.pulse_sequence import PhaseGrid, PulseSpec, SingleRunResult
from qudpy_sjh.experiments.ta import (
    TAPrePCRecipe,
    TAReadoutBundle,
    build_ta_pre_pc_observable,
    compute_ta_contrast,
)
from qudpy_sjh.utils.core import DynamicsResult, NLevelPhysicalParams
from qudpy_sjh.utils.fields.carrier_envelope import make_constant_carrier_envelope_field


def _field(name: str, *, amplitude: float) -> object:
    return make_constant_carrier_envelope_field(
        E0_MV_per_cm=amplitude,
        laser_energy_eV=1.55,
        name=name,
    )


def _pulse(name: str, *, amplitude: float) -> PulseSpec:
    return PulseSpec(
        name=name,
        field_template=_field(f"{name}_template", amplitude=amplitude),
        phase_tag=name,
        independent_phase=True,
    )


def _base_params() -> NLevelPhysicalParams:
    return NLevelPhysicalParams(
        energies_eV=(0.0, 1.55),
        dipole_matrix_D=((0.0, 1.0), (1.0, 0.0)),
        t_start_fs=0.0,
        t_end_fs=24.0,
        dt_fs=0.25,
        field=_field("base", amplitude=0.01),
    )


def _recipe(
    *,
    readout_mode: str = "full",
    observable: str = "delta_T_over_T",
    delays_fs: tuple[float, ...] = (-10.0, 20.0),
    phase_grid: PhaseGrid | None = None,
    denominator_rel_threshold: float = 0.0,
    denominator_abs_threshold: float = 0.0,
) -> TAPrePCRecipe:
    grid = phase_grid or PhaseGrid(
        {
            "pump": (0.0, math.pi),
            "probe": (0.0, 2.0 * math.pi / 3.0, 4.0 * math.pi / 3.0),
        }
    )
    return TAPrePCRecipe(
        base_params=_base_params(),
        pump=_pulse("pump", amplitude=0.02),
        probe=_pulse("probe", amplitude=0.01),
        delays_fs=delays_fs,
        phase_grid=grid,
        readout_plan=ReadoutPlan(
            mode=readout_mode,
            readout_field="probe",
            zero_padding_factor=1,
            emitted_field_scale=1.0e8,
        ),
        observable=observable,
        denominator_rel_threshold=denominator_rel_threshold,
        denominator_abs_threshold=denominator_abs_threshold,
        target_phase_vector={"pump": 0, "probe": 1},
        case_name="recipe_first_test",
    )


def _fake_dynamics(plan) -> SingleRunResult:
    params = plan.make_params()
    recipe_metadata = plan.input_metadata["ta_recipe_first"]
    condition = recipe_metadata["condition"]
    phases = recipe_metadata["phase_vector"]
    coordinates = recipe_metadata["recipe_coordinates"]
    delay = float(coordinates.get("T_fs", 0.0))
    pump_phase = float(phases.get("pump", 0.0))
    probe_phase = float(phases.get("probe", 0.0))
    time_fs = np.arange(0.0, 24.0, 0.25)
    carrier = 1.55 * 1.5192674479961278
    amplitude = 0.035 + 0.004 * np.cos(probe_phase)
    if condition == "pump_on":
        amplitude += 0.006 * np.cos(pump_phase) + 0.0001 * delay
    coherence = amplitude * np.cos(carrier * time_fs + 0.23 + 0.1 * np.sin(probe_phase))
    states = [
        Qobj(np.asarray([[0.5, value], [value, 0.5]], dtype=np.complex128))
        for value in coherence
    ]
    dynamics = DynamicsResult(
        mode="lab_exact",
        times=time_fs,
        times_fs=time_fs,
        states=states,
        parameters=None,
        physical_params=params,
    )
    return SingleRunResult(
        case_name=plan.case_name,
        params=params,
        dynamics_result=dynamics,
        field_metadata=params.field.to_dict(),
        metadata={"synthetic_solver": True},
    )


def _synthetic_readout(values, *, mode: str, key: str) -> ReadoutResult:
    values_array = np.asarray(values, dtype=float)
    energy = np.linspace(1.4, 1.6, values_array.size)
    return ReadoutResult(
        mode=mode,
        spectrum={
            "energy_eV": energy,
            "omega_fs_inv": energy / 0.6582119569509067,
            key: values_array,
        },
    )


class TARecipeCasePlanningTests(unittest.TestCase):
    def test_pump_off_runs_only_over_true_probe_phase_dependency(self):
        recipe = _recipe()
        plans = recipe.build_dynamics_plans()

        self.assertEqual(len(plans["pump_on"]), 2 * 2 * 3)
        self.assertEqual(len(plans["pump_off"]), 3)
        self.assertEqual(set(plans["pump_off"]), {(0,), (1,), (2,)})
        self.assertEqual(
            [
                plans["pump_off"][(index,)].field_plan.phase_vector["probe"]
                for index in range(3)
            ],
            list(recipe.phase_grid.phases_by_tag["probe"]),
        )
        self.assertTrue(
            all(
                plan.field_plan.sequence is next(iter(plans["pump_on"].values())).field_plan.sequence
                for plan in plans["pump_on"].values()
            )
        )
        self.assertTrue(
            all(
                plan.field_plan.sequence is next(iter(plans["pump_off"].values())).field_plan.sequence
                for plan in plans["pump_off"].values()
            )
        )
        self.assertIsNot(
            next(iter(plans["pump_on"].values())).field_plan.sequence,
            next(iter(plans["pump_off"].values())).field_plan.sequence,
        )
        positive_delay = plans["pump_on"][(1, 0, 0)].field_plan.centers_fs
        self.assertEqual(positive_delay, {"pump": -20.0, "probe": 0.0})

    def test_executor_call_count_matches_unique_dynamics_cases(self):
        recipe = _recipe()
        calls = []

        def executor(plan):
            calls.append(plan.case_name)
            return _fake_dynamics(plan)

        dynamics = recipe.execute_dynamics(executor=executor)

        self.assertEqual(len(calls), 15)
        self.assertEqual(
            dynamics["solver_case_counts"],
            {"pump_on": 12, "pump_off": 3, "total": 15},
        )


class TARecipePostprocessTests(unittest.TestCase):
    def test_detector_algebra_broadcast_and_named_axes(self):
        recipe = _recipe()
        pump_off = {
            (probe_index,): _synthetic_readout(
                [2.0 + probe_index, 4.0 + probe_index],
                mode="full",
                key="detector_intensity",
            )
            for probe_index in range(3)
        }
        pump_on = {}
        for delay_index in range(2):
            for pump_index in range(2):
                for probe_index in range(3):
                    reference = np.asarray([2.0 + probe_index, 4.0 + probe_index])
                    factor = 0.1 * (1 + delay_index + pump_index + probe_index)
                    pump_on[(delay_index, pump_index, probe_index)] = _synthetic_readout(
                        reference * (1.0 + factor),
                        mode="full",
                        key="detector_intensity",
                    )

        result = build_ta_pre_pc_observable(
            recipe,
            {"pump_on": pump_on, "pump_off": pump_off, "readout_plan": recipe.readout_plan},
        )

        self.assertEqual(result.quantity, "delta_T_over_T")
        self.assertEqual(result.difference_quantity, "delta_I")
        self.assertEqual(result.axis_names, ("T", "phase:pump", "phase:probe", "energy_eV"))
        self.assertEqual(result.data.shape, (2, 2, 3, 2))
        self.assertEqual(result.axis_values["T"].tolist(), [-10.0, 20.0])
        np.testing.assert_allclose(result.data[1, 1, 2], 0.5)
        np.testing.assert_allclose(result.difference[1, 1, 2], [2.0, 3.0])
        self.assertTrue(np.all(result.valid_reference_mask))
        self.assertEqual(result.metadata["phase_projection_status"], "not_applied_pre_pc_observable")
        self.assertEqual(result.metadata["target_phase_vector"], {"pump": 0, "probe": 1})
        json.dumps(result.to_dict(include_arrays=True))

    def test_zero_and_near_zero_denominators_are_nan_and_reported(self):
        recipe = _recipe(
            delays_fs=(0.0,),
            phase_grid=PhaseGrid({"pump": (0.0,), "probe": (0.0,)}),
            denominator_rel_threshold=1.0e-6,
        )
        result = recipe.postprocess(
            {
                "pump_on": {
                    (0, 0, 0): _synthetic_readout(
                        [1.0, 1.0, 4.0], mode="full", key="detector_intensity"
                    )
                },
                "pump_off": {
                    (0,): _synthetic_readout(
                        [0.0, 1.0e-9, 2.0], mode="full", key="detector_intensity"
                    )
                },
                "readout_plan": recipe.readout_plan,
            }
        )

        self.assertTrue(np.isnan(result.data[0, 0, 0, 0]))
        self.assertTrue(np.isnan(result.data[0, 0, 0, 1]))
        self.assertEqual(result.data[0, 0, 0, 2], 1.0)
        self.assertEqual(result.metadata["denominator_policy"]["invalid_count"], 2)
        self.assertIsNotNone(result.metadata["denominator_policy"]["warning"])

    def test_absorption_compatibility_matches_legacy_subtraction(self):
        recipe = _recipe(
            readout_mode="absorption_like",
            observable="delta_absorption_like",
            delays_fs=(7.0,),
            phase_grid=PhaseGrid({"pump": (0.0,), "probe": (0.0,)}),
        )
        energy = np.asarray([1.4, 1.5, 1.6])
        on = np.asarray([0.5, -0.25, 0.75])
        off = np.asarray([0.1, -0.1, 0.25])
        result = recipe.postprocess(
            {
                "pump_on": {
                    (0, 0, 0): _synthetic_readout(
                        on, mode="absorption_like", key="absorption_like_response"
                    )
                },
                "pump_off": {
                    (0,): _synthetic_readout(
                        off, mode="absorption_like", key="absorption_like_response"
                    )
                },
                "readout_plan": recipe.readout_plan,
            }
        )
        legacy = compute_ta_contrast(
            TAReadoutBundle(case_name="on", absorption=on, energy_eV=energy),
            TAReadoutBundle(case_name="off", absorption=off, energy_eV=energy),
            delay_fs=7.0,
        )

        np.testing.assert_allclose(result.data[0, 0, 0], legacy.delta_absorption)
        self.assertEqual(result.quantity, "delta_absorption_like")
        self.assertIn("not detector-level", result.metadata["condition_formula"])

    def test_pre_pc_observable_feeds_generic_phase_projector(self):
        grid = PhaseGrid(
            {
                "pump": tuple(2.0 * math.pi * index / 4 for index in range(4)),
                "probe": tuple(2.0 * math.pi * index / 3 for index in range(3)),
            }
        )
        recipe = _recipe(
            delays_fs=(0.0,),
            phase_grid=grid,
            denominator_abs_threshold=3.0,
        )
        off = np.asarray([2.0, 4.0])
        pump_off = {
            (probe_index,): _synthetic_readout(
                off, mode="full", key="detector_intensity"
            )
            for probe_index in range(3)
        }
        pump_on = {}
        for pump_index in range(4):
            for probe_index, probe_phase in enumerate(grid.phases_by_tag["probe"]):
                signal = math.cos(probe_phase)
                pump_on[(0, pump_index, probe_index)] = _synthetic_readout(
                    off * (1.0 + signal),
                    mode="full",
                    key="detector_intensity",
                )
        pre_pc = recipe.postprocess(
            {
                "pump_on": pump_on,
                "pump_off": pump_off,
                "readout_plan": recipe.readout_plan,
            }
        )

        projected = project_phase_orders(
            pre_pc.data,
            axis_names=pre_pc.axis_names,
            axis_values=pre_pc.axis_values,
            phase_grid=recipe.phase_grid,
            targets={"S_0_1": {"pump": 0, "probe": 1}},
        )
        projected["metadata"]["ta_denominator_policy"] = pre_pc.metadata[
            "denominator_policy"
        ]
        projected["metadata"]["ta_valid_reference_mask"] = pre_pc.valid_reference_mask
        base = Path.cwd() / f".tmp_ta_projected_{uuid4().hex}"
        npz_path = Path(f"{base}.npz")
        json_path = Path(f"{base}.json")
        self.addCleanup(npz_path.unlink, missing_ok=True)
        self.addCleanup(json_path.unlink, missing_ok=True)
        save_projected_result(projected, base)
        loaded = load_projected_result(base)

        self.assertEqual(projected["axis_names"], ("T", "energy_eV"))
        self.assertEqual(projected["projected"]["S_0_1"].shape, (1, 2))
        np.testing.assert_allclose(
            projected["projected"]["S_0_1"],
            np.asarray([[np.nan, 0.5]]),
            atol=1.0e-12,
            equal_nan=True,
        )
        self.assertEqual(loaded["axis_names"], ("T", "energy_eV"))
        np.testing.assert_allclose(loaded["axis_values"]["T"], [0.0])
        np.testing.assert_allclose(loaded["axis_values"]["energy_eV"], [1.4, 1.6])
        self.assertEqual(loaded["targets"], {"S_0_1": {"pump": 0, "probe": 1}})
        np.testing.assert_allclose(
            loaded["projected"]["S_0_1"],
            projected["projected"]["S_0_1"],
            equal_nan=True,
        )
        loaded_mask = np.asarray(
            loaded["metadata"]["ta_valid_reference_mask"],
            dtype=bool,
        )
        np.testing.assert_array_equal(loaded_mask, pre_pc.valid_reference_mask)
        self.assertFalse(loaded_mask[..., 0].any())


class TARecipeReadoutIntegrationTests(unittest.TestCase):
    def test_full_and_weak_readout_reuse_the_same_dynamics(self):
        recipe = _recipe()
        calls = []

        def executor(plan):
            calls.append(plan.case_name)
            return _fake_dynamics(plan)

        dynamics = recipe.execute_dynamics(executor=executor)
        calls_after_dynamics = len(calls)
        full = recipe.postprocess(recipe.apply_readout(dynamics))
        weak_plan = ReadoutPlan(
            mode="weak",
            readout_field="probe",
            zero_padding_factor=1,
            emitted_field_scale=1.0e8,
        )
        weak = recipe.postprocess(recipe.apply_readout(dynamics, readout_plan=weak_plan))

        self.assertEqual(calls_after_dynamics, 15)
        self.assertEqual(len(calls), calls_after_dynamics)
        self.assertEqual(full.metadata["readout_mode"], "full")
        self.assertEqual(weak.metadata["readout_mode"], "weak")
        self.assertEqual(full.data.shape, weak.data.shape)
        self.assertGreater(np.count_nonzero(np.isfinite(full.data)), 0)
        self.assertGreater(np.count_nonzero(np.isfinite(weak.data)), 0)
        self.assertFalse(np.allclose(full.data, weak.data, equal_nan=True))
        self.assertEqual(full.metadata["solver_case_counts"]["total"], 15)
        self.assertEqual(weak.metadata["solver_case_counts"]["total"], 15)


if __name__ == "__main__":
    unittest.main()
