from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

import numpy as np
from qutip import Qobj

from qudpy_sjh.experiments import (
    ReadoutPlan,
    compute_polarization_result,
    load_projected_result,
    project_phase_orders,
    save_projected_result,
)
from qudpy_sjh.experiments.pulse_sequence import PhaseGrid, build_uniform_phase_grid
from qudpy_sjh.utils.core import DynamicsResult, NLevelPhysicalParams, ParaNormalizer
from qudpy_sjh.utils.fields.carrier_envelope import make_constant_carrier_envelope_field


def _workspace_path(label: str) -> Path:
    return Path.cwd() / f".tmp_{label}_{uuid4().hex}"


def _remove_file(path: Path) -> None:
    path.unlink(missing_ok=True)


def _field(name: str, *, phase_rad: float = 0.0, amplitude: float = 0.02):
    return make_constant_carrier_envelope_field(
        E0_MV_per_cm=amplitude,
        laser_energy_eV=1.55,
        phase_rad=phase_rad,
        name=name,
    )


def _physical_params() -> NLevelPhysicalParams:
    return NLevelPhysicalParams(
        energies_eV=(0.0, 1.55),
        dipole_matrix_D=((0.0, 1.0), (1.0, 0.0)),
        t_start_fs=0.0,
        t_end_fs=24.0,
        dt_fs=0.25,
        field=_field("interaction"),
    )


def _dynamics() -> DynamicsResult:
    params = _physical_params()
    time_fs = np.arange(0.0, 24.0, 0.25)
    omega = 1.55 * ParaNormalizer.EV_TO_FS_INV
    coherence = 0.06 * np.cos(omega * time_fs + 0.27)
    states = [
        Qobj(np.asarray([[0.5, value], [value, 0.5]], dtype=np.complex128))
        for value in coherence
    ]
    return DynamicsResult(
        mode="lab_exact",
        times=time_fs,
        times_fs=time_fs,
        states=states,
        parameters=None,
        physical_params=params,
    )


class ProjectedResultPersistenceTests(unittest.TestCase):
    def test_real_complex_nan_axes_targets_and_grid_round_trip(self):
        grid = build_uniform_phase_grid(
            ("pump", "probe"),
            n_steps={"pump": 2, "probe": 3},
        )
        pump = np.asarray(grid.phases_by_tag["pump"])
        probe = np.asarray(grid.phases_by_tag["probe"])
        delay = np.asarray([-10.0, 20.0], dtype=np.float32)
        energy = np.asarray([1.4, 1.5, 1.6], dtype=np.float64)
        amplitude_a = np.asarray(
            [[1.0 + 0.5j, 2.0 - 0.25j, 3.0], [-1.0j, 0.5, 1.5 + 0.2j]]
        )
        amplitude_b = np.asarray(
            [[0.5, -0.25, 0.75], [1.0, 1.25, -0.5]],
            dtype=float,
        )
        data = (
            amplitude_a[:, None, None, :]
            * np.exp(-1j * probe)[None, None, :, None]
            + amplitude_b[:, None, None, :]
            * np.exp(-1j * pump)[None, :, None, None]
        )
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
            targets={
                "S_0_1": {"pump": 0, "probe": 1},
                "S_1_0": {"pump": 1, "probe": 0},
            },
        )
        result["projected"]["S_0_1"][0, 1] = np.nan + 0.0j
        result["metadata"]["nonfinite_diagnostic"] = np.asarray(
            [1.0, np.nan, np.inf, -np.inf]
        )

        base = _workspace_path("projected_scan")
        self.addCleanup(_remove_file, Path(f"{base}.npz"))
        self.addCleanup(_remove_file, Path(f"{base}.json"))
        paths = save_projected_result(result, base)
        loaded = load_projected_result(paths["json"])
        manifest = json.loads(paths["json"].read_text(encoding="utf-8"))

        self.assertEqual(loaded["axis_names"], ("T", "energy_eV"))
        self.assertEqual(loaded["axis_values"]["T"].dtype, np.dtype(np.float32))
        np.testing.assert_array_equal(loaded["axis_values"]["T"], delay)
        np.testing.assert_array_equal(loaded["axis_values"]["energy_eV"], energy)
        for name in result["projected"]:
            np.testing.assert_allclose(
                loaded["projected"][name],
                result["projected"][name],
                equal_nan=True,
            )
        self.assertEqual(loaded["targets"], result["targets"])
        self.assertEqual(
            loaded["metadata"]["phase_grid"]["phases_by_tag"],
            result["metadata"]["phase_grid"]["phases_by_tag"],
        )
        self.assertEqual(
            loaded["metadata"]["phase_projection_convention"],
            result["metadata"]["phase_projection_convention"],
        )
        self.assertEqual(manifest["metadata"]["nonfinite_diagnostic"], [1.0, None, None, None])

    def test_explicit_nonuniform_phase_grid_values_round_trip(self):
        phases = (0.0, 0.71, 2.13)
        grid = PhaseGrid({"probe": phases})
        result = project_phase_orders(
            np.asarray([1.0, 2.0, 4.0]),
            axis_names=("phase:probe",),
            axis_values={"phase:probe": np.asarray(phases)},
            phase_grid=grid,
            targets={"S_1": {"probe": 1}},
        )

        base = _workspace_path("nonuniform")
        self.addCleanup(_remove_file, Path(f"{base}.npz"))
        self.addCleanup(_remove_file, Path(f"{base}.json"))
        save_projected_result(result, base)
        loaded = load_projected_result(Path(f"{base}.npz"))

        self.assertEqual(
            loaded["metadata"]["phase_grid"]["phases_by_tag"]["probe"],
            list(phases),
        )
        np.testing.assert_allclose(loaded["projected"]["S_1"], result["projected"]["S_1"])


class DynamicsCheckpointRecomputeTests(unittest.TestCase):
    def test_loaded_dynamics_supports_multiple_readout_plans_without_solver(self):
        dynamics = _dynamics()
        polarization = compute_polarization_result(dynamics, number_density_m3=1.0e24)
        full_plan = ReadoutPlan(
            mode="full",
            readout_field=_field("full_lo", phase_rad=0.0, amplitude=0.03),
            emitted_field_scale=1.0e8,
            zero_padding_factor=1,
        )
        weak_plan = ReadoutPlan(
            mode="weak",
            readout_field=_field("weak_lo", phase_rad=0.4, amplitude=0.025),
            emitted_field_scale=1.0e8,
            zero_padding_factor=1,
        )
        full_before = full_plan.execute(polarization, interaction_field=dynamics.physical_params.field)
        weak_before = weak_plan.execute(polarization, interaction_field=dynamics.physical_params.field)

        checkpoint = _workspace_path("dynamics").with_suffix(".ckp")
        self.addCleanup(_remove_file, checkpoint)
        with patch("qudpy_sjh.experiments.pulse_sequence.single_run.run_case") as solver:
            dynamics.save_ckp(checkpoint)
            loaded = DynamicsResult.from_ckp(checkpoint)
            loaded_polarization = compute_polarization_result(
                loaded,
                number_density_m3=1.0e24,
            )
            full_after = full_plan.execute(
                loaded_polarization,
                interaction_field=loaded.physical_params.field,
            )
            weak_after = weak_plan.execute(
                loaded_polarization,
                interaction_field=loaded.physical_params.field,
            )

        self.assertEqual(solver.call_count, 0)
        assert full_before.spectrum is not None and full_after.spectrum is not None
        assert weak_before.spectrum is not None and weak_after.spectrum is not None
        np.testing.assert_allclose(
            full_after.spectrum["detector_intensity"],
            full_before.spectrum["detector_intensity"],
        )
        np.testing.assert_allclose(
            weak_after.spectrum["detector_intensity"],
            weak_before.spectrum["detector_intensity"],
        )
        self.assertFalse(
            np.allclose(
                full_after.spectrum["detector_intensity"],
                weak_after.spectrum["detector_intensity"],
            )
        )


if __name__ == "__main__":
    unittest.main()
