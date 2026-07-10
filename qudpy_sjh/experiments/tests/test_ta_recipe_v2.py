from __future__ import annotations

import unittest

import numpy as np

from qudpy_sjh.experiments.pulse_sequence import PulseSpec, SingleRunPlan, SingleRunReadoutResult, SingleRunResult
from qudpy_sjh.experiments.ta import (
    TADelayCenters,
    TAReadoutBundle,
    TASingleDelayPairResult,
    TASingleDelayPlan,
    extract_ta_absorption_bundle,
)
from qudpy_sjh.utils.core import NLevelPhysicalParams
from qudpy_sjh.utils.fields.carrier_envelope import make_constant_carrier_envelope_field


def _template(name: str):
    return make_constant_carrier_envelope_field(
        E0_MV_per_cm=0.01,
        laser_energy_eV=1.55,
        name=name,
    )


def _base_params() -> NLevelPhysicalParams:
    return NLevelPhysicalParams(
        energies_eV=(0.0, 1.55),
        dipole_matrix_D=((0.0, 1.0), (1.0, 0.0)),
        t_start_fs=-2.0,
        t_end_fs=2.0,
        dt_fs=1.0,
        field=_template("base"),
        input_metadata={"keep": "base_metadata"},
    )


def _pulse(name: str) -> PulseSpec:
    return PulseSpec(
        name=name,
        field_template=_template(f"{name}_template"),
        phase_tag=name,
        independent_phase=True,
    )


def _ta_plan() -> TASingleDelayPlan:
    return TASingleDelayPlan(
        base_params=_base_params(),
        pump=_pulse("pump"),
        probe=_pulse("probe"),
        delay=TADelayCenters(delay_fs=100.0, probe_center_fs=0.0),
        case_name="ta_case",
    )


def _fake_result(*, spectrum: dict | None = None, case_name: str = "fake_case") -> SingleRunResult:
    return SingleRunResult(
        case_name=case_name,
        params=None,
        dynamics_result=None,
        field_metadata={},
        readout=SingleRunReadoutResult(
            mode="absorption",
            spectrum={} if spectrum is None else spectrum,
        ),
    )


class TADelayCentersTests(unittest.TestCase):
    def test_delay_centers_convention(self):
        positive = TADelayCenters(delay_fs=100.0, probe_center_fs=0.0)
        negative = TADelayCenters(delay_fs=-50.0, probe_center_fs=10.0)

        self.assertEqual(positive.pump_center_fs, -100.0)
        self.assertEqual(negative.pump_center_fs, 60.0)
        payload = positive.to_dict()
        self.assertEqual(payload["delay_fs"], 100.0)
        self.assertEqual(payload["probe_center_fs"], 0.0)
        self.assertEqual(payload["pump_center_fs"], -100.0)


class TASingleDelayPlanTests(unittest.TestCase):
    def test_plan_construction(self):
        plan = _ta_plan()
        base_field = plan.base_params.field

        pump_probe_sequence = plan.build_pump_probe_sequence()
        probe_only_sequence = plan.build_probe_only_sequence()
        pump_probe = plan.make_pump_probe_plan()
        probe_only = plan.make_probe_only_plan()

        self.assertEqual([pulse.name for pulse in pump_probe_sequence.pulses], ["pump", "probe"])
        self.assertEqual([pulse.name for pulse in probe_only_sequence.pulses], ["probe"])
        self.assertIsInstance(pump_probe, SingleRunPlan)
        self.assertIsInstance(probe_only, SingleRunPlan)
        self.assertEqual(pump_probe.field_plan.centers_fs, {"pump": -100.0, "probe": 0.0})
        self.assertEqual(probe_only.field_plan.centers_fs, {"probe": 0.0})
        self.assertEqual(plan.readout.mode, "absorption")
        self.assertEqual(plan.readout.readout_field_name, "probe")
        self.assertIs(plan.base_params.field, base_field)

    def test_case_names_are_distinct(self):
        plan = _ta_plan()
        pump_probe = plan.make_pump_probe_plan()
        probe_only = plan.make_probe_only_plan()

        self.assertIn("pump_probe", pump_probe.case_name)
        self.assertIn("probe_only", probe_only.case_name)
        self.assertNotEqual(pump_probe.case_name, probe_only.case_name)


class TAReadoutExtractionTests(unittest.TestCase):
    def test_extract_ta_absorption_bundle(self):
        result = _fake_result(
            spectrum={
                "absorption": np.asarray([1.0, 2.0]),
                "energy_eV": np.asarray([1.5, 1.6]),
                "omega_fs_inv": np.asarray([0.1, 0.2]),
            },
            case_name="pump_probe",
        )

        bundle = extract_ta_absorption_bundle(result)

        self.assertIsInstance(bundle, TAReadoutBundle)
        self.assertEqual(bundle.case_name, "pump_probe")
        self.assertEqual(bundle.absorption.shape, (2,))
        self.assertEqual(bundle.energy_eV.shape, (2,))
        self.assertEqual(bundle.omega_fs_inv.shape, (2,))
        self.assertEqual(bundle.to_dict()["energy_range_eV"], (1.5, 1.6))

    def test_extract_ta_absorption_bundle_without_omega(self):
        result = _fake_result(
            spectrum={
                "absorption": np.asarray([1.0, 2.0]),
                "energy_eV": np.asarray([1.5, 1.6]),
            }
        )

        bundle = extract_ta_absorption_bundle(result)

        self.assertIsNone(bundle.omega_fs_inv)
        self.assertFalse(bundle.to_dict()["has_omega_fs_inv"])

    def test_missing_spectrum_keys(self):
        with self.assertRaisesRegex(KeyError, "Available keys"):
            extract_ta_absorption_bundle(
                _fake_result(spectrum={"energy_eV": np.asarray([1.5])})
            )
        with self.assertRaisesRegex(KeyError, "Available keys"):
            extract_ta_absorption_bundle(
                _fake_result(spectrum={"absorption": np.asarray([1.0])})
            )

    def test_shape_mismatch(self):
        with self.assertRaises(ValueError):
            extract_ta_absorption_bundle(
                _fake_result(
                    spectrum={
                        "absorption": np.asarray([1.0, 2.0]),
                        "energy_eV": np.asarray([1.5]),
                    }
                )
            )


class TASingleDelayPairResultTests(unittest.TestCase):
    def test_to_dict_summary_without_arrays(self):
        pump_probe = _fake_result(
            spectrum={
                "absorption": np.asarray([1.0, 2.0]),
                "energy_eV": np.asarray([1.5, 1.6]),
            },
            case_name="pump_probe",
        )
        probe_only = _fake_result(
            spectrum={
                "absorption": np.asarray([0.5, 0.8]),
                "energy_eV": np.asarray([1.5, 1.6]),
            },
            case_name="probe_only",
        )
        pair = TASingleDelayPairResult(
            case_name="ta_case",
            delay_fs=100.0,
            pump_probe=pump_probe,
            probe_only=probe_only,
            pump_probe_bundle=extract_ta_absorption_bundle(pump_probe),
            probe_only_bundle=extract_ta_absorption_bundle(probe_only),
        )

        payload = pair.to_dict(include_arrays=False)

        self.assertEqual(payload["delay_fs"], 100.0)
        self.assertEqual(payload["pump_probe"]["case_name"], "pump_probe")
        self.assertEqual(payload["probe_only"]["case_name"], "probe_only")
        self.assertEqual(payload["pump_probe_bundle"]["n_points"], 2)
        self.assertNotIn("absorption", payload["pump_probe_bundle"])


if __name__ == "__main__":
    unittest.main()
