from __future__ import annotations

import math
import unittest
from dataclasses import replace
from unittest.mock import patch

import numpy as np
from qutip import Qobj

from qudpy_sjh.experiments import (
    PolarizationResult,
    ReadoutPlan,
    coherent_detector_terms,
    compute_polarization_result,
)
from qudpy_sjh.experiments.pulse_sequence import (
    PulseSequenceSpec,
    PulseSpec,
    SingleRunFieldPlan,
    SingleRunPlan,
)
from qudpy_sjh.utils.core import DynamicsResult, NLevelPhysicalParams, ParaNormalizer
from qudpy_sjh.utils.fields import FieldPhySeries
from qudpy_sjh.utils.fields.carrier_envelope import (
    CarrierEnvelopeField,
    make_constant_carrier_envelope_field,
)


def _field(name: str, *, phase_rad: float = 0.0, amplitude: float = 0.02) -> CarrierEnvelopeField:
    return make_constant_carrier_envelope_field(
        E0_MV_per_cm=amplitude,
        laser_energy_eV=1.55,
        phase_rad=phase_rad,
        name=name,
    )


def _field_plan() -> SingleRunFieldPlan:
    sequence = PulseSequenceSpec(
        name="pump_probe",
        pulses=(
            PulseSpec(name="pump", field_template=_field("pump_template")),
            PulseSpec(name="probe", field_template=_field("probe_template", amplitude=0.01)),
        ),
    )
    return SingleRunFieldPlan(
        sequence=sequence,
        centers_fs={"pump": 0.0, "probe": 0.0},
        phase_vector={},
        case_name="readout_case",
    )


def _base_params():
    return NLevelPhysicalParams(
        energies_eV=(0.0, 1.55),
        dipole_matrix_D=((0.0, 1.0), (1.0, 0.0)),
        t_start_fs=0.0,
        t_end_fs=24.0,
        dt_fs=0.25,
        field=_field("base"),
    )


def _dynamics(physical_params: NLevelPhysicalParams) -> DynamicsResult:
    time_fs = np.arange(0.0, 24.0, 0.25)
    omega = 1.55 * ParaNormalizer.EV_TO_FS_INV
    coherences = 0.08 * np.cos(omega * time_fs + 0.31)
    states = [
        Qobj(
            np.asarray(
                [[0.5, coherence], [coherence, 0.5]],
                dtype=np.complex128,
            )
        )
        for coherence in coherences
    ]
    return DynamicsResult(
        mode="lab_exact",
        times=time_fs,
        times_fs=time_fs,
        states=states,
        parameters=None,
        physical_params=physical_params,
    )


class CoherentDetectorAlgebraTests(unittest.TestCase):
    def test_full_detector_matches_compact_and_expanded_forms(self):
        readout = np.asarray([1.0 + 0.5j, -0.2 + 0.7j, 0.3 - 0.4j])
        signal = np.asarray([0.1 - 0.2j, 0.05 + 0.03j, -0.4 + 0.1j])

        terms = coherent_detector_terms(readout, signal, mode="full")
        expanded = (
            np.abs(readout) ** 2
            + 2.0 * np.real(np.conjugate(readout) * signal)
            + np.abs(signal) ** 2
        )

        np.testing.assert_allclose(terms["detector_intensity"], np.abs(readout + signal) ** 2)
        np.testing.assert_allclose(terms["detector_intensity"], expanded)

    def test_weak_detector_omits_signal_intensity_and_has_weak_signal_limit(self):
        readout = np.asarray([1.0 + 0.2j, 0.7 - 0.1j])
        signal = 1.0e-8 * np.asarray([0.4 - 0.2j, -0.3 + 0.5j])

        full = coherent_detector_terms(readout, signal, mode="full")
        weak = coherent_detector_terms(readout, signal, mode="weak")

        np.testing.assert_allclose(
            full["detector_intensity"] - weak["detector_intensity"],
            np.abs(signal) ** 2,
            rtol=1.0e-6,
            atol=1.0e-16,
        )
        np.testing.assert_allclose(
            weak["detector_intensity"],
            np.abs(readout) ** 2 + 2.0 * np.real(np.conjugate(readout) * signal),
        )
        np.testing.assert_allclose(full["detector_intensity"], weak["detector_intensity"], rtol=1.0e-12)


class ReadoutPlanTests(unittest.TestCase):
    def test_absorption_like_uses_canonical_key_and_named_interaction_field(self):
        params = _base_params()
        params = replace(params, field=_field_plan().build_field())
        dynamics = _dynamics(params)
        polarization = compute_polarization_result(dynamics, number_density_m3=1.0e24)
        canonical = ReadoutPlan(
            mode="absorption_like",
            readout_field="probe",
            window="hann",
            subtract_mean=True,
            rel_threshold=1.0e-8,
            zero_padding_factor=2,
            return_intermediates=True,
        ).execute(polarization, interaction_field=params.field)

        assert canonical.spectrum is not None
        self.assertEqual(canonical.mode, "absorption_like")
        self.assertIn("absorption_like_response", canonical.spectrum)
        self.assertNotIn("absorption", canonical.spectrum)
        self.assertEqual(canonical.metadata["readout_field_source"], "interaction_subfield:probe")
        for key in ("energy_eV", "omega_fs_inv", "P_omega", "E_omega", "P_over_E"):
            self.assertIn(key, canonical.spectrum)

    def test_one_dynamics_execution_supports_multiple_external_readouts(self):
        plan = SingleRunPlan(
            base_params=_base_params(),
            field_plan=_field_plan(),
        )
        captured_fields: list[FieldPhySeries] = []

        def fake_run_case(params, **_kwargs):
            captured_fields.append(params.field)
            return _dynamics(params)

        with patch(
            "qudpy_sjh.experiments.pulse_sequence.single_run.run_case",
            side_effect=fake_run_case,
        ) as solver:
            execution = plan.execute()
            density_before = execution.dynamics_result.density_array().copy()
            polarization = compute_polarization_result(
                execution.dynamics_result,
                number_density_m3=1.0e24,
            )
            external = _field("external_lo", amplitude=0.03)
            full = ReadoutPlan(
                mode="full",
                readout_field=external,
                emitted_field_scale=1.0e8,
                zero_padding_factor=1,
            ).execute(polarization, interaction_field=execution.params.field)
            weak = ReadoutPlan(
                mode="weak",
                readout_field=external,
                emitted_field_scale=1.0e8,
                zero_padding_factor=1,
            ).execute(polarization, interaction_field=execution.params.field)

        self.assertEqual(solver.call_count, 1)
        self.assertEqual(captured_fields[0].sub_field_names, ("pump", "probe"))
        self.assertNotIn(external, captured_fields[0].fields)
        np.testing.assert_allclose(execution.dynamics_result.density_array(), density_before)
        assert full.spectrum is not None and weak.spectrum is not None
        self.assertFalse(
            np.allclose(full.spectrum["detector_intensity"], weak.spectrum["detector_intensity"])
        )
        self.assertEqual(full.metadata["readout_field_source"], "external_field")

    def test_fixed_external_readout_phase_changes_interference_not_phase_dimensions(self):
        dynamics = _dynamics(_base_params())
        polarization = compute_polarization_result(dynamics, number_density_m3=1.0e24)
        phase_zero = ReadoutPlan(
            mode="weak",
            readout_field=_field("lo_zero", phase_rad=0.0),
            emitted_field_scale=1.0e8,
            zero_padding_factor=1,
        )
        phase_quadrature = ReadoutPlan(
            mode="weak",
            readout_field=_field("lo_quadrature", phase_rad=0.5 * math.pi),
            emitted_field_scale=1.0e8,
            zero_padding_factor=1,
        )

        zero_result = phase_zero.execute(polarization)
        quadrature_result = phase_quadrature.execute(polarization)

        assert zero_result.spectrum is not None and quadrature_result.spectrum is not None
        self.assertFalse(
            np.allclose(
                zero_result.spectrum["interference_term"],
                quadrature_result.spectrum["interference_term"],
            )
        )
        payload = phase_quadrature.to_dict()
        self.assertEqual(payload["readout_field"]["source"], "external_field")
        self.assertNotIn("phase_tag", payload)

    def test_readout_plan_requires_explicit_polarization_boundary(self):
        with self.assertRaises(TypeError):
            ReadoutPlan(mode="full", readout_field=_field("lo")).execute(object())
        with self.assertRaises(ValueError):
            ReadoutPlan(mode="absorption_like", readout_field="probe").execute(
                PolarizationResult(
                    time_fs=np.asarray([0.0, 1.0]),
                    polarization_C_per_m2=np.asarray([0.0, 1.0]),
                )
            )


if __name__ == "__main__":
    unittest.main()
