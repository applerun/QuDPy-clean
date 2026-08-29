"""Core tests for MultiCarrierEnvelopeField."""

from __future__ import annotations

import numpy as np

from qudpy_sjh.utils.fields.carrier_envelope.carrier_spec import CarrierSpec
from qudpy_sjh.utils.fields.carrier_envelope.envelope_spec import GaussianEnvelopeSpec
from qudpy_sjh.utils.fields.carrier_envelope.multi_carrier_envelope_field import (
    MultiCarrierComponent,
    MultiCarrierEnvelopeField,
)


def _field() -> MultiCarrierEnvelopeField:
    return MultiCarrierEnvelopeField(
        E0_MV_per_cm=0.2,
        components=(
            MultiCarrierComponent(CarrierSpec(omega_fs_inv=2.0, phase_rad=0.10), 1.0),
            MultiCarrierComponent(CarrierSpec(omega_fs_inv=2.4, phase_rad=-0.30), 0.5),
        ),
        envelope=GaussianEnvelopeSpec(sigma_fs=40.0, center_fs=10.0),
        global_phase_rad=0.25,
    )


def test_real_field_is_twice_positive_frequency_real_part():
    field = _field()
    t = np.linspace(-20.0, 40.0, 101)
    assert np.allclose(
        field.physical_E_MV_per_cm(t),
        2.0 * np.real(field.positive_frequency_E_MV_per_cm(t)),
    )


def test_global_phase_shift_multiplies_positive_frequency_field():
    field = _field()
    t = np.linspace(-20.0, 40.0, 101)
    delta = 0.73
    before = field.positive_frequency_E_MV_per_cm(t)
    after = field.phase_shifted(delta).positive_frequency_E_MV_per_cm(t)
    assert np.allclose(after, np.exp(1j * delta) * before)


def test_with_phase_preserves_intrinsic_relative_carrier_phases():
    field = _field()
    shifted = field.with_phase(1.2)
    assert shifted.global_phase_rad == 1.2
    assert shifted.components == field.components


def test_time_shift_moves_common_envelope_center_only():
    field = _field()
    shifted = field.time_shifted(17.0)
    assert shifted.envelope.center_fs == field.envelope.center_fs + 17.0
    assert shifted.components == field.components
    assert shifted.global_phase_rad == field.global_phase_rad


def test_round_trip():
    field = _field()
    rebuilt = MultiCarrierEnvelopeField.rebuild(field.to_dict())
    t = np.linspace(-40.0, 60.0, 121)
    assert np.allclose(
        rebuilt.positive_frequency_E_MV_per_cm(t),
        field.positive_frequency_E_MV_per_cm(t),
    )
