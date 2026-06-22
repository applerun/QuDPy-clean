"""Structured carrier-envelope field API.

This subpackage provides a compact, finite-pulse field layer for workflows that
need explicit carrier phase and envelope semantics, such as phase cycling and
pump-probe delay scans.
"""

from .carrier_envelope_field import CarrierEnvelopeField
from .carrier_spec import CarrierSpec
from .envelope_spec import (
    ConstantEnvelopeSpec,
    EnvelopeSpec,
    GaussianEnvelopeSpec,
    SechEnvelopeSpec,
    rebuild_envelope_spec,
)
from .builders import (
    make_carrier_envelope_field,
    make_constant_carrier_envelope_field,
    make_gaussian_carrier_envelope_field,
    make_pump_probe_field_series,
    make_sech_carrier_envelope_field,
)


__all__ = [
    "CarrierSpec",
    "EnvelopeSpec",
    "GaussianEnvelopeSpec",
    "SechEnvelopeSpec",
    "ConstantEnvelopeSpec",
    "rebuild_envelope_spec",
    "CarrierEnvelopeField",
    "make_carrier_envelope_field",
    "make_gaussian_carrier_envelope_field",
    "make_sech_carrier_envelope_field",
    "make_constant_carrier_envelope_field",
    "make_pump_probe_field_series",
]
