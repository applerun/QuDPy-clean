"""Convenience builders for structured carrier-envelope fields."""

from __future__ import annotations

from typing import Any

from ..field_series import FieldPhySeries
from ..lab_fields import _metadata_copy

from .carrier_envelope_field import CarrierEnvelopeField
from .carrier_spec import CarrierSpec
from .envelope_spec import ConstantEnvelopeSpec, GaussianEnvelopeSpec, SechEnvelopeSpec


def make_carrier_envelope_field(
    *,
    E0_MV_per_cm: float,
    carrier: CarrierSpec,
    envelope,
    name: str = "carrier_envelope_field",
    metadata: dict[str, Any] | None = None,
) -> CarrierEnvelopeField:
    return CarrierEnvelopeField(
        E0_MV_per_cm=float(E0_MV_per_cm),
        carrier=carrier,
        envelope=envelope,
        name=name,
        metadata=_metadata_copy(metadata),
    )


def make_gaussian_carrier_envelope_field(
    *,
    E0_MV_per_cm: float,
    laser_energy_eV: float,
    center_fs: float,
    sigma_fs: float,
    phase_rad: float = 0.0,
    envelope_amplitude: float = 1.0,
    name: str = "gaussian_carrier_envelope_field",
    metadata: dict[str, Any] | None = None,
) -> CarrierEnvelopeField:
    return CarrierEnvelopeField(
        E0_MV_per_cm=float(E0_MV_per_cm),
        carrier=CarrierSpec.from_energy_eV(
            float(laser_energy_eV),
            phase_rad=float(phase_rad),
            metadata={"laser_energy_eV": float(laser_energy_eV)},
        ),
        envelope=GaussianEnvelopeSpec(
            center_fs=float(center_fs),
            sigma_fs=float(sigma_fs),
            amplitude=float(envelope_amplitude),
        ),
        name=name,
        metadata=_metadata_copy(metadata),
    )


def make_sech_carrier_envelope_field(
    *,
    E0_MV_per_cm: float,
    laser_energy_eV: float,
    center_fs: float,
    tau_fs: float,
    phase_rad: float = 0.0,
    envelope_amplitude: float = 1.0,
    name: str = "sech_carrier_envelope_field",
    metadata: dict[str, Any] | None = None,
) -> CarrierEnvelopeField:
    return CarrierEnvelopeField(
        E0_MV_per_cm=float(E0_MV_per_cm),
        carrier=CarrierSpec.from_energy_eV(
            float(laser_energy_eV),
            phase_rad=float(phase_rad),
            metadata={"laser_energy_eV": float(laser_energy_eV)},
        ),
        envelope = SechEnvelopeSpec(
            width_fs = float(tau_fs),
            center_fs = float(center_fs),
            amplitude = float(envelope_amplitude),
        ),
        name=name,
        metadata=_metadata_copy(metadata),
    )


def make_constant_carrier_envelope_field(
    *,
    E0_MV_per_cm: float,
    laser_energy_eV: float,
    phase_rad: float = 0.0,
    center_fs: float = 0.0,
    envelope_amplitude: float = 1.0,
    name: str = "constant_carrier_envelope_field",
    metadata: dict[str, Any] | None = None,
) -> CarrierEnvelopeField:
    return CarrierEnvelopeField(
        E0_MV_per_cm=float(E0_MV_per_cm),
        carrier=CarrierSpec.from_energy_eV(
            float(laser_energy_eV),
            phase_rad=float(phase_rad),
            metadata={"laser_energy_eV": float(laser_energy_eV)},
        ),
        envelope=ConstantEnvelopeSpec(
            center_fs=float(center_fs),
            amplitude=float(envelope_amplitude),
        ),
        name=name,
        metadata=_metadata_copy(metadata),
    )


def make_pump_probe_field_series(
    *,
    pump_field: CarrierEnvelopeField,
    probe_field: CarrierEnvelopeField,
    name: str = "pump_probe_field_series",
    metadata: dict[str, Any] | None = None,
) -> FieldPhySeries:
    """Create a pump+probe linear field series.

    Pump/probe role is represented here at the series/case level, not inside
    CarrierEnvelopeField itself.
    """

    return FieldPhySeries(
        fields=(pump_field, probe_field),
        sub_field_names=("pump", "probe"),
        name=name,
        metadata=_metadata_copy(metadata),
    )


__all__ = [
    "make_carrier_envelope_field",
    "make_gaussian_carrier_envelope_field",
    "make_sech_carrier_envelope_field",
    "make_constant_carrier_envelope_field",
    "make_pump_probe_field_series",
]
