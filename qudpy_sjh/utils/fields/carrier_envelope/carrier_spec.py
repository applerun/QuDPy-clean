"""Optical carrier specifications for structured finite fields.

A CarrierSpec represents one quasi-monochromatic carrier only.  Multi-carrier
or broadband special cases should be represented by a composite/special Field,
not by overloading this small spec object.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..lab_fields import _energy_eV_to_fs_inv, _metadata_copy


@dataclass(frozen=True)
class CarrierSpec:
    """Single optical carrier definition.

    Convention
    ----------
    ``phase_rad`` is defined relative to the envelope center.  A structured
    carrier-envelope field should evaluate the real carrier phase as

        omega_fs_inv * (t_fs - envelope.center_fs) + phase_rad

    not as a global lab-frame ``omega_fs_inv * t_fs + phase_rad`` phase.
    """

    omega_fs_inv: float
    phase_rad: float = 0.0
    label: str | None = None
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not np.isfinite(float(self.omega_fs_inv)):
            raise ValueError("omega_fs_inv must be finite.")
        if float(self.omega_fs_inv) < 0:
            raise ValueError("omega_fs_inv must be non-negative.")
        if not np.isfinite(float(self.phase_rad)):
            raise ValueError("phase_rad must be finite.")

    @classmethod
    def from_energy_eV(
        cls,
        laser_energy_eV: float,
        *,
        phase_rad: float = 0.0,
        label: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "CarrierSpec":
        payload = _metadata_copy(metadata)
        payload["laser_energy_eV"] = float(laser_energy_eV)
        return cls(
            omega_fs_inv=_energy_eV_to_fs_inv(float(laser_energy_eV)),
            phase_rad=float(phase_rad),
            label=label,
            metadata=payload,
        )

    @property
    def laser_energy_eV(self) -> float | None:
        metadata = _metadata_copy(self.metadata)
        value = metadata.get("laser_energy_eV")
        return None if value is None else float(value)

    def phase(self, t_fs, *, center_fs: float) -> np.ndarray:
        t = np.asarray(t_fs, dtype=float)
        return float(self.omega_fs_inv) * (t - float(center_fs)) + float(self.phase_rad)

    def to_dict(self) -> dict[str, Any]:
        metadata = _metadata_copy(self.metadata)
        return {
            "class": self.__class__.__name__,
            "omega_fs_inv": float(self.omega_fs_inv),
            "omega_L_fs_inv": float(self.omega_fs_inv),
            "laser_energy_eV": metadata.get("laser_energy_eV"),
            "phase_rad": float(self.phase_rad),
            "label": self.label,
            "phase_convention": "phase_rad is relative to the envelope center",
            "metadata": metadata,
        }

    @classmethod
    def rebuild(cls, payload: dict[str, Any]) -> "CarrierSpec":
        if not isinstance(payload, dict):
            raise TypeError("CarrierSpec.rebuild() expects a dict payload.")
        return cls(
            omega_fs_inv=float(payload.get("omega_fs_inv", payload.get("omega_L_fs_inv"))),
            phase_rad=float(payload.get("phase_rad", 0.0)),
            label=payload.get("label"),
            metadata=dict(payload.get("metadata") or {}),
        )


__all__ = ["CarrierSpec"]
