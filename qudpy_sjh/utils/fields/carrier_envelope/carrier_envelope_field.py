"""Structured carrier-envelope physical fields."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..lab_fields import FieldPhyRoot, _metadata_copy

from .carrier_spec import CarrierSpec
from .envelope_spec import EnvelopeSpec, rebuild_envelope_spec


@dataclass(frozen=True)
class CarrierEnvelopeField(FieldPhyRoot):
    """Single-carrier, single-envelope finite optical field.

    Physical convention
    -------------------
    The carrier phase is local to the envelope center:

        E(t) = 2 E0 f(t) cos[omega * (t - center) + phase]

    where ``center`` is supplied by ``self.envelope.center_fs`` and ``omega`` / ``phase``
    are supplied by ``self.carrier``.

    This class intentionally does not contain a pump/probe/LO role.  Role is an
    experiment-level concept and should be represented by FieldPhySeries
    ``sub_field_names`` or by a higher-level CasePlan.
    """

    E0_MV_per_cm: float
    carrier: CarrierSpec
    envelope: EnvelopeSpec
    name: str = "carrier_envelope_field"
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not np.isfinite(float(self.E0_MV_per_cm)):
            raise ValueError("E0_MV_per_cm must be finite.")
        if not isinstance(self.carrier, CarrierSpec):
            raise TypeError("carrier must be a CarrierSpec instance.")
        if not isinstance(self.envelope, EnvelopeSpec):
            raise TypeError("envelope must be an EnvelopeSpec instance.")

    @property
    def reference_MV_per_cm(self) -> float | None:
        return float(self.E0_MV_per_cm)

    @property
    def normalization_rate_candidates_fs_inv(self) -> tuple[float, ...]:
        return tuple(self.envelope.normalization_rate_candidates_fs_inv)

    def positive_frequency_E_MV_per_cm(self, t_fs: np.ndarray) -> np.ndarray:
        """返回正频复场，单位 MV/cm。

        约定为：

            E_positive(t) = E0 * f(t) * exp(i * [omega*(t-center) + phase])

        因此 real lab-frame field 满足：

            E_real(t) = 2 * Re[E_positive(t)]
        """

        t = np.asarray(t_fs, dtype=float)
        envelope = np.asarray(self.envelope.value(t), dtype=float)
        if envelope.shape != t.shape:
            raise ValueError(
                "envelope.value(t_fs) must return an array with the same shape as t_fs. "
                f"got {envelope.shape}, expected {t.shape}."
            )
        phase = self.carrier.phase(t, center_fs=float(self.envelope.center_fs))
        return float(self.E0_MV_per_cm) * envelope * np.exp(1j * phase)

    def physical_E_MV_per_cm(self, t_fs: np.ndarray) -> np.ndarray:
        return 2.0 * np.real(self.positive_frequency_E_MV_per_cm(t_fs))

    def __repr__(self) -> str:
        return (
            "CarrierEnvelopeField("
            f"E0_MV_per_cm={self.E0_MV_per_cm!r}, "
            f"carrier={self.carrier!r}, "
            f"envelope={self.envelope!r})"
        )

    def with_phase(
        self,
        phase_rad: float,
        *,
        name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "CarrierEnvelopeField":
        """返回 absolute carrier phase 被设为 ``phase_rad`` 的新 field。

        该方法作用于正频复场相位：

            E_positive(t; phi) = exp(i * phi) * E_positive(t; 0)

        对 real field，`physical_E_MV_per_cm` 由新的正频复场取实部得到。
        """

        phase = float(phase_rad)
        if not np.isfinite(phase):
            raise ValueError("phase_rad must be finite.")
        new_metadata = _metadata_copy(self.metadata)
        new_metadata.update(_metadata_copy(metadata))
        new_metadata.update(
            {
                "phase_override_applied": True,
                "phase_override_type": "absolute",
                "previous_phase_rad": float(self.carrier.phase_rad),
                "phase_rad": phase,
            }
        )
        return CarrierEnvelopeField(
            E0_MV_per_cm=float(self.E0_MV_per_cm),
            carrier=CarrierSpec(
                omega_fs_inv=float(self.carrier.omega_fs_inv),
                phase_rad=phase,
                label=self.carrier.label,
                metadata=_metadata_copy(self.carrier.metadata),
            ),
            envelope=self.envelope,
            name=name or self.name,
            metadata=new_metadata,
        )

    def phase_shifted(
        self,
        delta_phase_rad: float,
        *,
        name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "CarrierEnvelopeField":
        """返回 carrier phase 增加 ``delta_phase_rad`` 的新 field。"""

        delta = float(delta_phase_rad)
        if not np.isfinite(delta):
            raise ValueError("delta_phase_rad must be finite.")
        return self.with_phase(
            float(self.carrier.phase_rad) + delta,
            name=name,
            metadata={
                "phase_shift_delta_rad": delta,
                **_metadata_copy(metadata),
            },
        )

    def time_shifted(
        self,
        shift_fs: float,
        *,
        name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "CarrierEnvelopeField":
        """Return a shifted structured field without adding a shift-policy branch.

        The only convention is center shift under the carrier-envelope definition:

            center_new = center_old + shift_fs

        The CarrierSpec is unchanged.  Since carrier phase is defined relative to
        the envelope center, this shifts the finite pulse without introducing an
        additional lab-frame phase convention.
        """

        new_metadata = _metadata_copy(self.metadata)
        new_metadata.update(_metadata_copy(metadata))
        new_metadata.update(
            {
                "time_shift_fs": float(shift_fs),
                "time_shift_semantics": "envelope center shift under carrier-envelope convention",
                "source_field_name": self.name,
                "source_field_repr": repr(self),
            }
        )
        return CarrierEnvelopeField(
            E0_MV_per_cm=float(self.E0_MV_per_cm),
            carrier=self.carrier,
            envelope=self.envelope.shifted(float(shift_fs)),
            name=name or self.name,
            metadata=new_metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        metadata = _metadata_copy(self.metadata)
        carrier_payload = self.carrier.to_dict()
        envelope_payload = self.envelope.to_dict()
        laser_energy_eV = carrier_payload.get("laser_energy_eV")
        return {
            "class": self.__class__.__name__,
            "repr": repr(self),
            "name": self.name,
            "time_unit": self.time_unit,
            "field_unit": self.field_unit,
            "rebuildable": True,
            "E0_MV_per_cm": float(self.E0_MV_per_cm),
            "peak_E_MV_per_cm": 2.0 * float(self.E0_MV_per_cm),
            "omega_fs_inv": float(self.carrier.omega_fs_inv),
            "omega_L_fs_inv": float(self.carrier.omega_fs_inv),
            "laser_energy_eV": laser_energy_eV,
            "phase_rad": float(self.carrier.phase_rad),
            "center_fs": float(self.envelope.center_fs),
            "pulse_center_fs": float(self.envelope.center_fs),
            "carrier": carrier_payload,
            "envelope": envelope_payload,
            "expression": "E(t_fs) = 2 E0 envelope(t_fs) cos[omega*(t_fs-center_fs)+phase]",
            "amplitude_convention": "E0_MV_per_cm is E0 in E(t)=2E0 f(t) cos(...).",
            "phase_convention": "carrier phase is relative to envelope center, not global lab time",
            "description": metadata.get("description"),
            "metadata": metadata,
        }

    @classmethod
    def rebuild(cls, payload: dict[str, Any]) -> "CarrierEnvelopeField":
        if not isinstance(payload, dict):
            raise TypeError("CarrierEnvelopeField.rebuild() expects a dict payload.")
        return cls(
            E0_MV_per_cm=float(payload["E0_MV_per_cm"]),
            carrier=CarrierSpec.rebuild(payload["carrier"]),
            envelope=rebuild_envelope_spec(payload["envelope"]),
            name=str(payload.get("name", "carrier_envelope_field")),
            metadata=dict(payload.get("metadata") or {}),
        )


__all__ = ["CarrierEnvelopeField"]
