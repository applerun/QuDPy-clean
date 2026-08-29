"""Multi-carrier structured optical fields.

This module adds a deterministic coherent multi-carrier field while preserving
existing carrier/envelope separation. Experimental phase cycling is represented
as one global phase applied identically to every carrier component.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from ..lab_fields import FieldPhyRoot, _metadata_copy
from .carrier_spec import CarrierSpec
from .envelope_spec import EnvelopeSpec, rebuild_envelope_spec


@dataclass(frozen=True)
class MultiCarrierComponent:
    """One coherent spectral component.

    ``carrier.phase_rad`` is the intrinsic/spectral phase of this component.
    ``amplitude`` is a dimensionless field-amplitude weight.
    """

    carrier: CarrierSpec
    amplitude: float = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.carrier, CarrierSpec):
            raise TypeError("carrier must be a CarrierSpec instance.")
        if not np.isfinite(float(self.amplitude)):
            raise ValueError("amplitude must be finite.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "class": self.__class__.__name__,
            "carrier": self.carrier.to_dict(),
            "amplitude": float(self.amplitude),
        }

    @classmethod
    def rebuild(cls, payload: dict[str, Any]) -> "MultiCarrierComponent":
        if not isinstance(payload, dict):
            raise TypeError("MultiCarrierComponent.rebuild() expects a dict payload.")
        return cls(
            carrier=CarrierSpec.rebuild(payload["carrier"]),
            amplitude=float(payload.get("amplitude", 1.0)),
        )


@dataclass(frozen=True)
class MultiCarrierEnvelopeField(FieldPhyRoot):
    """Finite optical field with one envelope and multiple coherent carriers.

    Convention
    ----------
    E_positive(t) = E0 * f(t) * sum_k a_k exp{i[omega_k(t-center)+theta_k+phi_global]}
    E_real(t) = 2 Re[E_positive(t)]

    ``theta_k`` is intrinsic/spectral phase. ``phi_global`` is the experimental
    common optical phase. Phase cycling must act on ``phi_global`` only, so one
    physical pulse remains one phase-cycling degree of freedom even when it has
    many carrier components.
    """

    E0_MV_per_cm: float
    components: tuple[MultiCarrierComponent, ...]
    envelope: EnvelopeSpec
    global_phase_rad: float = 0.0
    name: str = "multi_carrier_envelope_field"
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not np.isfinite(float(self.E0_MV_per_cm)):
            raise ValueError("E0_MV_per_cm must be finite.")
        if not isinstance(self.envelope, EnvelopeSpec):
            raise TypeError("envelope must be an EnvelopeSpec instance.")
        if not np.isfinite(float(self.global_phase_rad)):
            raise ValueError("global_phase_rad must be finite.")
        components = tuple(self.components)
        if not components:
            raise ValueError("components must contain at least one carrier.")
        if not all(isinstance(item, MultiCarrierComponent) for item in components):
            raise TypeError("components must contain only MultiCarrierComponent instances.")
        object.__setattr__(self, "components", components)

    @classmethod
    def from_carriers(
        cls,
        *,
        E0_MV_per_cm: float,
        carriers: Iterable[CarrierSpec],
        envelope: EnvelopeSpec,
        amplitudes: Iterable[float] | None = None,
        global_phase_rad: float = 0.0,
        name: str = "multi_carrier_envelope_field",
        metadata: dict[str, Any] | None = None,
    ) -> "MultiCarrierEnvelopeField":
        carrier_tuple = tuple(carriers)
        if amplitudes is None:
            amplitude_tuple = (1.0,) * len(carrier_tuple)
        else:
            amplitude_tuple = tuple(float(x) for x in amplitudes)
            if len(amplitude_tuple) != len(carrier_tuple):
                raise ValueError("amplitudes must have the same length as carriers.")
        return cls(
            E0_MV_per_cm=float(E0_MV_per_cm),
            components=tuple(
                MultiCarrierComponent(carrier=c, amplitude=a)
                for c, a in zip(carrier_tuple, amplitude_tuple)
            ),
            envelope=envelope,
            global_phase_rad=float(global_phase_rad),
            name=name,
            metadata=metadata,
        )

    @property
    def reference_MV_per_cm(self) -> float | None:
        return float(self.E0_MV_per_cm)

    @property
    def normalization_rate_candidates_fs_inv(self) -> tuple[float, ...]:
        return tuple(self.envelope.normalization_rate_candidates_fs_inv)

    @property
    def carrier_count(self) -> int:
        return len(self.components)

    def positive_frequency_E_MV_per_cm(self, t_fs: np.ndarray) -> np.ndarray:
        t = np.asarray(t_fs, dtype=float)
        envelope = np.asarray(self.envelope.value(t), dtype=float)
        if envelope.shape != t.shape:
            raise ValueError(
                "envelope.value(t_fs) must return an array with the same shape as t_fs. "
                f"got {envelope.shape}, expected {t.shape}."
            )
        center_fs = float(self.envelope.center_fs)
        phi_global = float(self.global_phase_rad)
        spectral_sum = np.zeros(t.shape, dtype=complex)
        for component in self.components:
            phase = component.carrier.phase(t, center_fs=center_fs)
            spectral_sum += float(component.amplitude) * np.exp(1j * (phase + phi_global))
        return float(self.E0_MV_per_cm) * envelope * spectral_sum

    def physical_E_MV_per_cm(self, t_fs: np.ndarray) -> np.ndarray:
        return 2.0 * np.real(self.positive_frequency_E_MV_per_cm(t_fs))

    def with_phase(
        self,
        phase_rad: float,
        *,
        name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "MultiCarrierEnvelopeField":
        """Set the global experimental optical phase.

        All carriers receive the same phase shift; intrinsic relative spectral
        phases are unchanged.
        """
        phase = float(phase_rad)
        if not np.isfinite(phase):
            raise ValueError("phase_rad must be finite.")
        new_metadata = _metadata_copy(self.metadata)
        new_metadata.update(_metadata_copy(metadata))
        new_metadata.update({
            "phase_override_applied": True,
            "phase_override_type": "absolute_global",
            "previous_global_phase_rad": float(self.global_phase_rad),
            "global_phase_rad": phase,
        })
        return MultiCarrierEnvelopeField(
            E0_MV_per_cm=float(self.E0_MV_per_cm),
            components=self.components,
            envelope=self.envelope,
            global_phase_rad=phase,
            name=name or self.name,
            metadata=new_metadata,
        )

    def phase_shifted(
        self,
        delta_phase_rad: float,
        *,
        name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "MultiCarrierEnvelopeField":
        delta = float(delta_phase_rad)
        if not np.isfinite(delta):
            raise ValueError("delta_phase_rad must be finite.")
        return self.with_phase(
            float(self.global_phase_rad) + delta,
            name=name,
            metadata={"phase_shift_delta_rad": delta, **_metadata_copy(metadata)},
        )

    def time_shifted(
        self,
        shift_fs: float,
        *,
        name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "MultiCarrierEnvelopeField":
        shift = float(shift_fs)
        if not np.isfinite(shift):
            raise ValueError("shift_fs must be finite.")
        new_metadata = _metadata_copy(self.metadata)
        new_metadata.update(_metadata_copy(metadata))
        new_metadata.update({
            "time_shift_fs": shift,
            "time_shift_semantics": "common envelope center shift under multi-carrier carrier-envelope convention",
            "source_field_name": self.name,
            "source_field_repr": repr(self),
        })
        return MultiCarrierEnvelopeField(
            E0_MV_per_cm=float(self.E0_MV_per_cm),
            components=self.components,
            envelope=self.envelope.shifted(shift),
            global_phase_rad=float(self.global_phase_rad),
            name=name or self.name,
            metadata=new_metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        metadata = _metadata_copy(self.metadata)
        return {
            "class": self.__class__.__name__,
            "repr": repr(self),
            "name": self.name,
            "time_unit": self.time_unit,
            "field_unit": self.field_unit,
            "rebuildable": True,
            "E0_MV_per_cm": float(self.E0_MV_per_cm),
            "global_phase_rad": float(self.global_phase_rad),
            "carrier_count": int(self.carrier_count),
            "components": [item.to_dict() for item in self.components],
            "envelope": self.envelope.to_dict(),
            "center_fs": float(self.envelope.center_fs),
            "expression": (
                "E_positive(t)=E0*envelope(t)*sum_k a_k*exp(i*[omega_k*(t-center)+theta_k+phi_global]); "
                "E_real(t)=2*Re[E_positive(t)]"
            ),
            "amplitude_convention": (
                "E0_MV_per_cm is a common field-amplitude scale; each component has a dimensionless relative amplitude a_k."
            ),
            "phase_convention": (
                "CarrierSpec.phase_rad stores intrinsic/spectral phase; global_phase_rad is the common experimental phase applied to all carriers."
            ),
            "description": metadata.get("description"),
            "metadata": metadata,
        }

    @classmethod
    def rebuild(cls, payload: dict[str, Any]) -> "MultiCarrierEnvelopeField":
        if not isinstance(payload, dict):
            raise TypeError("MultiCarrierEnvelopeField.rebuild() expects a dict payload.")
        return cls(
            E0_MV_per_cm=float(payload["E0_MV_per_cm"]),
            components=tuple(MultiCarrierComponent.rebuild(x) for x in payload["components"]),
            envelope=rebuild_envelope_spec(payload["envelope"]),
            global_phase_rad=float(payload.get("global_phase_rad", 0.0)),
            name=str(payload.get("name", "multi_carrier_envelope_field")),
            metadata=dict(payload.get("metadata") or {}),
        )


__all__ = ["MultiCarrierComponent", "MultiCarrierEnvelopeField"]
