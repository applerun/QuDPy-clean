"""Built-in lab-frame physical field implementations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..lab_fields import FieldPhyRoot, _energy_eV_to_fs_inv, _metadata_copy


@dataclass(frozen=True)
class CarrierFieldPhysical(FieldPhyRoot):
	"""CW lab-frame carrier field."""

	E0_MV_per_cm: float
	omega_L_fs_inv: float
	phase_rad: float = 0.0
	name: str = "carrier_field_physical"
	metadata: dict[str, Any] | None = None

	@property
	def reference_MV_per_cm(self) -> float | None:
		return float(self.E0_MV_per_cm)

	def physical_E_MV_per_cm(self, t_fs: np.ndarray) -> np.ndarray:
		return 2.0 * float(self.E0_MV_per_cm) * np.cos(
			float(self.omega_L_fs_inv) * t_fs + float(self.phase_rad)
		)

	def __repr__(self) -> str:
		return (
			"CarrierFieldPhysical("
			f"E0_MV_per_cm={self.E0_MV_per_cm!r}, "
			f"omega_L_fs_inv={self.omega_L_fs_inv!r}, "
			f"phase_rad={self.phase_rad!r})"
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
			"peak_E_MV_per_cm": 2.0 * float(self.E0_MV_per_cm),
			"omega_L_fs_inv": float(self.omega_L_fs_inv),
			"laser_energy_eV": metadata.get("laser_energy_eV"),
			"phase_rad": float(self.phase_rad),
			"envelope": "constant",
			"expression": "E(t_fs) = 2 E0 cos(omega_L t_fs + phase)",
			"amplitude_convention": "E0_MV_per_cm is E0 in E(t)=2E0 cos(...).",
			"description": metadata.get("description"),
			"metadata": metadata,
		}

	@classmethod
	def rebuild(cls, payload):
		if not isinstance(payload, dict):
			raise TypeError("CarrierFieldPhysical.rebuild() expects a dict payload.")
		return cls(
			E0_MV_per_cm=float(payload["E0_MV_per_cm"]),
			omega_L_fs_inv=float(payload["omega_L_fs_inv"]),
			phase_rad=float(payload.get("phase_rad", 0.0)),
			name=str(payload.get("name", "carrier_field_physical")),
			metadata=dict(payload.get("metadata") or {}),
		)


@dataclass(frozen=True)
class GaussianCarrierFieldPhysical(FieldPhyRoot):
	"""Gaussian-envelope lab-frame carrier field."""

	E0_MV_per_cm: float
	omega_L_fs_inv: float
	center_fs: float
	sigma_fs: float
	phase_rad: float = 0.0
	name: str = "gaussian_carrier_field_physical"
	metadata: dict[str, Any] | None = None

	@property
	def reference_MV_per_cm(self) -> float | None:
		return float(self.E0_MV_per_cm)

	def __post_init__(self) -> None:
		if self.sigma_fs <= 0:
			raise ValueError("sigma_fs must be positive.")

	@property
	def normalization_rate_candidates_fs_inv(self) -> tuple[float, ...]:
		return (1.0 / float(self.sigma_fs),)

	def physical_E_MV_per_cm(self, t_fs: np.ndarray) -> np.ndarray:
		envelope = np.exp(
			-((t_fs - float(self.center_fs)) ** 2)
			/ (2.0 * float(self.sigma_fs) ** 2)
		)
		return 2.0 * float(self.E0_MV_per_cm) * envelope * np.cos(
			float(self.omega_L_fs_inv) * t_fs + float(self.phase_rad)
		)

	def __repr__(self) -> str:
		return (
			"GaussianCarrierFieldPhysical("
			f"E0_MV_per_cm={self.E0_MV_per_cm!r}, "
			f"omega_L_fs_inv={self.omega_L_fs_inv!r}, "
			f"center_fs={self.center_fs!r}, "
			f"sigma_fs={self.sigma_fs!r}, "
			f"phase_rad={self.phase_rad!r})"
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
			"peak_E_MV_per_cm": 2.0 * float(self.E0_MV_per_cm),
			"omega_L_fs_inv": float(self.omega_L_fs_inv),
			"laser_energy_eV": metadata.get("laser_energy_eV"),
			"phase_rad": float(self.phase_rad),
			"center_fs": float(self.center_fs),
			"sigma_fs": float(self.sigma_fs),
			"pulse_center_fs": float(self.center_fs),
			"pulse_sigma_fs": float(self.sigma_fs),
			"envelope": "gaussian",
			"expression": "E(t_fs) = 2 E0 exp[-(t_fs-center)^2/(2 sigma^2)] cos(omega_L t_fs + phase)",
			"amplitude_convention": "E0_MV_per_cm is E0 in E(t)=2E0 f(t) cos(...).",
			"description": metadata.get("description"),
			"metadata": metadata,
		}

	@classmethod
	def rebuild(cls, payload):
		if not isinstance(payload, dict):
			raise TypeError("GaussianCarrierFieldPhysical.rebuild() expects a dict payload.")
		return cls(
			E0_MV_per_cm=float(payload["E0_MV_per_cm"]),
			omega_L_fs_inv=float(payload["omega_L_fs_inv"]),
			center_fs=float(payload["center_fs"]),
			sigma_fs=float(payload["sigma_fs"]),
			phase_rad=float(payload.get("phase_rad", 0.0)),
			name=str(payload.get("name", "gaussian_carrier_field_physical")),
			metadata=dict(payload.get("metadata") or {}),
		)


def make_default_carrier_field(
		*,
		E0_MV_per_cm: float,
		laser_energy_eV: float,
		phase_rad: float = 0.0,
		name: str = "explicit_carrier_field",
		metadata: dict[str, Any] | None = None,
) -> CarrierFieldPhysical:
	payload = _metadata_copy(metadata)
	payload["laser_energy_eV"] = float(laser_energy_eV)
	payload.setdefault("source", "explicit field helper")
	return CarrierFieldPhysical(
		E0_MV_per_cm=float(E0_MV_per_cm),
		omega_L_fs_inv=_energy_eV_to_fs_inv(laser_energy_eV),
		phase_rad=float(phase_rad),
		name=name,
		metadata=payload,
	)


def make_default_gaussian_carrier_field(
		*,
		E0_MV_per_cm: float,
		laser_energy_eV: float,
		pulse_center_fs: float,
		pulse_sigma_fs: float,
		phase_rad: float = 0.0,
		name: str = "explicit_gaussian_carrier_field",
		metadata: dict[str, Any] | None = None,
) -> GaussianCarrierFieldPhysical:
	payload = _metadata_copy(metadata)
	payload["laser_energy_eV"] = float(laser_energy_eV)
	payload.setdefault("source", "explicit field helper")
	return GaussianCarrierFieldPhysical(
		E0_MV_per_cm=float(E0_MV_per_cm),
		omega_L_fs_inv=_energy_eV_to_fs_inv(laser_energy_eV),
		center_fs=float(pulse_center_fs),
		sigma_fs=float(pulse_sigma_fs),
		phase_rad=float(phase_rad),
		name=name,
		metadata=payload,
	)


def rebuild_physical_field(payload) -> FieldPhyRoot:
	if not isinstance(payload, dict):
		raise TypeError("rebuild_physical_field() expects a dict payload.")
	class_name = payload.get("class")
	registry = {
		"CarrierFieldPhysical": CarrierFieldPhysical,
		"GaussianCarrierFieldPhysical": GaussianCarrierFieldPhysical,
	}
	if class_name not in registry:
		raise ValueError(f"Unknown or non-rebuildable physical field class: {class_name!r}.")
	return registry[class_name].rebuild(payload)


__all__ = [
	"CarrierFieldPhysical",
	"GaussianCarrierFieldPhysical",
	"make_default_carrier_field",
	"make_default_gaussian_carrier_field",
	"rebuild_physical_field",
]
