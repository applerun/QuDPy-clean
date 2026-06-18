"""Base lab-frame physical field interfaces and generic wrappers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np

HBAR_J_S = 1.054571817e-34
E_CHARGE_C = 1.602176634e-19
FS_TO_S = 1e-15
EV_TO_FS_INV = (E_CHARGE_C / HBAR_J_S) * FS_TO_S


def _metadata_copy(metadata: dict[str, Any] | None) -> dict[str, Any]:
	return dict(metadata or {})


def _energy_eV_to_fs_inv(energy_eV: float) -> float:
	return float(energy_eV) * EV_TO_FS_INV


class FieldPhyRoot(ABC):
	"""Base class for lab-frame physical electric fields.

	Subclasses implement ``physical_E_MV_per_cm(t_fs)`` in physical units:
	time in fs and electric field in MV/cm.
	"""

	time_unit = "fs"
	field_unit = "MV/cm"

	def __call__(self, t_fs):
		t_array = np.asarray(t_fs, dtype=float)
		values = np.asarray(self.physical_E_MV_per_cm(t_array), dtype=float)
		if values.shape != t_array.shape:
			raise ValueError(
				"physical_E_MV_per_cm(t_fs) must return an array with the same shape as t_fs. "
				f"got {values.shape}, expected {t_array.shape}."
			)
		if np.ndim(t_fs) == 0:
			return float(values)
		return values

	@abstractmethod
	def physical_E_MV_per_cm(self, t_fs: np.ndarray) -> np.ndarray:
		"""Return lab-frame ``E(t_fs)`` in MV/cm."""

	@abstractmethod
	def __repr__(self) -> str:
		"""Return a readable field description."""

	@property
	def reference_MV_per_cm(self) -> float | None:
		return None

	@property
	def normalization_rate_candidates_fs_inv(self) -> tuple[float, ...]:
		return ()

	def to_dict(self) -> dict[str, Any]:
		return {
			"class": self.__class__.__name__,
			"repr": repr(self),
			"time_unit": self.time_unit,
			"field_unit": self.field_unit,
			"rebuildable": False,
		}

	@classmethod
	def rebuild(cls, payload):
		raise NotImplementedError(f"{cls.__name__}.rebuild() is not implemented.")

	def time_shifted(
			self,
			shift_fs: float,
			*,
			name: str | None = None,
			metadata: dict[str, Any] | None = None,
	):
		return TimeShiftedField(self, shift_fs=shift_fs, name=name, metadata=metadata)


class TimeShiftedField(FieldPhyRoot):
	"""Non-mutating time-shift wrapper for a physical field.

	``shift_fs > 0`` moves the field later in time:
	``E_shifted(t) = E_original(t - shift_fs)``.
	"""

	def __init__(
			self,
			base_field: FieldPhyRoot,
			shift_fs: float,
			*,
			name: str | None = None,
			metadata: dict[str, Any] | None = None,
	):
		if not isinstance(base_field, FieldPhyRoot):
			raise TypeError("base_field must be a FieldPhyRoot instance.")
		self.base_field = base_field
		self.shift_fs = float(shift_fs)
		base_name = getattr(base_field, "name", base_field.__class__.__name__)
		self.name = name or f"{base_name}_shifted_{self.shift_fs:g}_fs"
		self.metadata = _metadata_copy(metadata)

	@property
	def reference_MV_per_cm(self) -> float | None:
		return self.base_field.reference_MV_per_cm

	@property
	def normalization_rate_candidates_fs_inv(self) -> tuple[float, ...]:
		return self.base_field.normalization_rate_candidates_fs_inv

	def physical_E_MV_per_cm(self, t_fs: np.ndarray) -> np.ndarray:
		t_array = np.asarray(t_fs, dtype=float)
		return self.base_field.physical_E_MV_per_cm(t_array - self.shift_fs)

	def __repr__(self) -> str:
		return f"TimeShiftedField(base_field={self.base_field!r}, shift_fs={self.shift_fs!r})"

	def to_dict(self) -> dict[str, Any]:
		base_payload = self.base_field.to_dict() if hasattr(self.base_field, "to_dict") else {}
		source_name = base_payload.get("name") or getattr(self.base_field, "name", None)
		metadata = _metadata_copy(self.metadata)
		metadata.update(
			{
				"time_shift_fs": float(self.shift_fs),
				"source_field_name": source_name,
				"source_field_repr": repr(self.base_field),
				"shift_rule": "E_shifted(t) = E_original(t - shift_fs)",
			}
		)
		payload = {
			"class": self.__class__.__name__,
			"repr": repr(self),
			"name": self.name,
			"time_unit": self.time_unit,
			"field_unit": self.field_unit,
			"rebuildable": False,
			"time_shift_fs": float(self.shift_fs),
			"source_field_name": source_name,
			"source_field_repr": repr(self.base_field),
			"source_field": base_payload if base_payload else repr(self.base_field),
			"shift_rule": "E_shifted(t) = E_original(t - shift_fs)",
			"expression": "E_shifted(t_fs) = E_original(t_fs - shift_fs)",
			"metadata": metadata,
		}
		for key in (
				"sigma_fs",
				"pulse_sigma_fs",
				"omega_L_fs_inv",
				"laser_energy_eV",
				"phase_rad",
				"E0_MV_per_cm",
				"peak_E_MV_per_cm",
		):
			if key in base_payload:
				payload[key] = base_payload[key]
		if "center_fs" in base_payload:
			payload["center_fs"] = float(base_payload["center_fs"]) + float(self.shift_fs)
		if "pulse_center_fs" in base_payload:
			payload["pulse_center_fs"] = float(base_payload["pulse_center_fs"]) + float(self.shift_fs)
		return payload

	def time_shifted(
			self,
			shift_fs: float,
			*,
			name: str | None = None,
			metadata: dict[str, Any] | None = None,
	):
		combined_metadata = _metadata_copy(self.metadata)
		combined_metadata.update(_metadata_copy(metadata))
		return TimeShiftedField(
			self.base_field,
			shift_fs=self.shift_fs + float(shift_fs),
			name=name,
			metadata=combined_metadata,
		)


class FieldPhyCustomed(FieldPhyRoot):
	"""Recommended base class for user-defined physical fields."""

	@classmethod
	def rebuild(cls, payload):
		raise NotImplementedError(
			"Custom physical fields are not rebuildable unless the subclass implements rebuild()."
		)


@dataclass(frozen=True)
class _CodeFieldAdapter:
	"""Internal solver-unit adapter for physical fields."""

	field_phy: FieldPhyRoot
	normalizer: Any
	solver: Any
	reference_field_MV_per_cm: float
	name: str = "internal_code_field_adapter"

	def __call__(self, t_code):
		t_fs = self.normalizer.denormalize_time_array(
			np.asarray(t_code, dtype=float), self.solver
		)
		E_MV_per_cm = self.field_phy(t_fs)
		return self.normalizer.normalize_field_MV_per_cm(
			E_MV_per_cm,
			reference_field_MV_per_cm=self.reference_field_MV_per_cm,
		)

	def physical(self, t_fs):
		return self.field_phy(t_fs)

	def to_dict(self) -> dict[str, Any]:
		return {
			"class": self.__class__.__name__,
			"domain": "solver_code",
			"time_unit": "code",
			"field_unit": "code",
			"time_scale_fs": float(self.solver.time_scale_fs),
			"field_scale": "E_code = E_MV_per_cm / reference_field_MV_per_cm",
			"reference_field_MV_per_cm": float(self.reference_field_MV_per_cm),
			"source_field": self.field_phy.to_dict(),
		}

	def to_expr(self) -> str:
		return f"E_code(t_code) = E_phys(t_fs) / {self.reference_field_MV_per_cm:g} MV/cm"


def make_code_field_adapter(
		field_phy: FieldPhyRoot,
		normalizer,
		solver,
		*,
		reference_field_MV_per_cm: float,
):
	if not isinstance(field_phy, FieldPhyRoot):
		raise TypeError("field_phy must be a FieldPhyRoot instance.")
	return _CodeFieldAdapter(
		field_phy=field_phy,
		normalizer=normalizer,
		solver=solver,
		reference_field_MV_per_cm=float(reference_field_MV_per_cm),
	)


__all__ = [
	"HBAR_J_S",
	"E_CHARGE_C",
	"FS_TO_S",
	"EV_TO_FS_INV",
	"_metadata_copy",
	"_energy_eV_to_fs_inv",
	"FieldPhyRoot",
	"TimeShiftedField",
	"FieldPhyCustomed",
	"_CodeFieldAdapter",
	"make_code_field_adapter",
]
