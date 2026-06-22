"""Envelope specifications for structured carrier-envelope optical fields."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from typing import Any

import numpy as np


def _metadata_copy(metadata: dict[str, Any] | None) -> dict[str, Any]:
	return dict(metadata or {})


class EnvelopeSpec(ABC):
	"""Abstract base class for optical pulse envelopes.

	Notes
	-----
	The envelope is dimensionless. It should normally be normalized so that its
	peak value is close to one, while the field amplitude is controlled by
	CarrierEnvelopeField.E0_MV_per_cm.

	Important implementation note
	-----------------------------
	Do not define an abstract ``center_fs`` property here. In Python 3.9,
	dataclass subclasses may treat an inherited property with the same name as a
	default value, causing field-order errors such as:

		TypeError: non-default argument 'sigma_fs' follows default argument

	Concrete envelope dataclasses should define ``center_fs`` directly.
	"""

	@abstractmethod
	def value(self, t_fs) -> np.ndarray:
		"""Return dimensionless envelope values."""

	@abstractmethod
	def shifted(self, shift_fs: float) -> "EnvelopeSpec":
		"""Return a new envelope with its reference center shifted by shift_fs."""

	@abstractmethod
	def to_dict(self) -> dict[str, Any]:
		"""Return rebuildable metadata."""

	@property
	def normalization_rate_candidates_fs_inv(self) -> tuple[float, ...]:
		"""Characteristic envelope rates for numerical normalization."""
		return ()


@dataclass(frozen = True)
class GaussianEnvelopeSpec(EnvelopeSpec):
	"""Gaussian envelope.

	envelope(t) = amplitude * exp[-(t-center)^2 / (2 sigma^2)]
	"""

	sigma_fs: float
	center_fs: float = 0.0
	amplitude: float = 1.0
	label: str | None = None
	metadata: dict[str, Any] | None = None

	def __post_init__(self) -> None:
		if float(self.sigma_fs) <= 0.0:
			raise ValueError("sigma_fs must be positive.")

	def value(self, t_fs) -> np.ndarray:
		t = np.asarray(t_fs, dtype = float)
		center = float(self.center_fs)
		sigma = float(self.sigma_fs)
		return float(self.amplitude) * np.exp(-((t - center) ** 2) / (2.0 * sigma ** 2))

	def shifted(self, shift_fs: float) -> "GaussianEnvelopeSpec":
		return replace(self, center_fs = float(self.center_fs) + float(shift_fs))

	@property
	def normalization_rate_candidates_fs_inv(self) -> tuple[float, ...]:
		return (1.0 / float(self.sigma_fs),)

	def to_dict(self) -> dict[str, Any]:
		return {
			"class": self.__class__.__name__,
			"center_fs": float(self.center_fs),
			"sigma_fs": float(self.sigma_fs),
			"amplitude": float(self.amplitude),
			"label": self.label,
			"metadata": _metadata_copy(self.metadata),
			"expression": "amplitude * exp[-(t-center)^2/(2*sigma^2)]",
		}

	@classmethod
	def rebuild(cls, payload: dict[str, Any]) -> "GaussianEnvelopeSpec":
		if not isinstance(payload, dict):
			raise TypeError("GaussianEnvelopeSpec.rebuild() expects a dict payload.")
		return cls(
			center_fs = float(payload["center_fs"]),
			sigma_fs = float(payload["sigma_fs"]),
			amplitude = float(payload.get("amplitude", 1.0)),
			label = payload.get("label"),
			metadata = dict(payload.get("metadata") or {}),
		)


@dataclass(frozen = True)
class SechEnvelopeSpec(EnvelopeSpec):
	"""Hyperbolic secant envelope.

	envelope(t) = amplitude / cosh[(t-center)/width]
	"""

	width_fs: float
	center_fs: float = .0
	amplitude: float = 1.0
	label: str | None = None
	metadata: dict[str, Any] | None = None

	def __post_init__(self) -> None:
		if float(self.width_fs) <= 0.0:
			raise ValueError("width_fs must be positive.")

	def value(self, t_fs) -> np.ndarray:
		t = np.asarray(t_fs, dtype = float)
		x = (t - float(self.center_fs)) / float(self.width_fs)
		return float(self.amplitude) / np.cosh(x)

	def shifted(self, shift_fs: float) -> "SechEnvelopeSpec":
		return replace(self, center_fs = float(self.center_fs) + float(shift_fs))

	@property
	def normalization_rate_candidates_fs_inv(self) -> tuple[float, ...]:
		return (1.0 / float(self.width_fs),)

	def to_dict(self) -> dict[str, Any]:
		return {
			"class": self.__class__.__name__,
			"center_fs": float(self.center_fs),
			"width_fs": float(self.width_fs),
			"amplitude": float(self.amplitude),
			"label": self.label,
			"metadata": _metadata_copy(self.metadata),
			"expression": "amplitude / cosh[(t-center)/width]",
		}

	@classmethod
	def rebuild(cls, payload: dict[str, Any]) -> "SechEnvelopeSpec":
		if not isinstance(payload, dict):
			raise TypeError("SechEnvelopeSpec.rebuild() expects a dict payload.")
		return cls(
			center_fs = float(payload["center_fs"]),
			width_fs = float(payload["width_fs"]),
			amplitude = float(payload.get("amplitude", 1.0)),
			label = payload.get("label"),
			metadata = dict(payload.get("metadata") or {}),
		)


@dataclass(frozen = True)
class ConstantEnvelopeSpec(EnvelopeSpec):
	"""Constant envelope.

	Mainly useful for testing or CW-like fields under CarrierEnvelopeField.
	"""

	center_fs: float = 0.0
	amplitude: float = 1.0
	label: str | None = None
	metadata: dict[str, Any] | None = None

	def value(self, t_fs) -> np.ndarray:
		t = np.asarray(t_fs, dtype = float)
		return np.full_like(t, fill_value = float(self.amplitude), dtype = float)

	def shifted(self, shift_fs: float) -> "ConstantEnvelopeSpec":
		return replace(self, center_fs = float(self.center_fs) + float(shift_fs))

	def to_dict(self) -> dict[str, Any]:
		return {
			"class": self.__class__.__name__,
			"center_fs": float(self.center_fs),
			"amplitude": float(self.amplitude),
			"label": self.label,
			"metadata": _metadata_copy(self.metadata),
			"expression": "amplitude",
		}

	@classmethod
	def rebuild(cls, payload: dict[str, Any]) -> "ConstantEnvelopeSpec":
		if not isinstance(payload, dict):
			raise TypeError("ConstantEnvelopeSpec.rebuild() expects a dict payload.")
		return cls(
			center_fs = float(payload.get("center_fs", 0.0)),
			amplitude = float(payload.get("amplitude", 1.0)),
			label = payload.get("label"),
			metadata = dict(payload.get("metadata") or {}),
		)


def rebuild_envelope_spec(payload: dict[str, Any]) -> EnvelopeSpec:
	if not isinstance(payload, dict):
		raise TypeError("rebuild_envelope_spec() expects a dict payload.")

	class_name = payload.get("class")
	registry = {
		"GaussianEnvelopeSpec": GaussianEnvelopeSpec,
		"SechEnvelopeSpec": SechEnvelopeSpec,
		"ConstantEnvelopeSpec": ConstantEnvelopeSpec,
	}

	if class_name not in registry:
		raise ValueError(f"Unknown or non-rebuildable envelope spec class: {class_name!r}.")

	return registry[class_name].rebuild(payload)


__all__ = [
	"EnvelopeSpec",
	"GaussianEnvelopeSpec",
	"SechEnvelopeSpec",
	"ConstantEnvelopeSpec",
	"rebuild_envelope_spec",
]
