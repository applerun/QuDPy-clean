"""Generic lab-frame physical field series containers."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from itertools import product
from typing import Any

import numpy as np

from .lab_fields import FieldPhyRoot, _metadata_copy


def _is_scan_value(value: Any) -> bool:
	if isinstance(value, (str, bytes, dict)):
		return False
	if isinstance(value, FieldPhyRoot):
		return False
	return isinstance(value, Iterable)


def _scan_items(params: dict[str, Any]) -> tuple[list[str], list[list[Any]]]:
	names: list[str] = []
	values: list[list[Any]] = []
	for key, value in params.items():
		if _is_scan_value(value):
			names.append(key)
			values.append(list(value))
	return names, values


def iter_scan_params(params: dict[str, Any]) -> Iterator[dict[str, Any]]:
	names, values = _scan_items(params)
	if not names:
		yield dict(params)
		return

	for combo in product(*values):
		item = dict(params)
		for key, value in zip(names, combo):
			item[key] = value
		yield item


@dataclass(frozen=True)
class FieldPhySeries(FieldPhyRoot):
	"""Linear sum of multiple physical fields."""

	fields: tuple[FieldPhyRoot, ...]
	sub_field_names: tuple[str, ...] | None = None
	name: str = "field_phy_series"
	metadata: dict[str, Any] | None = None

	def __post_init__(self) -> None:
		fields = tuple(self.fields)
		if not fields:
			raise ValueError("FieldPhySeries requires at least one subfield.")
		for field in fields:
			if not isinstance(field, FieldPhyRoot):
				raise TypeError("FieldPhySeries.fields must contain FieldPhyRoot instances.")
		object.__setattr__(self, "fields", fields)

		if self.sub_field_names is None:
			names = []
			for idx, field in enumerate(fields):
				payload = field.to_dict()
				names.append(str(payload.get("name") or getattr(field, "name", f"field_{idx}")))
			object.__setattr__(self, "sub_field_names", tuple(names))
			return

		names = tuple(str(name) for name in self.sub_field_names)
		if len(names) != len(fields):
			raise ValueError("sub_field_names length must match fields length.")
		if len(set(names)) != len(names):
			raise ValueError("sub_field_names must be unique.")
		object.__setattr__(self, "sub_field_names", names)

	@property
	def reference_MV_per_cm(self) -> float | None:
		references: list[float] = []
		for field in self.fields:
			reference = field.reference_MV_per_cm
			if reference is None:
				return None
			references.append(abs(float(reference)))
		total = sum(references)
		return None if total == 0.0 else float(total)

	@property
	def normalization_rate_candidates_fs_inv(self) -> tuple[float, ...]:
		candidates: list[float] = []
		for field in self.fields:
			candidates.extend(field.normalization_rate_candidates_fs_inv)
		return tuple(candidates)

	def physical_E_MV_per_cm(self, t_fs: np.ndarray) -> np.ndarray:
		total = np.zeros_like(t_fs, dtype=float)
		for field in self.fields:
			total = total + field(t_fs)
		return total

	def get_field(self, key: int | str) -> FieldPhyRoot:
		if isinstance(key, int):
			return self.fields[key]
		if isinstance(key, str):
			assert self.sub_field_names is not None
			try:
				idx = self.sub_field_names.index(key)
			except ValueError as exc:
				raise KeyError(f"Unknown sub_field_name: {key!r}") from exc
			return self.fields[idx]
		raise TypeError("key must be int or str.")

	def __getitem__(self, key: int | str) -> FieldPhyRoot:
		return self.get_field(key)

	def __iter__(self) -> Iterator[FieldPhyRoot]:
		return iter(self.fields)

	def __len__(self) -> int:
		return len(self.fields)

	def __repr__(self) -> str:
		assert self.sub_field_names is not None
		items = ", ".join(
			f"{name}={field!r}"
			for name, field in zip(self.sub_field_names, self.fields)
		)
		return f"{self.__class__.__name__}({items})"

	def to_dict(self) -> dict[str, Any]:
		metadata = _metadata_copy(self.metadata)
		rebuildable = all(bool(field.to_dict().get("rebuildable", False)) for field in self.fields)
		return {
			"class": self.__class__.__name__,
			"repr": repr(self),
			"name": self.name,
			"time_unit": self.time_unit,
			"field_unit": self.field_unit,
			"rebuildable": rebuildable,
			"sub_field_names": list(self.sub_field_names or ()),
			"fields": [field.to_dict() for field in self.fields],
			"expression": "E_total(t_fs) = sum_k E_k(t_fs)",
			"description": metadata.get("description"),
			"metadata": metadata,
		}


__all__ = [
	"_is_scan_value",
	"_scan_items",
	"iter_scan_params",
	"FieldPhySeries",
]
