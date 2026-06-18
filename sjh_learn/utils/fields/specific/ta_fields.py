"""Transient absorption pump-probe field helpers."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from ..field_series import FieldPhySeries, iter_scan_params
from ..lab_fields import FieldPhyRoot, _metadata_copy
from .basic_fields import make_default_gaussian_carrier_field


@dataclass(frozen=True)
class TAField(FieldPhySeries):
	"""Transient absorption pump-probe field."""

	probe_delay_fs: float = 0.0

	@property
	def probe_delay(self) -> float:
		return float(self.probe_delay_fs)

	@property
	def pump_tau(self) -> None:
		return None

	def to_dict(self) -> dict[str, Any]:
		payload = super().to_dict()
		payload["probe_delay_fs"] = float(self.probe_delay_fs)
		payload["pump_tau_fs"] = None
		return payload


def make_ta_gaussian_field(
		*,
		probe_delay_fs: float,
		pump_E0_MV_per_cm: float,
		probe_E0_MV_per_cm: float,
		pump_laser_energy_eV: float,
		probe_laser_energy_eV: float | None = None,
		pump_center_fs: float = 0.0,
		pump_sigma_fs: float = 10.0,
		probe_sigma_fs: float | None = None,
		pump_phase_rad: float = 0.0,
		probe_phase_rad: float = 0.0,
		name: str = "ta_gaussian_field",
		metadata: dict[str, Any] | None = None,
) -> TAField:
	probe_laser_energy = pump_laser_energy_eV if probe_laser_energy_eV is None else probe_laser_energy_eV
	probe_sigma = pump_sigma_fs if probe_sigma_fs is None else probe_sigma_fs
	probe_center = float(pump_center_fs) + float(probe_delay_fs)

	pump = make_default_gaussian_carrier_field(
		E0_MV_per_cm=float(pump_E0_MV_per_cm),
		laser_energy_eV=float(pump_laser_energy_eV),
		pulse_center_fs=float(pump_center_fs),
		pulse_sigma_fs=float(pump_sigma_fs),
		phase_rad=float(pump_phase_rad),
		name="pump",
		metadata={"role": "pump", "parent_field": name},
	)
	probe = make_default_gaussian_carrier_field(
		E0_MV_per_cm=float(probe_E0_MV_per_cm),
		laser_energy_eV=float(probe_laser_energy),
		pulse_center_fs=probe_center,
		pulse_sigma_fs=float(probe_sigma),
		phase_rad=float(probe_phase_rad),
		name="probe",
		metadata={"role": "probe", "parent_field": name},
	)

	payload = _metadata_copy(metadata)
	payload.setdefault("experiment", "TA")
	return TAField(
		fields=(pump, probe),
		sub_field_names=("pump", "probe"),
		name=name,
		metadata=payload,
		probe_delay_fs=float(probe_delay_fs),
	)


def make_pump_probe_field_from_templates(
		*,
		pump_template: FieldPhyRoot,
		probe_template: FieldPhyRoot,
		delay_fs: float,
		probe_center_fs: float = 0.0,
		name: str | None = None,
		metadata: dict[str, Any] | None = None,
) -> TAField:
	"""Build a pump-probe field by time-shifting zero-centered templates.

	The probe is fixed at ``probe_center_fs`` and the pump is placed by
	``pump_center_fs = probe_center_fs - delay_fs``.
	"""

	if not isinstance(pump_template, FieldPhyRoot):
		raise TypeError("pump_template must be a FieldPhyRoot instance.")
	if not isinstance(probe_template, FieldPhyRoot):
		raise TypeError("probe_template must be a FieldPhyRoot instance.")

	delay = float(delay_fs)
	probe_center = float(probe_center_fs)
	pump_center = probe_center - delay
	field_name = name or "pump_probe_template_field"

	pump = pump_template.time_shifted(
		pump_center,
		name="pump",
		metadata={"role": "pump", "parent_field": field_name},
	)
	probe = probe_template.time_shifted(
		probe_center,
		name="probe",
		metadata={"role": "probe", "parent_field": field_name},
	)

	payload = _metadata_copy(metadata)
	payload.setdefault("experiment", "TA")
	payload.update(
		{
			"delay_fs": delay,
			"probe_delay_fs": delay,
			"probe_center_fs": probe_center,
			"pump_center_fs": pump_center,
			"center_rule": "pump_center_fs = probe_center_fs - delay_fs",
			"template_convention": "pump/probe templates are expected to be centered at 0 fs.",
		}
	)
	return TAField(
		fields=(pump, probe),
		sub_field_names=("pump", "probe"),
		name=field_name,
		metadata=payload,
		probe_delay_fs=delay,
	)


def make_ta_field_from_templates(
		*,
		pump_template: FieldPhyRoot,
		probe_template: FieldPhyRoot,
		probe_delay_fs: float | None = None,
		delay_fs: float | None = None,
		probe_center_fs: float = 0.0,
		name: str | None = None,
		metadata: dict[str, Any] | None = None,
) -> TAField:
	if delay_fs is None and probe_delay_fs is None:
		raise TypeError("Either delay_fs or probe_delay_fs must be provided.")
	if delay_fs is not None and probe_delay_fs is not None and float(delay_fs) != float(probe_delay_fs):
		raise ValueError("delay_fs and probe_delay_fs must agree when both are provided.")
	delay = float(delay_fs if delay_fs is not None else probe_delay_fs)
	return make_pump_probe_field_from_templates(
		pump_template=pump_template,
		probe_template=probe_template,
		delay_fs=delay,
		probe_center_fs=probe_center_fs,
		name=name,
		metadata=metadata,
	)


def iter_ta_gaussian_fields(**kwargs) -> Iterator[tuple[dict[str, Any], TAField]]:
	for params in iter_scan_params(kwargs):
		yield params, make_ta_gaussian_field(**params)


__all__ = [
	"TAField",
	"make_ta_gaussian_field",
	"make_pump_probe_field_from_templates",
	"make_ta_field_from_templates",
	"iter_ta_gaussian_fields",
]
