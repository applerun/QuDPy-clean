"""2DES field helpers."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from ..field_series import FieldPhySeries, iter_scan_params
from ..lab_fields import _metadata_copy
from .basic_fields import make_default_gaussian_carrier_field


@dataclass(frozen=True)
class TwoDESField(FieldPhySeries):
	"""2DES pump1-pump2-probe field."""

	pump_tau_fs: float = 0.0
	probe_delay_fs: float = 0.0

	@property
	def probe_delay(self) -> float:
		return float(self.probe_delay_fs)

	@property
	def pump_tau(self) -> float:
		return float(self.pump_tau_fs)

	def to_dict(self) -> dict[str, Any]:
		payload = super().to_dict()
		payload["probe_delay_fs"] = float(self.probe_delay_fs)
		payload["pump_tau_fs"] = float(self.pump_tau_fs)
		return payload


def make_twodes_gaussian_field(
		*,
		pump_tau_fs: float,
		probe_delay_fs: float,
		pump1_E0_MV_per_cm: float,
		pump2_E0_MV_per_cm: float,
		probe_E0_MV_per_cm: float,
		pump1_laser_energy_eV: float,
		pump2_laser_energy_eV: float | None = None,
		probe_laser_energy_eV: float | None = None,
		pump1_center_fs: float = 0.0,
		pump1_sigma_fs: float = 10.0,
		pump2_sigma_fs: float | None = None,
		probe_sigma_fs: float | None = None,
		pump1_phase_rad: float = 0.0,
		pump2_phase_rad: float = 0.0,
		probe_phase_rad: float = 0.0,
		name: str = "twodes_gaussian_field",
		metadata: dict[str, Any] | None = None,
) -> TwoDESField:
	pump2_laser_energy = pump1_laser_energy_eV if pump2_laser_energy_eV is None else pump2_laser_energy_eV
	probe_laser_energy = pump1_laser_energy_eV if probe_laser_energy_eV is None else probe_laser_energy_eV
	pump2_sigma = pump1_sigma_fs if pump2_sigma_fs is None else pump2_sigma_fs
	probe_sigma = pump1_sigma_fs if probe_sigma_fs is None else probe_sigma_fs

	pump2_center_fs = float(pump1_center_fs) + float(pump_tau_fs)
	probe_center_fs = pump2_center_fs + float(probe_delay_fs)

	pump1 = make_default_gaussian_carrier_field(
		E0_MV_per_cm=float(pump1_E0_MV_per_cm),
		laser_energy_eV=float(pump1_laser_energy_eV),
		pulse_center_fs=float(pump1_center_fs),
		pulse_sigma_fs=float(pump1_sigma_fs),
		phase_rad=float(pump1_phase_rad),
		name="pump1",
		metadata={"role": "pump1", "parent_field": name},
	)
	pump2 = make_default_gaussian_carrier_field(
		E0_MV_per_cm=float(pump2_E0_MV_per_cm),
		laser_energy_eV=float(pump2_laser_energy),
		pulse_center_fs=pump2_center_fs,
		pulse_sigma_fs=float(pump2_sigma),
		phase_rad=float(pump2_phase_rad),
		name="pump2",
		metadata={"role": "pump2", "parent_field": name},
	)
	probe = make_default_gaussian_carrier_field(
		E0_MV_per_cm=float(probe_E0_MV_per_cm),
		laser_energy_eV=float(probe_laser_energy),
		pulse_center_fs=probe_center_fs,
		pulse_sigma_fs=float(probe_sigma),
		phase_rad=float(probe_phase_rad),
		name="probe",
		metadata={"role": "probe", "parent_field": name},
	)

	payload = _metadata_copy(metadata)
	payload.setdefault("experiment", "2DES")
	return TwoDESField(
		fields=(pump1, pump2, probe),
		sub_field_names=("pump1", "pump2", "probe"),
		name=name,
		metadata=payload,
		pump_tau_fs=float(pump_tau_fs),
		probe_delay_fs=float(probe_delay_fs),
	)


def iter_twodes_gaussian_fields(**kwargs) -> Iterator[tuple[dict[str, Any], TwoDESField]]:
	for params in iter_scan_params(kwargs):
		yield params, make_twodes_gaussian_field(**params)


__all__ = [
	"TwoDESField",
	"make_twodes_gaussian_field",
	"iter_twodes_gaussian_fields",
]
