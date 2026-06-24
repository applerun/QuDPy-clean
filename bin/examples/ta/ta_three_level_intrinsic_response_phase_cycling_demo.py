#!/usr/bin/env python3
"""Three-level TA intrinsic response: phase-cycling comparison demo.

This script is example/scratch-level only. It does not modify the core solver,
DynamicsResult, or IO layer, and it does not use piecewise/dark propagation.

Main goals
----------
1. Compare four pump phase cases:
       pump_phase = 0, pi/2, pi, 3pi/2
   with probe_phase = 0.

2. Compute the physical phase average:
       TA_phase_avg = mean(TA_phase_cases, axis=0)

3. Provide report-friendly figures:
   - final TA phase-average map
   - normalized multi-panel comparison
   - process / preview figures:
       * selected-delay E(t) / P(t)
       * rho preview
       * selected-delay phase-resolved TA spectra
       * selected-delay overlay spectra with mean spectrum

Sign convention
---------------
The spectrum uses the current spectroscopy helper definition:

    absorption = omega * Im[P(omega) / E_probe(omega)]

If plotted signs do not match the usual GSB/SE/ESA intuition, do not change
the sign here; interpret the figure using this convention.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import argparse
import csv
import json
import math
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

if __package__ is None or __package__ == "":
	# Intended path:
	#   QuDPy-clean/bin/examples/ta/ta_three_level_intrinsic_response_phase_cycling_demo.py
	sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from qudpy_sjh.utils.core import (
	NLevelPhysicalParams,
	ParaNormalizer,
	PureDephasingChannel,
	RelaxationChannel,
	run_case,
)
from qudpy_sjh.utils.fields import FieldPhySeries
from qudpy_sjh.utils.fields.carrier_envelope import (
	make_gaussian_carrier_envelope_field,
)
from qudpy_sjh.utils.spectroscopy import (
	lab_frame_absorption_response,
	polarization_C_per_m2,
)

EXAMPLE_NAME = "ta_three_level_intrinsic_response_phase_cycling_demo_no_relaxation"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / EXAMPLE_NAME
HC_EV_NM = 1239.8419843320026


@dataclass(frozen = True)
class DemoConfig:
	example_name: str = EXAMPLE_NAME

	probe_delays_fs: tuple[float, ...] = (
		-300.0, -220.0, -160.0, -110.0, -100.0, -80.0, -60.0, -45.0, -30.0,
		-20.0, -10.0, 0.0, 10.0, 20.0, 30.0, 45.0, 60.0, 80.0, 100.0, 110.0,
		150.0, 220.0, 320.0, 460.0, 650.0, 900.0, 1200.0,
	)
	quick_probe_delays_fs: tuple[float, ...] = (
		-220.0, -100.0, -30.0, 0.0, 30.0, 100.0, 220.0, 650.0, 1200.0,
	)

	preview_delays_fs: tuple[float, ...] = (-100.0, 0.0, 100.0)

	probe_center_fs: float = 0.0

	# Three-level ladder:
	# g<->e near 1.55 eV, e<->f near 1.70 eV.
	basis: tuple[str, ...] = ("g", "e", "f")
	energies_eV: tuple[float, ...] = (0.0, 1.55, 3.25)
	dipole_matrix_D: tuple[tuple[float, ...], ...] = (
		(0.0, 5.0, 0.0),
		(5.0, 0.0, 9.0),
		(0.0, 9.0, 0.0),
	)

	pump_E0_MV_per_cm: float = 0.30
	probe_E0_MV_per_cm: float = 0.008
	pump_laser_energy_eV: float = 1.55
	probe_laser_energy_eV: float = 1.62
	pump_sigma_fs: float = 12.0
	probe_sigma_fs: float = 7.0
	probe_phase_rad: float = 0.0
	pump_phase_cases_rad: tuple[float, ...] = (
		0.0,
		0.5 * math.pi,
		math.pi,
		1.5 * math.pi,
	)

	T1_2_to_1_fs: float = 500.0
	T1_1_to_0_fs: float = 1200.0
	Tphi_1_fs: float = 120.0
	Tphi_2_fs: float = 100.0

	t_start_fs: float = -1500.0
	t_end_fs: float = 1500
	dt_fs: float = 0.2

	number_density_m3: float = 1.0e24
	window: str | None = "hann"
	subtract_mean: bool = True
	rel_threshold: float = 1.0e-6
	zero_padding_factor: int = 4

	plot_energy_range_eV: tuple[float, float] = (1.35, 1.90)
	plot_use_wavelength: bool = False
	cmap: str = "plasma"
	figure_dpi: int = 180

	use_checkpoints: bool = True
	force_run: bool = False


# ---------------------------------------------------------------------
# basic utilities
# ---------------------------------------------------------------------
def _json_safe(value: Any) -> Any:
	if hasattr(value, "to_dict") and callable(value.to_dict):
		return _json_safe(value.to_dict())
	if hasattr(value, "__dataclass_fields__"):
		return _json_safe(asdict(value))
	if isinstance(value, dict):
		return {str(k): _json_safe(v) for k, v in value.items()}
	if isinstance(value, (list, tuple)):
		return [_json_safe(v) for v in value]
	if isinstance(value, np.ndarray):
		return _json_safe(value.tolist())
	if isinstance(value, np.generic):
		return value.item()
	if isinstance(value, complex):
		return {"real": float(value.real), "imag": float(value.imag)}
	if isinstance(value, Path):
		return str(value)
	return value


def write_json(path: Path, payload: dict[str, Any]) -> Path:
	path.parent.mkdir(parents = True, exist_ok = True)
	path.write_text(json.dumps(_json_safe(payload), indent = 2, ensure_ascii = False), encoding = "utf-8")
	return path


def write_csv_rows(path: Path, rows: list[dict[str, Any]]) -> Path:
	if not rows:
		raise ValueError(f"No rows to write: {path}")
	path.parent.mkdir(parents = True, exist_ok = True)
	with path.open("w", newline = "", encoding = "utf-8") as handle:
		writer = csv.DictWriter(handle, fieldnames = list(rows[0].keys()))
		writer.writeheader()
		writer.writerows(rows)
	return path


def write_all_delay_spectra_csv(
	path: Path,
	*,
	delays_fs: np.ndarray,
	energy_eV: np.ndarray,
	phase_stack: np.ndarray,
	phase_avg: np.ndarray,
	phase_rms: np.ndarray,
	phase_avg_unitnorm_diagnostic: np.ndarray,
	phase_labels: list[str],
) -> Path:
	"""保存所有 delay、所有 phase 及所有绘图派生谱为长表 CSV。"""
	delays = np.asarray(delays_fs, dtype = float)
	energy = np.asarray(energy_eV, dtype = float)
	phase_cases = np.asarray(phase_stack, dtype = float)
	expected_shape = (len(phase_labels), delays.size, energy.size)
	if phase_cases.shape != expected_shape:
		raise ValueError(f"phase_stack shape {phase_cases.shape} != {expected_shape}.")
	for name, values in {
		"phase_avg": phase_avg,
		"phase_rms": phase_rms,
		"phase_avg_unitnorm_diagnostic": phase_avg_unitnorm_diagnostic,
	}.items():
		if np.asarray(values).shape != (delays.size, energy.size):
			raise ValueError(f"{name} has incompatible shape {np.asarray(values).shape}.")

	rows: list[dict[str, Any]] = []
	for delay_index, delay_fs in enumerate(delays):
		for energy_index, photon_energy_eV in enumerate(energy):
			row = {
				"delay_index": int(delay_index),
				"delay_fs": float(delay_fs),
				"energy_index": int(energy_index),
				"energy_eV": float(photon_energy_eV),
				"wavelength_nm": float(HC_EV_NM / photon_energy_eV),
				"TA_phase_avg": float(phase_avg[delay_index, energy_index]),
				"TA_phase_rms": float(phase_rms[delay_index, energy_index]),
				"TA_phase_avg_unitnorm_diagnostic": float(
					phase_avg_unitnorm_diagnostic[delay_index, energy_index]
				),
			}
			for phase_index, label in enumerate(phase_labels):
				row[f"TA_phase_{label}"] = float(phase_cases[phase_index, delay_index, energy_index])
			rows.append(row)
	return write_csv_rows(path, rows)


def safe_delay_label(delay_fs: float) -> str:
	value = 0.0 if abs(float(delay_fs)) < 1e-12 else float(delay_fs)
	text = f"{value:.6f}".rstrip("0").rstrip(".")
	if text == "-0":
		text = "0"
	return text.replace("-", "m").replace(".", "p")


def phase_label(phase: float) -> str:
	phase = float(phase) % (2.0 * np.pi)
	if np.isclose(phase, 0.0):
		return "0"
	if np.isclose(phase, 0.5 * np.pi):
		return "pi2"
	if np.isclose(phase, np.pi):
		return "pi"
	if np.isclose(phase, 1.5 * np.pi):
		return "3pi2"
	return f"{phase:.4f}".rstrip("0").rstrip(".").replace(".", "p")


def finite_values(arr: np.ndarray) -> np.ndarray:
	values = np.asarray(arr, dtype = float).ravel()
	return values[np.isfinite(values)]


def map_stats(name: str, arr: np.ndarray, *, reference_maxabs: float | None = None) -> dict[str, Any]:
	values = finite_values(arr)
	if values.size == 0:
		return {
			"name": name,
			"min": np.nan,
			"max": np.nan,
			"mean": np.nan,
			"rms": np.nan,
			"maxabs": np.nan,
			"p95abs": np.nan,
			"p99abs": np.nan,
			"ratio_to_reference_maxabs": np.nan,
		}

	maxabs = float(np.max(np.abs(values)))
	ratio = np.nan
	if reference_maxabs is not None and reference_maxabs > 0:
		ratio = maxabs / float(reference_maxabs)

	return {
		"name": name,
		"min": float(np.min(values)),
		"max": float(np.max(values)),
		"mean": float(np.mean(values)),
		"rms": float(np.sqrt(np.mean(values ** 2))),
		"maxabs": maxabs,
		"p95abs": float(np.percentile(np.abs(values), 95.0)),
		"p99abs": float(np.percentile(np.abs(values), 99.0)),
		"ratio_to_reference_maxabs": ratio,
	}


def robust_vlim(arrays: list[np.ndarray], percentile: float = 99.0) -> float:
	merged = np.concatenate([finite_values(item) for item in arrays])
	merged = merged[np.isfinite(merged)]
	if merged.size == 0:
		return 1.0
	value = float(np.nanpercentile(np.abs(merged), percentile))
	return value if value > 0 else 1.0


def normalize_map_for_diagnostic(arr: np.ndarray, *, scale: str = "p99abs") -> np.ndarray:
	values = finite_values(arr)
	if values.size == 0:
		return np.asarray(arr, dtype = float)

	if scale == "maxabs":
		denom = float(np.max(np.abs(values)))
	elif scale == "p99abs":
		denom = float(np.percentile(np.abs(values), 99.0))
	elif scale == "rms":
		denom = float(np.sqrt(np.mean(values ** 2)))
	else:
		raise ValueError(f"Unknown diagnostic normalization scale: {scale!r}")

	if denom <= 0 or not np.isfinite(denom):
		return np.asarray(arr, dtype = float)

	return np.asarray(arr, dtype = float) / denom


def normalize_for_panel_display(
		values: np.ndarray,
		*,
		scale_mode: str = "p99abs",
) -> tuple[np.ndarray, float, float, float]:
	raw = np.asarray(values, dtype = float)
	finite = raw[np.isfinite(raw)]

	if finite.size == 0:
		return raw, 1.0, np.nan, np.nan

	raw_min = float(np.min(finite))
	raw_max = float(np.max(finite))

	if scale_mode == "maxabs":
		scale = float(np.max(np.abs(finite)))
	elif scale_mode == "p99abs":
		scale = float(np.percentile(np.abs(finite), 99.0))
	elif scale_mode == "rms":
		scale = float(np.sqrt(np.mean(finite ** 2)))
	else:
		raise ValueError(f"Unknown scale_mode: {scale_mode!r}")

	if not np.isfinite(scale) or scale <= 0:
		scale = 1.0

	normalized = raw / scale
	return normalized, scale, raw_min, raw_max


# ---------------------------------------------------------------------
# field / model builders
# ---------------------------------------------------------------------
def make_probe_reference_field(config: DemoConfig):
	return make_gaussian_carrier_envelope_field(
		E0_MV_per_cm = config.probe_E0_MV_per_cm,
		laser_energy_eV = config.probe_laser_energy_eV,
		center_fs = config.probe_center_fs,
		sigma_fs = config.probe_sigma_fs,
		phase_rad = config.probe_phase_rad,
		name = "probe",
		metadata = {
			"role": "probe_only_reference",
			"phase_convention": "phase relative to envelope center",
		},
	)


def make_ta_field(
		config: DemoConfig,
		*,
		delay_fs: float,
		pump_phase_rad: float,
		name: str,
) -> FieldPhySeries:
	delay = float(delay_fs)
	probe_center = float(config.probe_center_fs)
	pump_center = probe_center - delay

	pump = make_gaussian_carrier_envelope_field(
		E0_MV_per_cm = config.pump_E0_MV_per_cm,
		laser_energy_eV = config.pump_laser_energy_eV,
		center_fs = pump_center,
		sigma_fs = config.pump_sigma_fs,
		phase_rad = float(pump_phase_rad),
		name = "pump",
		metadata = {
			"delay_fs": delay,
			"center_fs": pump_center,
			"parent_field": name,
			"phase_convention": "phase relative to envelope center",
		},
	)
	probe = make_gaussian_carrier_envelope_field(
		E0_MV_per_cm = config.probe_E0_MV_per_cm,
		laser_energy_eV = config.probe_laser_energy_eV,
		center_fs = probe_center,
		sigma_fs = config.probe_sigma_fs,
		phase_rad = float(config.probe_phase_rad),
		name = "probe",
		metadata = {
			"delay_fs": delay,
			"center_fs": probe_center,
			"parent_field": name,
			"phase_convention": "phase relative to envelope center",
		},
	)

	return FieldPhySeries(
		fields = (pump, probe),
		sub_field_names = ("pump", "probe"),
		name = name,
		metadata = {
			"experiment": "TA",
			"delay_fs": delay,
			"pump_phase_rad": float(pump_phase_rad),
			"probe_phase_rad": float(config.probe_phase_rad),
			"pump_center_fs": pump_center,
			"probe_center_fs": probe_center,
			"delay_convention": "pump_center_fs = probe_center_fs - delay_fs; positive delay means pump before probe",
			"field_convention": "E(t)=2E0 envelope(t-center) cos[omega*(t-center)+phase]",
		},
	)


def make_physical_params(
		config: DemoConfig,
		field,
		*,
		case_name: str,
		description: str,
) -> NLevelPhysicalParams:
	return NLevelPhysicalParams(
		energies_eV = tuple(float(x) for x in config.energies_eV),
		dipole_matrix_D = tuple(tuple(float(v) for v in row) for row in config.dipole_matrix_D),
		t_start_fs = float(config.t_start_fs),
		t_end_fs = float(config.t_end_fs),
		dt_fs = float(config.dt_fs),
		field = field,
		basis = tuple(config.basis),

		pure_dephasing_channels = (
			PureDephasingChannel(
				name = "pure_dephasing_level_1",
				level = 1,
				Tphi_fs = float(config.Tphi_1_fs),
			),
			PureDephasingChannel(
				name = "pure_dephasing_level_2",
				level = 2,
				Tphi_fs = float(config.Tphi_2_fs),
			),
		),
		solver_mode = "lab_exact",
		input_description = description,
		input_metadata = {
			"example_name": config.example_name,
			"case_name": case_name,
			"response_definition": "S_TA = omega*Im[P_pump_probe/E_probe] - omega*Im[P_probe_only/E_probe]",
			"sign_convention": "absorption = omega * Im[P(omega)/E_probe(omega)]",
			"model_note": "Three-level ladder demo; parameters are diagnostic rather than fitted.",
		},
	)


# ---------------------------------------------------------------------
# running / post-processing
# ---------------------------------------------------------------------
def run_with_checkpoint(
		params: NLevelPhysicalParams,
		*,
		normalizer: ParaNormalizer,
		output_dir: Path,
		case_key: str,
		config: DemoConfig,
):
	if not config.use_checkpoints:
		return run_case(params, normalizer = normalizer)

	ckp = output_dir / "checkpoints" / "carrier_envelope_v2" / f"{case_key}.ckp"
	return run_case(
		params,
		normalizer = normalizer,
		load_ckp = ckp,
		save_ckp = ckp,
		force_run = bool(config.force_run),
	)


def response_from_result(result, probe_field, config: DemoConfig) -> dict[str, np.ndarray]:
	physical = result.physical_params
	if physical is None:
		raise ValueError("DynamicsResult.physical_params is required.")

	t_fs = np.asarray(result.times_fs, dtype = float)
	e_probe = np.asarray(probe_field(t_fs), dtype = float)
	p_t = polarization_C_per_m2(
		result.density_array(),
		physical.dipole_matrix_D,
		float(config.number_density_m3),
	)

	return lab_frame_absorption_response(
		time_fs = t_fs,
		polarization_C_per_m2 = p_t,
		field = e_probe,
		window = config.window,
		subtract_mean = bool(config.subtract_mean),
		rel_threshold = float(config.rel_threshold),
		zero_padding_factor = int(config.zero_padding_factor),
		return_intermediates = True,
	)


def assert_same_axis(name: str, reference: np.ndarray, current: np.ndarray) -> None:
	if reference.shape != current.shape:
		raise ValueError(f"{name} axis shape mismatch: {reference.shape} vs {current.shape}")
	diff = float(np.max(np.abs(reference - current))) if reference.size else 0.0
	if diff > 1e-12:
		raise ValueError(f"{name} axis mismatch. max_abs_diff={diff:.6e}")


def compute_map(
		config: DemoConfig,
		*,
		delays_fs: tuple[float, ...],
		output_dir: Path,
		normalizer: ParaNormalizer,
		probe_field,
		probe_response: dict[str, np.ndarray],
		pump_phase_rad: float,
		map_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	energy_ref = np.asarray(probe_response["energy_eV"], dtype = float)
	omega_ref = np.asarray(probe_response["omega_fs_inv"], dtype = float)
	s_probe = np.asarray(probe_response["absorption"], dtype = float)

	rows = []
	for delay_fs in delays_fs:
		print(f"[{map_name}] delay={delay_fs:g} fs, pump_phase={pump_phase_rad:.6g}")
		field = make_ta_field(
			config,
			delay_fs = float(delay_fs),
			pump_phase_rad = float(pump_phase_rad),
			name = f"{map_name}_delay_{safe_delay_label(delay_fs)}_fs",
		)
		params = make_physical_params(
			config,
			field,
			case_name = f"{map_name}_delay_{safe_delay_label(delay_fs)}_fs",
			description = f"TA pump+probe case for {map_name}, delay={delay_fs:g} fs.",
		)
		result = run_with_checkpoint(
			params,
			normalizer = normalizer,
			output_dir = output_dir,
			case_key = f"{map_name}_delay_{safe_delay_label(delay_fs)}_fs",
			config = config,
		)
		response = response_from_result(result, probe_field, config)

		energy = np.asarray(response["energy_eV"], dtype = float)
		omega = np.asarray(response["omega_fs_inv"], dtype = float)
		assert_same_axis("energy_eV", energy_ref, energy)
		assert_same_axis("omega_fs_inv", omega_ref, omega)

		s_pump_probe = np.asarray(response["absorption"], dtype = float)
		rows.append(s_pump_probe - s_probe)

	return energy_ref, omega_ref, np.vstack(rows)


def run_single_trace_case(
		config: DemoConfig,
		*,
		delay_fs: float,
		pump_phase_rad: float,
		output_dir: Path,
		normalizer: ParaNormalizer,
):
	field = make_ta_field(
		config,
		delay_fs = delay_fs,
		pump_phase_rad = pump_phase_rad,
		name = f"trace_delay_{safe_delay_label(delay_fs)}_phase_{phase_label(pump_phase_rad)}",
	)
	params = make_physical_params(
		config,
		field,
		case_name = f"trace_delay_{safe_delay_label(delay_fs)}_phase_{phase_label(pump_phase_rad)}",
		description = f"Trace/preview case at delay={delay_fs:g} fs, phase={pump_phase_rad:.6g}.",
	)
	result = run_with_checkpoint(
		params,
		normalizer = normalizer,
		output_dir = output_dir,
		case_key = f"trace_delay_{safe_delay_label(delay_fs)}_phase_{phase_label(pump_phase_rad)}",
		config = config,
	)

	physical = result.physical_params
	if physical is None:
		raise ValueError("DynamicsResult.physical_params is required.")

	t_fs = np.asarray(result.times_fs, dtype = float)
	density = result.density_array()
	p_t = polarization_C_per_m2(
		density,
		physical.dipole_matrix_D,
		float(config.number_density_m3),
	)
	e_t = np.asarray(physical.field(t_fs), dtype = float)

	return {
		"result": result,
		"times_fs": t_fs,
		"density": density,
		"polarization_t": p_t,
		"field_t": e_t,
	}


# ---------------------------------------------------------------------
# plotting helpers
# ---------------------------------------------------------------------
def to_plot_x(energy_eV: np.ndarray, *, use_wavelength: bool) -> tuple[np.ndarray, str]:
	if use_wavelength:
		return HC_EV_NM / energy_eV, "Probe wavelength (nm)"
	return energy_eV, "Probe photon energy (eV)"


def apply_plot_energy_mask(
		x: np.ndarray,
		values: np.ndarray,
		config: DemoConfig,
) -> tuple[np.ndarray, np.ndarray]:
	if config.plot_energy_range_eV is None:
		return x, values

	e_min, e_max = config.plot_energy_range_eV

	if config.plot_use_wavelength:
		x_min = HC_EV_NM / e_max
		x_max = HC_EV_NM / e_min
		mask = (x >= min(x_min, x_max)) & (x <= max(x_min, x_max))
	else:
		mask = (x >= e_min) & (x <= e_max)

	if not np.any(mask):
		return x, values

	return x[mask], values[:, mask]


def prepare_plot_arrays(
		energy_eV: np.ndarray,
		values: np.ndarray,
		config: DemoConfig,
) -> tuple[np.ndarray, np.ndarray, str]:
	x, xlabel = to_plot_x(energy_eV, use_wavelength = config.plot_use_wavelength)
	plot_values = np.asarray(values, dtype = float)

	if config.plot_use_wavelength:
		order = np.argsort(x)
		x = x[order]
		plot_values = plot_values[:, order]

	x, plot_values = apply_plot_energy_mask(x, plot_values, config)
	return x, plot_values, xlabel


def displayed_energy_map_values(
		energy_eV: np.ndarray,
		values: np.ndarray,
		config: DemoConfig,
) -> np.ndarray:
	_x, plot_values, _xlabel = prepare_plot_arrays(energy_eV, values, config)
	return plot_values


def plot_one_map(
		*,
		path: Path,
		title: str,
		energy_eV: np.ndarray,
		delays_fs: np.ndarray,
		values: np.ndarray,
		config: DemoConfig,
		vlim: float | None,
) -> Path:
	"""Plot one TA map.

	If vlim is None, normalize the displayed energy range by its own p99abs,
	use a fixed color scale [-1, 1], and annotate raw min/max/scale.
	"""

	fig, ax = plt.subplots(figsize = (6.6, 4.7))
	x, plot_values, xlabel = prepare_plot_arrays(energy_eV, values, config)

	if vlim is None:
		plot_values, scale_used, raw_min, raw_max = normalize_for_panel_display(
			plot_values,
			scale_mode = "p99abs",
		)
		local_vmin = -1.0
		local_vmax = 1.0
		cbar_label = "Normalized S_TA (displayed map / p99abs)"
		ax.text(
			0.02,
			0.98,
			(
				f"min={raw_min:.2e}\n"
				f"max={raw_max:.2e}\n"
				f"scale={scale_used:.2e}"
			),
			transform = ax.transAxes,
			ha = "left",
			va = "top",
			fontsize = 8,
			bbox = {
				"facecolor": "white",
				"alpha": 0.72,
				"edgecolor": "none",
				"pad": 2,
			},
		)
		title = f"{title}\nnormalized display, colorbar fixed to [-1, 1]"
	else:
		local_vmin = -float(vlim)
		local_vmax = float(vlim)
		cbar_label = "S_TA (arb., current sign)"

	mesh = ax.pcolormesh(
		x,
		delays_fs,
		plot_values,
		shading = "auto",
		cmap = config.cmap,
		vmin = local_vmin,
		vmax = local_vmax,
	)

	ax.set_title(title)
	ax.set_xlabel(xlabel)
	ax.set_ylabel("Pump-probe delay (fs)")

	cbar = fig.colorbar(mesh, ax = ax)
	cbar.set_label(cbar_label)
	if vlim is None:
		cbar.set_ticks([-1.0, -0.5, 0.0, 0.5, 1.0])

	fig.tight_layout()

	path.parent.mkdir(parents = True, exist_ok = True)
	fig.savefig(path, dpi = int(config.figure_dpi))
	plt.close(fig)
	return path


def plot_compare(
		*,
		path: Path,
		energy_eV: np.ndarray,
		delays_fs: np.ndarray,
		maps: list[tuple[str, np.ndarray]],
		config: DemoConfig,
		shared_vlim: float | None,
) -> Path:
	"""Plot 2x3 comparison.

	If shared_vlim is None, each panel is normalized by its displayed p99abs and
	plotted with the same fixed colorbar [-1, 1]. The raw min/max/scale are
	annotated in the upper-left corner of each panel.
	"""

	fig, axes = plt.subplots(
		2,
		3,
		figsize = (16.0, 8.4),
		sharey = True,
		constrained_layout = False,
	)
	axes_flat = axes.ravel()

	last_mesh = None
	for ax, (title, values) in zip(axes_flat, maps):
		x, plot_values, xlabel = prepare_plot_arrays(energy_eV, values, config)

		if shared_vlim is None:
			plot_values, scale_used, raw_min, raw_max = normalize_for_panel_display(
				plot_values,
				scale_mode = "p99abs",
			)
			last_mesh = ax.pcolormesh(
				x,
				delays_fs,
				plot_values,
				shading = "auto",
				cmap = config.cmap,
				vmin = -1.0,
				vmax = 1.0,
			)
			ax.text(
				0.02,
				0.98,
				(
					f"min={raw_min:.2e}\n"
					f"max={raw_max:.2e}\n"
					f"scale={scale_used:.2e}"
				),
				transform = ax.transAxes,
				ha = "left",
				va = "top",
				fontsize = 8,
				bbox = {
					"facecolor": "white",
					"alpha": 0.72,
					"edgecolor": "none",
					"pad": 2,
				},
			)
		else:
			last_mesh = ax.pcolormesh(
				x,
				delays_fs,
				plot_values,
				shading = "auto",
				cmap = config.cmap,
				vmin = -float(shared_vlim),
				vmax = float(shared_vlim),
			)

		ax.set_title(title)
		ax.set_xlabel(xlabel)

	for ax in axes[:, 0]:
		ax.set_ylabel("Pump-probe delay (fs)")

	mode = "shared raw scale" if shared_vlim is not None else "per-panel normalized to [-1, 1]"
	fig.suptitle(f"TA phase handling comparison ({mode})", y = 0.965)

	fig.subplots_adjust(
		left = 0.07,
		right = 0.86,
		bottom = 0.08,
		top = 0.91,
		wspace = 0.28,
		hspace = 0.32,
	)

	if last_mesh is not None:
		cax = fig.add_axes([0.89, 0.15, 0.018, 0.68])
		cbar = fig.colorbar(last_mesh, cax = cax)
		if shared_vlim is None:
			cbar.set_label("Normalized S_TA (each panel / displayed p99abs)")
			cbar.set_ticks([-1.0, -0.5, 0.0, 0.5, 1.0])
		else:
			cbar.set_label("S_TA (arb., current sign)")

	path.parent.mkdir(parents = True, exist_ok = True)
	fig.savefig(path, dpi = int(config.figure_dpi))
	plt.close(fig)
	return path


def plot_field_polarization_selected_delays(
		*,
		path: Path,
		delay_to_trace: dict[float, dict[str, Any]],
) -> Path:
	fig, axes = plt.subplots(2, 3, figsize = (15.0, 7.4), sharex = False, constrained_layout = False)

	selected = sorted(delay_to_trace.items(), key = lambda kv: kv[0])
	for col, (delay_fs, payload) in enumerate(selected):
		t = np.asarray(payload["times_fs"], dtype = float)
		e_t = np.asarray(payload["field_t"], dtype = float)
		p_t = np.asarray(payload["polarization_t"], dtype = float)

		ax_e = axes[0, col]
		ax_p = axes[1, col]

		ax_e.plot(t, e_t, linewidth = 1.0)
		ax_e.set_title(f"delay = {delay_fs:g} fs")
		ax_e.set_xlabel("Time (fs)")
		ax_e.set_ylabel("E(t) (MV/cm)")
		ax_e.text(
			0.02,
			0.98,
			f"max|E|={np.max(np.abs(e_t)):.2e}",
			transform = ax_e.transAxes,
			ha = "left",
			va = "top",
			fontsize = 8,
			bbox = {"facecolor": "white", "alpha": 0.72, "edgecolor": "none", "pad": 2},
		)

		ax_p.plot(t, p_t, linewidth = 1.0)
		ax_p.set_xlabel("Time (fs)")
		ax_p.set_ylabel("P(t) (C/m$^2$)")
		ax_p.text(
			0.02,
			0.98,
			f"max|P|={np.max(np.abs(p_t)):.2e}",
			transform = ax_p.transAxes,
			ha = "left",
			va = "top",
			fontsize = 8,
			bbox = {"facecolor": "white", "alpha": 0.72, "edgecolor": "none", "pad": 2},
		)

	fig.suptitle("Figure 1. Full-time field and polarization at selected delays (phase 0)", y = 0.97)
	fig.tight_layout(rect = (0, 0, 1, 0.95))
	path.parent.mkdir(parents = True, exist_ok = True)
	fig.savefig(path, dpi = 180)
	plt.close(fig)
	return path


def plot_rho_preview(
		*,
		path: Path,
		delay_fs: float,
		payload: dict[str, Any],
) -> Path:
	t = np.asarray(payload["times_fs"], dtype = float)
	rho = np.asarray(payload["density"])
	e_t = np.asarray(payload["field_t"], dtype = float)
	p_t = np.asarray(payload["polarization_t"], dtype = float)

	pop_gg = np.real(rho[:, 0, 0])
	pop_ee = np.real(rho[:, 1, 1])
	pop_ff = np.real(rho[:, 2, 2])

	coh_ge = np.abs(rho[:, 0, 1])
	coh_ef = np.abs(rho[:, 1, 2])
	coh_gf = np.abs(rho[:, 0, 2])

	fig, axes = plt.subplots(2, 2, figsize = (12.0, 8.0), constrained_layout = False)

	ax = axes[0, 0]
	ax.plot(t, pop_gg, label = "rho_gg")
	ax.plot(t, pop_ee, label = "rho_ee")
	ax.plot(t, pop_ff, label = "rho_ff")
	ax.set_title("Populations")
	ax.set_xlabel("Time (fs)")
	ax.set_ylabel("Population")
	ax.legend(fontsize = 8)

	ax = axes[0, 1]
	ax.plot(t, coh_ge, label = "|rho_ge|")
	ax.plot(t, coh_ef, label = "|rho_ef|")
	ax.plot(t, coh_gf, label = "|rho_gf|")
	ax.set_title("Coherence magnitudes")
	ax.set_xlabel("Time (fs)")
	ax.set_ylabel("Magnitude")
	ax.legend(fontsize = 8)

	ax = axes[1, 0]
	ax.plot(t, e_t)
	ax.set_title("Total field E(t)")
	ax.set_xlabel("Time (fs)")
	ax.set_ylabel("E(t) (MV/cm)")

	ax = axes[1, 1]
	ax.plot(t, p_t)
	ax.set_title("Macroscopic polarization P(t)")
	ax.set_xlabel("Time (fs)")
	ax.set_ylabel("P(t) (C/m$^2$)")

	fig.suptitle(f"rho preview at delay = {delay_fs:g} fs (phase 0)", y = 0.97)
	fig.tight_layout(rect = (0, 0, 1, 0.95))
	path.parent.mkdir(parents = True, exist_ok = True)
	fig.savefig(path, dpi = 180)
	plt.close(fig)
	return path


def plot_selected_delay_phase_spectra(
		*,
		path: Path,
		delay_fs: float,
		delay_index: int,
		energy_eV: np.ndarray,
		phase_maps: list[np.ndarray],
		phase_names: list[str],
		config: DemoConfig,
) -> Path:
	"""For one selected delay, plot 4 TA difference spectra for 4 phase cases."""

	fig, axes = plt.subplots(2, 2, figsize = (10.5, 7.2), constrained_layout = False)
	axes_flat = axes.ravel()

	y_min_all = []
	y_max_all = []
	prepared_spectra = []

	for phase_map in phase_maps:
		_x, y2d, xlabel = prepare_plot_arrays(
			energy_eV,
			np.asarray(phase_map[delay_index:delay_index + 1, :]),
			config,
		)
		y = np.asarray(y2d[0], dtype = float)
		prepared_spectra.append((_x, y, xlabel))
		finite = y[np.isfinite(y)]
		if finite.size:
			y_min_all.append(float(np.min(finite)))
			y_max_all.append(float(np.max(finite)))

	if y_min_all and y_max_all:
		y_abs = max(abs(min(y_min_all)), abs(max(y_max_all)))
		y_lim = 1.05 * y_abs if y_abs > 0 else 1.0
	else:
		y_lim = 1.0

	phase_title_map = {
		"0": "phase 0",
		"pi2": "phase π/2",
		"pi": "phase π",
		"3pi2": "phase 3π/2",
	}

	for ax, phase_name, (x_plot, y_plot, xlabel) in zip(axes_flat, phase_names, prepared_spectra):
		ax.plot(x_plot, y_plot, linewidth = 1.4)
		ax.axhline(0.0, linewidth = 0.8, linestyle = "--")
		ax.set_title(phase_title_map.get(phase_name, phase_name))
		ax.set_xlabel(xlabel)
		ax.set_ylabel("S_TA (arb., current sign)")
		ax.set_ylim(-y_lim, y_lim)

		finite = y_plot[np.isfinite(y_plot)]
		if finite.size:
			ax.text(
				0.02,
				0.98,
				(
					f"min={np.min(finite):.2e}\n"
					f"max={np.max(finite):.2e}\n"
					f"rms={np.sqrt(np.mean(finite ** 2)):.2e}"
				),
				transform = ax.transAxes,
				ha = "left",
				va = "top",
				fontsize = 8,
				bbox = {
					"facecolor": "white",
					"alpha": 0.72,
					"edgecolor": "none",
					"pad": 2,
				},
			)

	fig.suptitle(f"TA difference spectra at delay = {delay_fs:g} fs", y = 0.97)
	fig.tight_layout(rect = (0, 0, 1, 0.95))

	path.parent.mkdir(parents = True, exist_ok = True)
	fig.savefig(path, dpi = 180)
	plt.close(fig)
	return path


def plot_selected_delay_phase_spectra_overlay(
		*,
		path: Path,
		delay_fs: float,
		delay_index: int,
		energy_eV: np.ndarray,
		phase_maps: list[np.ndarray],
		phase_names: list[str],
		config: DemoConfig,
) -> Path:
	"""Overlay 4 phase spectra and the physical mean spectrum at one delay.

	Left y-axis:
		physical phase-averaged spectrum, black

	Right y-axis:
		four phase-case spectra, red with different line styles
	"""

	fig, ax_avg = plt.subplots(figsize = (8.6, 5.4))
	ax_phase = ax_avg.twinx()

	spectra = []
	x_ref = None
	xlabel_ref = None

	phase_title_map = {
		"0": "phase 0",
		"pi2": "phase π/2",
		"pi": "phase π",
		"3pi2": "phase 3π/2",
	}
	line_styles = {
		"0": "-",
		"pi2": "--",
		"pi": "-.",
		"3pi2": ":",
	}

	for phase_map, phase_name in zip(phase_maps, phase_names):
		x_plot, y2d, xlabel = prepare_plot_arrays(
			energy_eV,
			np.asarray(phase_map[delay_index:delay_index + 1, :]),
			config,
		)
		y_plot = np.asarray(y2d[0], dtype = float)

		if x_ref is None:
			x_ref = x_plot
			xlabel_ref = xlabel

		spectra.append(y_plot)

		ax_phase.plot(
			x_plot,
			y_plot,
			linestyle = line_styles.get(phase_name, "-"),
			linewidth = 1.2,
			color = "red",
			alpha = 0.75,
			label = phase_title_map.get(phase_name, phase_name),
		)

	avg_spectrum = np.mean(np.vstack(spectra), axis = 0)

	ax_avg.plot(
		x_ref,
		avg_spectrum,
		linestyle = "-",
		linewidth = 2.2,
		color = "black",
		label = "phase average",
	)

	ax_avg.axhline(0.0, linewidth = 0.8, linestyle = "--", color = "black", alpha = 0.5)
	ax_phase.axhline(0.0, linewidth = 0.8, linestyle = "--", color = "red", alpha = 0.35)

	ax_avg.set_title(f"Overlay TA spectra at delay = {delay_fs:g} fs")
	ax_avg.set_xlabel(xlabel_ref)
	ax_avg.set_ylabel("Phase-averaged S_TA", color = "black")
	ax_phase.set_ylabel("Single-phase S_TA", color = "red")

	ax_avg.tick_params(axis = "y", labelcolor = "black")
	ax_phase.tick_params(axis = "y", labelcolor = "red")

	avg_finite = avg_spectrum[np.isfinite(avg_spectrum)]
	phase_stack = np.vstack(spectra)
	phase_finite = phase_stack[np.isfinite(phase_stack)]

	if avg_finite.size:
		avg_abs = float(np.max(np.abs(avg_finite)))
		if avg_abs > 0:
			ax_avg.set_ylim(-1.08 * avg_abs, 1.08 * avg_abs)

	if phase_finite.size:
		phase_abs = float(np.max(np.abs(phase_finite)))
		if phase_abs > 0:
			ax_phase.set_ylim(-1.08 * phase_abs, 1.08 * phase_abs)

	if avg_finite.size and phase_finite.size:
		ax_avg.text(
			0.02,
			0.98,
			(
				f"avg max={np.max(avg_finite):.2e}\n"
				f"avg min={np.min(avg_finite):.2e}\n"
				f"avg rms={np.sqrt(np.mean(avg_finite ** 2)):.2e}\n"
				f"phase maxabs={np.max(np.abs(phase_finite)):.2e}"
			),
			transform = ax_avg.transAxes,
			ha = "left",
			va = "top",
			fontsize = 8,
			bbox = {
				"facecolor": "white",
				"alpha": 0.72,
				"edgecolor": "none",
				"pad": 2,
			},
		)

	lines_avg, labels_avg = ax_avg.get_legend_handles_labels()
	lines_phase, labels_phase = ax_phase.get_legend_handles_labels()
	ax_avg.legend(
		lines_avg + lines_phase,
		labels_avg + labels_phase,
		fontsize = 8,
		loc = "lower right",
	)

	fig.tight_layout()
	path.parent.mkdir(parents = True, exist_ok = True)
	fig.savefig(path, dpi = 180)
	plt.close(fig)
	return path


def plot_selected_delay_mean_spectrum(
		*,
		path: Path,
		delay_fs: float,
		delay_index: int,
		energy_eV: np.ndarray,
		phase_maps: list[np.ndarray],
		config: DemoConfig,
) -> Path:
	"""Plot only the physical phase-averaged TA spectrum at one selected delay."""

	spectra = []
	x_ref = None
	xlabel_ref = None

	for phase_map in phase_maps:
		x_plot, y2d, xlabel = prepare_plot_arrays(
			energy_eV,
			np.asarray(phase_map[delay_index:delay_index + 1, :]),
			config,
		)
		y_plot = np.asarray(y2d[0], dtype = float)

		if x_ref is None:
			x_ref = x_plot
			xlabel_ref = xlabel

		spectra.append(y_plot)

	avg_spectrum = np.mean(np.vstack(spectra), axis = 0)

	fig, ax = plt.subplots(figsize = (7.4, 4.8))
	ax.plot(
		x_ref,
		avg_spectrum,
		linewidth = 2.0,
		color = "black",
		label = "phase average",
	)
	ax.axhline(0.0, linewidth = 0.8, linestyle = "--", color = "black", alpha = 0.5)

	ax.set_title(f"Phase-averaged TA spectrum at delay = {delay_fs:g} fs")
	ax.set_xlabel(xlabel_ref)
	ax.set_ylabel("Phase-averaged S_TA")

	finite = avg_spectrum[np.isfinite(avg_spectrum)]
	if finite.size:
		y_abs = float(np.max(np.abs(finite)))
		if y_abs > 0:
			ax.set_ylim(-1.08 * y_abs, 1.08 * y_abs)

		ax.text(
			0.02,
			0.98,
			(
				f"min={np.min(finite):.2e}\n"
				f"max={np.max(finite):.2e}\n"
				f"rms={np.sqrt(np.mean(finite ** 2)):.2e}"
			),
			transform = ax.transAxes,
			ha = "left",
			va = "top",
			fontsize = 8,
			bbox = {
				"facecolor": "white",
				"alpha": 0.72,
				"edgecolor": "none",
				"pad": 2,
			},
		)

	ax.legend(fontsize = 8, loc = "best")
	fig.tight_layout()

	path.parent.mkdir(parents = True, exist_ok = True)
	fig.savefig(path, dpi = 180)
	plt.close(fig)
	return path


def plot_phase_cycling_suppression_stats(
		*,
		path: Path,
		phase_names: list[str],
		phase_maps: list[np.ndarray],
		ta_phase_avg: np.ndarray,
		config: DemoConfig,
		energy_eV: np.ndarray,
) -> Path:
	labels = []
	p99_values = []
	rms_values = []

	for phase_name, phase_map in zip(phase_names, phase_maps):
		displayed = displayed_energy_map_values(energy_eV, phase_map, config)
		stats = map_stats(f"phase_{phase_name}", displayed)
		labels.append(phase_name)
		p99_values.append(stats["p99abs"])
		rms_values.append(stats["rms"])

	avg_displayed = displayed_energy_map_values(energy_eV, ta_phase_avg, config)
	avg_stats = map_stats("phase_avg", avg_displayed)
	labels.append("avg")
	p99_values.append(avg_stats["p99abs"])
	rms_values.append(avg_stats["rms"])

	x = np.arange(len(labels))
	width = 0.36

	fig, ax = plt.subplots(figsize = (8.8, 5.2))
	ax.bar(x - width / 2, p99_values, width = width, label = "p99abs")
	ax.bar(x + width / 2, rms_values, width = width, label = "RMS")
	ax.set_yscale("log")
	ax.set_xticks(x)
	ax.set_xticklabels(labels)
	ax.set_ylabel("Amplitude (log scale)")
	ax.set_title("Figure 2. Phase-cycling suppression statistics")
	ax.legend()

	for xi, y in zip(x - width / 2, p99_values):
		ax.text(xi, y, f"{y:.1e}", ha = "center", va = "bottom", fontsize = 7, rotation = 90)
	for xi, y in zip(x + width / 2, rms_values):
		ax.text(xi, y, f"{y:.1e}", ha = "center", va = "bottom", fontsize = 7, rotation = 90)

	fig.tight_layout()
	path.parent.mkdir(parents = True, exist_ok = True)
	fig.savefig(path, dpi = 180)
	plt.close(fig)
	return path


# ---------------------------------------------------------------------
# main workflow
# ---------------------------------------------------------------------
def run_demo(config: DemoConfig, *, output_dir: Path, quick: bool = False) -> dict[str, Any]:
	output_dir.mkdir(parents = True, exist_ok = True)
	data_dir = output_dir / "data"
	plot_dir = output_dir / "figures" / "plot"
	preview_dir = output_dir / "figures" / "preview"
	legacy_dir = output_dir / "figures" / "legacy"

	delays = tuple(float(x) for x in (config.quick_probe_delays_fs if quick else config.probe_delays_fs))
	delays_array = np.asarray(delays, dtype = float)
	normalizer = ParaNormalizer(auto_scale = True)

	probe_field = make_probe_reference_field(config)
	probe_params = make_physical_params(
		config,
		probe_field,
		case_name = "probe_only",
		description = "Probe-only reference shared by all TA maps.",
	)
	probe_result = run_with_checkpoint(
		probe_params,
		normalizer = normalizer,
		output_dir = output_dir,
		case_key = "probe_only",
		config = config,
	)
	probe_response = response_from_result(probe_result, probe_field, config)

	energy_eV = np.asarray(probe_response["energy_eV"], dtype = float)
	omega_fs_inv = np.asarray(probe_response["omega_fs_inv"], dtype = float)

	phase_maps: list[np.ndarray] = []
	phase_names: list[str] = []

	for phase in config.pump_phase_cases_rad:
		label = phase_label(phase)
		map_name = f"phase_{label}"
		_, _, ta_map = compute_map(
			config,
			delays_fs = delays,
			output_dir = output_dir,
			normalizer = normalizer,
			probe_field = probe_field,
			probe_response = probe_response,
			pump_phase_rad = float(phase),
			map_name = map_name,
		)
		phase_maps.append(ta_map)
		phase_names.append(label)

	phase_stack = np.stack(phase_maps, axis = 0)
	ta_phase_avg = np.mean(phase_stack, axis = 0)
	ta_phase_rms = np.sqrt(np.mean(phase_stack ** 2, axis = 0))

	# Diagnostic only: normalize each phase case before averaging.
	phase_stack_unitnorm = np.stack(
		[normalize_map_for_diagnostic(item, scale = "p99abs") for item in phase_maps],
		axis = 0,
	)
	ta_phase_avg_unitnorm_diagnostic = np.mean(phase_stack_unitnorm, axis = 0)

	shared_vlim = robust_vlim(phase_maps + [ta_phase_avg], percentile = 99.0)

	phase_mean_maxabs = float(
		np.mean([map_stats(f"phase_{label}", ta_map)["maxabs"] for label, ta_map in zip(phase_names, phase_maps)])
	)

	stats_rows = []
	for label, ta_map in zip(phase_names, phase_maps):
		stats_rows.append(
			map_stats(f"TA_phase_{label}_full_energy", ta_map, reference_maxabs = phase_mean_maxabs)
		)
		stats_rows.append(
			map_stats(
				f"TA_phase_{label}_displayed_energy",
				displayed_energy_map_values(energy_eV, ta_map, config),
				reference_maxabs = phase_mean_maxabs,
			)
		)

	stats_rows.append(
		map_stats("TA_phase_avg_raw_full_energy", ta_phase_avg, reference_maxabs = phase_mean_maxabs)
	)
	stats_rows.append(
		map_stats(
			"TA_phase_avg_raw_displayed_energy",
			displayed_energy_map_values(energy_eV, ta_phase_avg, config),
			reference_maxabs = phase_mean_maxabs,
		)
	)
	stats_rows.append(
		map_stats(
			"TA_phase_rms_displayed_energy",
			displayed_energy_map_values(energy_eV, ta_phase_rms, config),
			reference_maxabs = phase_mean_maxabs,
		)
	)
	stats_rows.append(
		map_stats(
			"TA_phase_avg_unitnorm_diagnostic_displayed_energy",
			displayed_energy_map_values(energy_eV, ta_phase_avg_unitnorm_diagnostic, config),
		)
	)

	data_dir.mkdir(parents = True, exist_ok = True)
	stats_csv = write_csv_rows(data_dir / "map_stats.csv", stats_rows)
	stats_json = write_json(data_dir / "map_stats.json", {"map_stats": stats_rows})

	print("\nMap statistics:")
	for row in stats_rows:
		print(
			f"{row['name']:48s} "
			f"maxabs={row['maxabs']:.3e} "
			f"p99abs={row['p99abs']:.3e} "
			f"rms={row['rms']:.3e} "
			f"ratio={row['ratio_to_reference_maxabs']:.3e}"
		)
	print()

	figure_paths: dict[str, str] = {}

	phase_titles = {
		"0": "pump phase 0",
		"pi2": "pump phase π/2",
		"pi": "pump phase π",
		"3pi2": "pump phase 3π/2",
	}
	phase_filenames = {
		"0": "ta_phase_case_0.png",
		"pi2": "ta_phase_case_pi2.png",
		"pi": "ta_phase_case_pi.png",
		"3pi2": "ta_phase_case_3pi2.png",
	}

	# -----------------------------------------------------------------
	# plot/: only final report-style maps
	# -----------------------------------------------------------------
	figure_paths["phase_avg_autoscale"] = str(
		plot_one_map(
			path = plot_dir / "ta_phase_avg_autoscale.png",
			title = "TA map: 4-step pump-phase average",
			energy_eV = energy_eV,
			delays_fs = delays_array,
			values = ta_phase_avg,
			config = config,
			vlim = None,
		)
	)

	compare_maps_norm = [
		("phase 0", phase_maps[0]),
		("phase π/2", phase_maps[1]),
		("phase π", phase_maps[2]),
		("phase 3π/2", phase_maps[3]),
		("phase average", ta_phase_avg),
		("phase-case RMS", ta_phase_rms),
	]
	figure_paths["compare_autoscale"] = str(
		plot_compare(
			path = plot_dir / "ta_phase_cycling_compare_autoscale.png",
			energy_eV = energy_eV,
			delays_fs = delays_array,
			maps = compare_maps_norm,
			config = config,
			shared_vlim = None,
		)
	)

	# -----------------------------------------------------------------
	# legacy/: supporting or historical outputs
	# -----------------------------------------------------------------
	for label, ta_map in zip(phase_names, phase_maps):
		filename = phase_filenames.get(label, f"ta_phase_case_{label}.png")
		figure_paths[f"legacy_phase_case_{label}"] = str(
			plot_one_map(
				path = legacy_dir / filename,
				title = f"TA map: {phase_titles.get(label, label)}",
				energy_eV = energy_eV,
				delays_fs = delays_array,
				values = ta_map,
				config = config,
				vlim = shared_vlim,
			)
		)

	figure_paths["legacy_phase_avg_shared"] = str(
		plot_one_map(
			path = legacy_dir / "ta_phase_avg.png",
			title = "TA map: 4-step pump-phase average, shared raw scale",
			energy_eV = energy_eV,
			delays_fs = delays_array,
			values = ta_phase_avg,
			config = config,
			vlim = shared_vlim,
		)
	)
	figure_paths["legacy_phase_avg_unitnorm_diagnostic"] = str(
		plot_one_map(
			path = legacy_dir / "ta_phase_avg_unitnorm_diagnostic.png",
			title = "Diagnostic only: mean of p99-normalized phase maps",
			energy_eV = energy_eV,
			delays_fs = delays_array,
			values = ta_phase_avg_unitnorm_diagnostic,
			config = config,
			vlim = None,
		)
	)

	compare_maps_shared = [
		("phase 0", phase_maps[0]),
		("phase π/2", phase_maps[1]),
		("phase π", phase_maps[2]),
		("phase 3π/2", phase_maps[3]),
		("phase average", ta_phase_avg),
		("phase-case RMS", ta_phase_rms),
	]
	figure_paths["legacy_compare_shared"] = str(
		plot_compare(
			path = legacy_dir / "ta_phase_cycling_compare.png",
			energy_eV = energy_eV,
			delays_fs = delays_array,
			maps = compare_maps_shared,
			config = config,
			shared_vlim = shared_vlim,
		)
	)

	# -----------------------------------------------------------------
	# preview/: process / selected-delay diagnostics
	# -----------------------------------------------------------------
	delay_to_trace: dict[float, dict[str, Any]] = {}
	for target_delay in config.preview_delays_fs:
		trace_payload = run_single_trace_case(
			config,
			delay_fs = float(target_delay),
			pump_phase_rad = 0.0,
			output_dir = output_dir,
			normalizer = normalizer,
		)
		delay_to_trace[float(target_delay)] = trace_payload

		figure_paths[f"preview_rho_delay_{safe_delay_label(target_delay)}"] = str(
			plot_rho_preview(
				path = preview_dir / f"rho_preview_delay_{safe_delay_label(target_delay)}.png",
				delay_fs = float(target_delay),
				payload = trace_payload,
			)
		)

	figure_paths["preview_figure_1_field_polarization"] = str(
		plot_field_polarization_selected_delays(
			path = preview_dir / "figure_1_field_polarization_selected_delays.png",
			delay_to_trace = delay_to_trace,
		)
	)

	figure_paths["preview_figure_2_suppression_stats"] = str(
		plot_phase_cycling_suppression_stats(
			path = preview_dir / "figure_2_phase_cycling_suppression_stats.png",
			phase_names = phase_names,
			phase_maps = phase_maps,
			ta_phase_avg = ta_phase_avg,
			config = config,
			energy_eV = energy_eV,
		)
	)

	def find_nearest_delay_index(target_fs: float) -> int:
		return int(np.argmin(np.abs(delays_array - float(target_fs))))

	for target_delay in config.preview_delays_fs:
		idx = find_nearest_delay_index(target_delay)
		actual_delay = float(delays_array[idx])

		figure_paths[f"preview_diff_spectra_delay_{safe_delay_label(actual_delay)}"] = str(
			plot_selected_delay_phase_spectra(
				path = preview_dir / f"ta_diff_spectra_delay_{safe_delay_label(actual_delay)}.png",
				delay_fs = actual_delay,
				delay_index = idx,
				energy_eV = energy_eV,
				phase_maps = phase_maps,
				phase_names = phase_names,
				config = config,
			)
		)

		figure_paths[f"preview_diff_spectra_overlay_delay_{safe_delay_label(actual_delay)}"] = str(
			plot_selected_delay_phase_spectra_overlay(
				path = preview_dir / f"ta_diff_spectra_overlay_delay_{safe_delay_label(actual_delay)}.png",
				delay_fs = actual_delay,
				delay_index = idx,
				energy_eV = energy_eV,
				phase_maps = phase_maps,
				phase_names = phase_names,
				config = config,
			)
		)
		figure_paths[f"preview_mean_spectrum_delay_{safe_delay_label(actual_delay)}"] = str(
			plot_selected_delay_mean_spectrum(
				path = preview_dir / f"ta_phase_avg_spectrum_delay_{safe_delay_label(actual_delay)}.png",
				delay_fs = actual_delay,
				delay_index = idx,
				energy_eV = energy_eV,
				phase_maps = phase_maps,
				config = config,
			)
		)
	# -----------------------------------------------------------------
	# data save
	# -----------------------------------------------------------------
	all_delay_spectra_csv = write_all_delay_spectra_csv(
		data_dir / "ta_all_delay_spectra.csv",
		delays_fs = delays_array,
		energy_eV = energy_eV,
		phase_stack = phase_stack,
		phase_avg = ta_phase_avg,
		phase_rms = ta_phase_rms,
		phase_avg_unitnorm_diagnostic = ta_phase_avg_unitnorm_diagnostic,
		phase_labels = phase_names,
	)
	npz_path = data_dir / "ta_phase_cycling_comparison.npz"
	np.savez_compressed(
		npz_path,
		delays_fs = delays_array,
		energy_eV = energy_eV,
		omega_fs_inv = omega_fs_inv,
		wavelength_nm = HC_EV_NM / energy_eV,
		TA_phase_cases = phase_stack,
		TA_phase_avg = ta_phase_avg,
		TA_phase_rms = ta_phase_rms,
		TA_phase_avg_unitnorm_diagnostic = ta_phase_avg_unitnorm_diagnostic,
		phase_values_rad = np.asarray(config.pump_phase_cases_rad, dtype = float),
		phase_labels = np.asarray(phase_names, dtype = str),
	)

	meta = {
		"example_name": config.example_name,
		"quick": bool(quick),
		"output_dir": output_dir,
		"data_npz": npz_path,
		"all_delay_spectra_csv": all_delay_spectra_csv,
		"stats_csv": stats_csv,
		"stats_json": stats_json,
		"figures": figure_paths,
		"phase_cases": {
			"pump_phase_rad": list(config.pump_phase_cases_rad),
			"probe_phase_rad": config.probe_phase_rad,
			"physical_phase_average": "unweighted arithmetic mean across four pump phases",
			"diagnostic_unitnorm_average": (
				"Each phase map is divided by its own p99 abs value before averaging. "
				"This is not a physical phase-cycling result."
			),
			"phase_case_rms": (
				"sqrt(mean(TA_phase_case^2, axis=phase)); diagnostic map showing pre-average magnitude."
			),
		},
		"field_convention": "E(t)=2E0 envelope(t-center) cos[omega*(t-center)+phase]",
		"model_parameters": {
			"basis": config.basis,
			"energies_eV": config.energies_eV,
			"dipole_matrix_D": config.dipole_matrix_D,
		},
		"field_parameters": {
			"pump_E0_MV_per_cm": config.pump_E0_MV_per_cm,
			"probe_E0_MV_per_cm": config.probe_E0_MV_per_cm,
			"pump_laser_energy_eV": config.pump_laser_energy_eV,
			"probe_laser_energy_eV": config.probe_laser_energy_eV,
			"pump_sigma_fs": config.pump_sigma_fs,
			"probe_sigma_fs": config.probe_sigma_fs,
			"probe_center_fs": config.probe_center_fs,
		},
		"time_grid": {
			"t_start_fs": config.t_start_fs,
			"t_end_fs": config.t_end_fs,
			"dt_fs": config.dt_fs,
			"n_time_points": int(np.asarray(probe_result.times_fs).size),
		},
		"spectroscopy": {
			"definition": "absorption = omega * Im[P(omega)/E_probe(omega)]",
			"TA_definition": "S_TA = S_pump_probe - S_probe_only",
			"number_density_m3": config.number_density_m3,
			"window": config.window,
			"subtract_mean": config.subtract_mean,
			"rel_threshold": config.rel_threshold,
			"zero_padding_factor": config.zero_padding_factor,
		},
	}
	meta_path = write_json(output_dir / "meta.json", meta)

	print("TA phase-cycling comparison finished.")
	print(f"n delays          : {len(delays)}")
	print(f"energy points     : {energy_eV.size}")
	print(f"output directory  : {output_dir}")
	print(f"data npz          : {npz_path}")
	print(f"all-delay spectra : {all_delay_spectra_csv}")
	print(f"stats csv         : {stats_csv}")
	print(f"metadata          : {meta_path}")
	print(f"final phase avg   : {figure_paths['phase_avg_autoscale']}")
	print(f"compare autoscale : {figure_paths['compare_autoscale']}")

	return {
		"output_dir": str(output_dir),
		"data_npz": str(npz_path),
		"all_delay_spectra_csv": str(all_delay_spectra_csv),
		"stats_csv": str(stats_csv),
		"stats_json": str(stats_json),
		"meta_json": str(meta_path),
		"figures": figure_paths,
	}


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description = __doc__)
	parser.add_argument(
		"--output-dir",
		type = Path,
		default = DEFAULT_OUTPUT_DIR,
		help = "Output directory for checkpoints, data, metadata, and figures.",
	)
	parser.add_argument(
		"--force-run",
		action = "store_true",
		help = "Ignore existing checkpoints and rerun all simulations.",
	)
	parser.add_argument(
		"--no-checkpoints",
		action = "store_true",
		help = "Run without checkpoint load/save.",
	)
	parser.add_argument(
		"--quick",
		action = "store_true",
		help = "Use a smaller delay grid for smoke testing.",
	)
	parser.add_argument(
		"--wavelength",
		action = "store_true",
		help = "Plot wavelength instead of photon energy on the x-axis.",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	base_config = DemoConfig()
	config = DemoConfig(
		**{
			**asdict(base_config),
			"force_run": bool(args.force_run),
			"use_checkpoints": not bool(args.no_checkpoints),
			"plot_use_wavelength": bool(args.wavelength),
		}
	)
	run_demo(config, output_dir = Path(args.output_dir), quick = bool(args.quick))


if __name__ == "__main__":
	main()
