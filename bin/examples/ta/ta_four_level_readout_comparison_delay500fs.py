#!/usr/bin/env python3
"""比较 500 fs delay 下三种四能级 TA readout 的试验脚本。

本脚本刻意只做一个 delay，并沿用项目的 lab-frame Lindblad 密度矩阵传播。
核心比较对象是：时间门控的共线 readout、长通滤波的共线 readout，以及
4 x 4 phase cycling 取得的 ordinary TA probe channel。

求解器目前不传播 Maxwell 输出场。因此这里把
``E_out(t) = E_in(t) + source_field_scale * P(t)`` 定义为薄样品的等价
readout field；比例系数是显式的模型参数，并不代表完整的传播模型。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import argparse
import json
import math
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

if __package__ is None or __package__ == "":
	# 脚本可直接从仓库根目录外运行。
	sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from qudpy_sjh.utils.core import (
	NLevelPhysicalParams,
	ParaNormalizer,
	PureDephasingChannel,
	RelaxationChannel,
	run_case,
)
from qudpy_sjh.utils.fields import FieldPhySeries
from qudpy_sjh.utils.fields.carrier_envelope import make_gaussian_carrier_envelope_field
from qudpy_sjh.utils.spectroscopy import polarization_C_per_m2


EXAMPLE_NAME = "ta_four_level_readout_comparison_delay500fs"
DEFAULT_OUTPUT_DIR = (
	Path(__file__).resolve().parents[3]
	/ "outputs/examples/ta/four_level_readout_comparison_delay500fs"
)
HC_EV_NM = 1239.8419843320026
EPS = 1.0e-30


@dataclass(frozen = True)
class Config:
	delay_fs: float = 500.0
	probe_center_fs: float = 0.0
	t_start_fs: float = -900.0
	t_end_fs: float = 4500.0
	dt_fs: float = 0.05
	gate_start_fs: float = -200.0
	gate_end_fs: float = 1500.0
	lambda_cut_nm: float = 550.0

	pump_wavelength_nm: float = 400.0
	pump_fwhm_fs: float = 40.0
	pump_amplitude_MV_per_cm: float = 5.0
	probe_wavelength_nm: float = 670.0
	probe_fwhm_fs: float = 6.0
	probe_amplitude_MV_per_cm: float = 1.0

	# 该比例把宏观极化映射到等价读出场，单位为 (MV/cm)/(C/m^2)。
	source_field_scale: float = 1.0e5
	number_density_m3: float = 1.0e24
	phase_grid_rad: tuple[float, ...] = (0.0, 0.5 * math.pi, math.pi, 1.5 * math.pi)
	use_checkpoints: bool = True
	force_run: bool = False
	figure_dpi: int = 180


def write_json(path: Path, payload: dict[str, Any]) -> Path:
	"""以可读 JSON 保存元数据。"""
	def convert(value: Any) -> Any:
		if hasattr(value, "__dataclass_fields__"):
			return convert(asdict(value))
		if isinstance(value, dict):
			return {str(k): convert(v) for k, v in value.items()}
		if isinstance(value, (tuple, list)):
			return [convert(v) for v in value]
		if isinstance(value, np.ndarray):
			return convert(value.tolist())
		if isinstance(value, np.generic):
			return value.item()
		if isinstance(value, complex):
			return {"real": value.real, "imag": value.imag}
		if isinstance(value, Path):
			return str(value)
		return value

	path.parent.mkdir(parents = True, exist_ok = True)
	path.write_text(json.dumps(convert(payload), indent = 2, ensure_ascii = False), encoding = "utf-8")
	return path


def fwhm_to_sigma_fs(fwhm_fs: float) -> float:
	return float(fwhm_fs) / (2.0 * math.sqrt(2.0 * math.log(2.0)))


def build_four_level_model() -> dict[str, Any]:
	"""构造四能级 Hamiltonian、Hermitian dipole 与 Lindblad 配置。"""
	energies_eV = (0.0, 1.72, 3.10, 5.05)
	mu_matrix_D = (
		(0.0, 0.6, 1.0, 0.0),
		(0.6, 0.0, 0.0, 0.0),
		(1.0, 0.0, 0.0, 0.8),
		(0.0, 0.0, 0.8, 0.0),
	)
	transitions = ((0, 2), (0, 1), (2, 3))
	transition_info = {
		f"{lower}-{upper}": {
			"energy_eV": float(energies_eV[upper] - energies_eV[lower]),
			"wavelength_nm": float(HC_EV_NM / (energies_eV[upper] - energies_eV[lower])),
		}
		for lower, upper in transitions
	}
	return {
		"energies_eV": energies_eV,
		"hamiltonian_eV": tuple(
			tuple(energy if row == column else 0.0 for column in range(len(energies_eV)))
			for row, energy in enumerate(energies_eV)
		),
		"mu_matrix_D": mu_matrix_D,
		"collapse_ops": {
			"relaxation": ((2, 1, 500.0), (1, 0, 5000.0)),
			"pure_dephasing": ((0, 0.0), (1, 0.0038), (2, 0.0), (3, 0.0020)),
		},
		"transition_info": transition_info,
	}


def make_field(config: Config, *, pump_phase: float | None, probe_phase: float, name: str):
	"""构造 probe-only 或 pump+probe 驱动场。"""
	pump_center = config.probe_center_fs - config.delay_fs
	probe = make_gaussian_carrier_envelope_field(
		E0_MV_per_cm = config.probe_amplitude_MV_per_cm,
		laser_energy_eV = HC_EV_NM / config.probe_wavelength_nm,
		center_fs = config.probe_center_fs,
		sigma_fs = fwhm_to_sigma_fs(config.probe_fwhm_fs),
		phase_rad = probe_phase,
		name = "probe",
	)
	if pump_phase is None:
		return probe
	pump = make_gaussian_carrier_envelope_field(
		E0_MV_per_cm = config.pump_amplitude_MV_per_cm,
		laser_energy_eV = HC_EV_NM / config.pump_wavelength_nm,
		center_fs = pump_center,
		sigma_fs = fwhm_to_sigma_fs(config.pump_fwhm_fs),
		phase_rad = pump_phase,
		name = "pump",
	)
	return FieldPhySeries(fields = (pump, probe), sub_field_names = ("pump", "probe"), name = name)


def make_params(config: Config, field, *, case_name: str) -> NLevelPhysicalParams:
	"""将四能级模型和驱动场装配为项目的物理参数对象。"""
	model = build_four_level_model()
	return NLevelPhysicalParams(
		energies_eV = model["energies_eV"],
		dipole_matrix_D = model["mu_matrix_D"],
		t_start_fs = config.t_start_fs,
		t_end_fs = config.t_end_fs,
		dt_fs = config.dt_fs,
		field = field,
		basis = ("0", "1", "2", "3"),
		relaxation_channels = tuple(
			RelaxationChannel(
				name = f"relaxation_{src}_to_{dst}", from_level = src, to_level = dst,
				T1_fs = t1, rate_fs_inv = 1.0 / t1,
			)
			for src, dst, t1 in model["collapse_ops"]["relaxation"]
		),
		pure_dephasing_channels = tuple(
			PureDephasingChannel(
				name = f"pure_dephasing_level_{level}", level = level,
				Tphi_fs = 1.0 / gamma, rate_fs_inv = gamma,
			)
			for level, gamma in model["collapse_ops"]["pure_dephasing"] if gamma > 0.0
		),
		solver_mode = "lab_exact",
		input_description = f"Four-level TA readout comparison: {case_name}",
		input_metadata = {"example_name": EXAMPLE_NAME, "case_name": case_name},
	)


def run_trace(config: Config, *, field, key: str, output_dir: Path, normalizer: ParaNormalizer) -> dict[str, np.ndarray]:
	"""传播密度矩阵，并产生输入场、极化和等价输出场。"""
	checkpoint = output_dir / "checkpoints" / f"{key}.ckp"
	result = run_case(
		make_params(config, field, case_name = key), normalizer = normalizer,
		load_ckp = checkpoint if config.use_checkpoints else None,
		save_ckp = checkpoint if config.use_checkpoints else None,
		force_run = config.force_run,
	)
	physical = result.physical_params
	if physical is None:
		raise ValueError("DynamicsResult.physical_params is required.")
	time_fs = np.asarray(result.times_fs, dtype = float)
	e_in = np.asarray(field(time_fs), dtype = float)
	polarization = polarization_C_per_m2(result.density_array(), physical.dipole_matrix_D, config.number_density_m3)
	e_out = e_in + config.source_field_scale * np.real(polarization)
	return {"time_fs": time_fs, "e_in": e_in, "polarization": polarization, "e_out": e_out}


def fft_field(time_fs: np.ndarray, field: np.ndarray) -> dict[str, np.ndarray]:
	"""对 field 本身做 FFT；绝不对 field**2 做 FFT。"""
	dt_fs = float(np.median(np.diff(time_fs)))
	if not np.allclose(np.diff(time_fs), dt_fs, rtol = 1e-7, atol = 1e-10):
		raise ValueError("FFT readout requires a uniform time axis.")
	n_fft = 1 << int(np.ceil(np.log2(time_fs.size)))
	field_w = np.fft.fft(field, n = n_fft)
	freq_fs_inv = np.fft.fftfreq(n_fft, d = dt_fs)
	positive = freq_fs_inv > 0.0
	freq = freq_fs_inv[positive]
	return {
		"frequency_fs_inv": freq,
		"energy_eV": freq / ParaNormalizer.EV_TO_FS_INV,
		"wavelength_nm": HC_EV_NM / (freq / ParaNormalizer.EV_TO_FS_INV),
		"field_w": field_w[positive],
	}


def delta_absorbance(on_w: np.ndarray, off_w: np.ndarray) -> tuple[np.ndarray, bool]:
	"""由输出场强度比给出 DeltaA，并标记近零 reference。"""
	i_on = np.abs(on_w) ** 2
	i_off = np.abs(off_w) ** 2
	near_zero = bool(np.min(i_off) < 1.0e-12 * np.max(i_off))
	return -np.log10((i_on + EPS) / (i_off + EPS)), near_zero


def signal_metrics(signal: np.ndarray) -> dict[str, float]:
	values = np.asarray(signal, dtype = float)
	return {
		"max_abs": float(np.max(np.abs(values))),
		"rms": float(np.sqrt(np.mean(values ** 2))),
		"peak_to_peak": float(np.ptp(values)),
	}


def plot_band(ax, wavelength_nm: np.ndarray, values: np.ndarray, *, title: str, ylabel: str, cut_nm: float | None = None) -> None:
	mask = (wavelength_nm >= 350.0) & (wavelength_nm <= 850.0)
	order = np.argsort(wavelength_nm[mask])
	ax.plot(wavelength_nm[mask][order], values[mask][order])
	if cut_nm is not None:
		ax.axvline(cut_nm, color = "k", linestyle = "--", linewidth = 1.0)
	ax.set(title = title, xlabel = "Wavelength (nm)", ylabel = ylabel, xlim = (350.0, 850.0))
	ax.grid(alpha = 0.25)


def make_figures(output_dir: Path, config: Config, *, pump_input: np.ndarray, probe_input: np.ndarray, on: dict[str, np.ndarray], off: dict[str, np.ndarray], gate: np.ndarray, spec: dict[str, Any]) -> dict[str, str]:
	"""生成所要求的四张诊断图。"""
	fig_dir = output_dir / "figures"
	fig_dir.mkdir(parents = True, exist_ok = True)
	t = on["time_fs"]
	paths: dict[str, str] = {}

	fig, axes = plt.subplots(2, 2, figsize = (12, 7), sharex = True)
	axes[0, 0].plot(t, pump_input, label = "pump")
	axes[0, 0].plot(t, probe_input, label = "probe")
	axes[0, 0].axvline(-500, color = "C0", ls = "--"); axes[0, 0].axvline(0, color = "C1", ls = "--")
	axes[0, 0].legend(); axes[0, 0].set_title("Input pump and probe")
	axes[0, 1].plot(t, gate); axes[0, 1].axvline(-200, color = "k", ls = "--")
	axes[0, 1].set_title("Case A time gate [-200, 1500] fs")
	axes[1, 0].plot(t, on["e_out"], label = "E_out_on")
	axes[1, 0].plot(t, gate * on["e_out"], label = "g E_out_on")
	axes[1, 0].legend(); axes[1, 0].set_title("Pump-on equivalent output field")
	axes[1, 1].plot(t, off["e_out"], label = "E_out_off")
	axes[1, 1].plot(t, gate * off["e_out"], label = "g E_out_off")
	axes[1, 1].legend(); axes[1, 1].set_title("Probe-only equivalent output field")
	for ax in axes.flat: ax.set(xlim = (-900, 1800), xlabel = "Time (fs)"); ax.grid(alpha = 0.25)
	fig.suptitle("Figure 1. Time-domain overview")
	fig.tight_layout(); path = fig_dir / "figure_1_time_domain_overview.png"; fig.savefig(path, dpi = config.figure_dpi); plt.close(fig); paths["figure_1"] = str(path)

	w = spec["wavelength_nm"]
	fig, axes = plt.subplots(2, 3, figsize = (16, 8))
	plot_band(axes[0, 0], w, spec["I_on_g"], title = "Case A gated spectra", ylabel = "Intensity")
	plot_band(axes[0, 0], w, spec["I_off_g"], title = "Case A gated spectra", ylabel = "Intensity")
	plot_band(axes[0, 1], w, spec["A_g"], title = "Case A: collinear + ideal time gate", ylabel = "DeltaA")
	plot_band(axes[0, 2], w, spec["I_on"], title = "Case B full spectra", ylabel = "Intensity")
	plot_band(axes[0, 2], w, spec["I_off"], title = "Case B full spectra", ylabel = "Intensity")
	plot_band(axes[1, 0], w, spec["H_lp"], title = "Ideal long-pass mask", ylabel = "Transmission", cut_nm = config.lambda_cut_nm)
	plot_band(axes[1, 1], w, spec["I_on_lp"], title = "Case B filtered spectra", ylabel = "Intensity")
	plot_band(axes[1, 1], w, spec["I_off_lp"], title = "Case B filtered spectra", ylabel = "Intensity")
	plot_band(axes[1, 2], w, spec["A_lp"], title = "Case B: collinear + ideal long-pass filter", ylabel = "DeltaA")
	fig.suptitle("Figure 2. Collinear readout spectra"); fig.tight_layout(); path = fig_dir / "figure_2_collinear_readout_spectra.png"; fig.savefig(path, dpi = config.figure_dpi); plt.close(fig); paths["figure_2"] = str(path)

	fig, axes = plt.subplots(2, 2, figsize = (12, 8))
	for ax, key, title in ((axes[0, 0], "A_g", "Case A"), (axes[0, 1], "A_lp", "Case B"), (axes[1, 0], "A_pc", "Case C: phase-cycled probe-channel reference")):
		plot_band(ax, w, spec[key], title = title, ylabel = "DeltaA")
	for key, label in (("A_g", "Case A gate"), ("A_lp", "Case B long-pass"), ("A_pc", "Case C phase-cycled")):
		mask = (w >= 350) & (w <= 850); order = np.argsort(w[mask]); axes[1, 1].plot(w[mask][order], spec[key][mask][order], label = label)
	axes[1, 1].set(title = "Three-readout comparison", xlabel = "Wavelength (nm)", ylabel = "DeltaA", xlim = (350, 850)); axes[1, 1].legend(); axes[1, 1].grid(alpha = 0.25)
	fig.suptitle("Figure 3. Three-readout comparison"); fig.tight_layout(); path = fig_dir / "figure_3_three_readout_comparison.png"; fig.savefig(path, dpi = config.figure_dpi); plt.close(fig); paths["figure_3"] = str(path)

	pump_gate = fft_field(t, gate * pump_input); pump_full = fft_field(t, pump_input)
	fig, axes = plt.subplots(2, 2, figsize = (12, 8))
	plot_band(axes[0, 0], pump_full["wavelength_nm"], np.abs(pump_full["field_w"]) ** 2, title = "Pump-only before/after time gate", ylabel = "Intensity")
	plot_band(axes[0, 0], pump_gate["wavelength_nm"], np.abs(pump_gate["field_w"]) ** 2, title = "Pump-only before/after time gate", ylabel = "Intensity")
	plot_band(axes[0, 1], w, spec["I_pump"], title = "Pump-only before/after long-pass", ylabel = "Intensity")
	plot_band(axes[0, 1], w, spec["I_pump_lp"], title = "Pump-only before/after long-pass", ylabel = "Intensity")
	axes[1, 0].bar(["on", "off"], [np.sum((gate * on["e_out"]) ** 2) / np.sum(on["e_out"] ** 2), np.sum((gate * off["e_out"]) ** 2) / np.sum(off["e_out"] ** 2)])
	axes[1, 0].set(title = "Case A retained field energy", ylabel = "Fraction")
	metric_names = ("max_abs", "rms", "peak_to_peak")
	x = np.arange(3); width = 0.24
	for idx, (label, metrics) in enumerate(spec["metrics"].items()): axes[1, 1].bar(x + (idx - 1) * width, [metrics[name] for name in metric_names], width, label = label)
	axes[1, 1].set_xticks(x, metric_names); axes[1, 1].set_yscale("log"); axes[1, 1].legend(); axes[1, 1].set_title("Readout artifact metrics")
	fig.suptitle("Figure 4. Diagnostics"); fig.tight_layout(); path = fig_dir / "figure_4_diagnostics.png"; fig.savefig(path, dpi = config.figure_dpi); plt.close(fig); paths["figure_4"] = str(path)
	return paths


def run_demo(config: Config, *, output_dir: Path) -> dict[str, Any]:
	"""运行所有 readout，并保存谱、图和可复现实验元数据。"""
	output_dir.mkdir(parents = True, exist_ok = True)
	normalizer = ParaNormalizer(auto_scale = True)
	zero_phase_probe = make_field(config, pump_phase = None, probe_phase = 0.0, name = "probe_only")
	off = run_trace(config, field = zero_phase_probe, key = "probe_only", output_dir = output_dir, normalizer = normalizer)
	on_field = make_field(config, pump_phase = 0.0, probe_phase = 0.0, name = "pump_probe")
	on = run_trace(config, field = on_field, key = "pump_probe_phase_0", output_dir = output_dir, normalizer = normalizer)
	t = on["time_fs"]
	if not np.array_equal(t, off["time_fs"]): raise ValueError("Pump-on and probe-only time axes differ.")
	pump_input = np.asarray(on_field.fields[0](t), dtype = float)
	probe_input = np.asarray(on_field.fields[1](t), dtype = float)
	gate = ((t >= config.gate_start_fs) & (t <= config.gate_end_fs)).astype(float)

	on_g = fft_field(t, gate * on["e_out"]); off_g = fft_field(t, gate * off["e_out"])
	on_full = fft_field(t, on["e_out"]); off_full = fft_field(t, off["e_out"]); pump_full = fft_field(t, pump_input)
	for spectrum in (off_g, on_full, off_full, pump_full):
		if not np.allclose(on_g["frequency_fs_inv"], spectrum["frequency_fs_inv"]): raise ValueError("FFT axes differ.")
	h_lp = (on_full["wavelength_nm"] > config.lambda_cut_nm).astype(float)
	a_g, warn_g = delta_absorbance(on_g["field_w"], off_g["field_w"])
	a_lp, warn_lp = delta_absorbance(h_lp * on_full["field_w"], h_lp * off_full["field_w"])

	# 对 pump 相位做不变平均，对 probe 相位取一阶 Fourier 分量。
	pc_on = np.zeros_like(on_full["field_w"], dtype = complex)
	pc_off = np.zeros_like(on_full["field_w"], dtype = complex)
	for pump_phase in config.phase_grid_rad:
		for probe_phase in config.phase_grid_rad:
			weight = np.exp(-1j * probe_phase) / (len(config.phase_grid_rad) ** 2)
			field = make_field(config, pump_phase = pump_phase, probe_phase = probe_phase, name = "phase_cycled")
			trace = run_trace(config, field = field, key = f"pc_pump_{pump_phase:.6f}_probe_{probe_phase:.6f}", output_dir = output_dir, normalizer = normalizer)
			pc_on += weight * fft_field(t, trace["e_out"])["field_w"]
			probe = make_field(config, pump_phase = None, probe_phase = probe_phase, name = "probe_phase_reference")
			trace_off = run_trace(config, field = probe, key = f"pc_off_probe_{probe_phase:.6f}", output_dir = output_dir, normalizer = normalizer)
			pc_off += weight * fft_field(t, trace_off["e_out"])["field_w"]
	a_pc, warn_pc = delta_absorbance(pc_on, pc_off)
	metrics = {"Case A": signal_metrics(a_g), "Case B": signal_metrics(a_lp), "Case C": signal_metrics(a_pc)}
	spec = {
		"wavelength_nm": on_full["wavelength_nm"], "energy_eV": on_full["energy_eV"], "I_on_g": np.abs(on_g["field_w"]) ** 2, "I_off_g": np.abs(off_g["field_w"]) ** 2,
		"I_on": np.abs(on_full["field_w"]) ** 2, "I_off": np.abs(off_full["field_w"]) ** 2, "H_lp": h_lp,
		"I_on_lp": np.abs(h_lp * on_full["field_w"]) ** 2, "I_off_lp": np.abs(h_lp * off_full["field_w"]) ** 2,
		"I_pump": np.abs(pump_full["field_w"]) ** 2, "I_pump_lp": np.abs(h_lp * pump_full["field_w"]) ** 2,
		"A_g": a_g, "A_lp": a_lp, "A_pc": a_pc, "metrics": metrics,
	}
	figures = make_figures(output_dir, config, pump_input = pump_input, probe_input = probe_input, on = on, off = off, gate = gate, spec = spec)
	data_path = output_dir / "data" / "readout_comparison.npz"; data_path.parent.mkdir(parents = True, exist_ok = True)
	np.savez_compressed(data_path, time_fs = t, gate = gate, e_out_on = on["e_out"], e_out_off = off["e_out"], pc_on_w = pc_on, pc_off_w = pc_off, **{k: v for k, v in spec.items() if k != "metrics"})
	model = build_four_level_model()
	meta = {"config": config, "model": model, "figures": figures, "data_npz": data_path, "metrics": metrics, "warnings": {"case_a_near_zero_denominator": warn_g, "case_b_near_zero_denominator": warn_lp, "case_c_near_zero_denominator": warn_pc}, "readout_definition": "E_out = E_in + source_field_scale * Re[P(t)]"}
	meta_path = write_json(output_dir / "meta.json", meta)
	print(f"delay = {config.delay_fs:g} fs; pump center = {-config.delay_fs:g} fs; probe center = {config.probe_center_fs:g} fs")
	print(f"gate open = {config.gate_start_fs:g} fs; readout window = [{config.gate_start_fs:g}, {config.gate_end_fs:g}] fs")
	print(f"four-level energies (eV) = {model['energies_eV']}; transition wavelengths (nm) = {model['transition_info']}")
	print("allowed dipoles (D): mu_02=1.0, mu_01=0.6, mu_23=0.8")
	print("population relaxation rates (fs^-1): k21=0.002, k10=0.0002; pure dephasing gamma: [0, 0.0038, 0, 0.0020]")
	print("effective T2 targets: rho_02≈1000 fs, rho_01=500 fs, rho_23=500 fs")
	print(f"pump/probe: {config.pump_fwhm_fs:g} fs at {config.pump_wavelength_nm:g} nm; {config.probe_fwhm_fs:g} fs at {config.probe_wavelength_nm:g} nm")
	print(f"long-pass cutoff wavelength = {config.lambda_cut_nm:g} nm")
	for name, values in metrics.items(): print(f"{name}: max(abs)={values['max_abs']:.3e}, RMS={values['rms']:.3e}, peak-to-peak={values['peak_to_peak']:.3e}")
	if any((warn_g, warn_lp, warn_pc)): print("WARNING: denominator spectrum contains near-zero intensity bins.")
	return {"data": str(data_path), "metadata": str(meta_path), "figures": figures}


def main() -> None:
	parser = argparse.ArgumentParser(description = __doc__)
	parser.add_argument("--output-dir", type = Path, default = DEFAULT_OUTPUT_DIR)
	parser.add_argument("--force-run", action = "store_true")
	parser.add_argument("--no-checkpoints", action = "store_true")
	args = parser.parse_args()
	config = Config(force_run = args.force_run, use_checkpoints = not args.no_checkpoints)
	run_demo(config, output_dir = args.output_dir)


if __name__ == "__main__":
	main()
