#!/usr/bin/env python3
"""读取 TA 全 delay CSV，并比较 100 fs 谱与 pure-dephasing Lorentz 线型。

该脚本不调用求解器，也不读取 checkpoint。它只读取基础 TA demo 写出的
``data/ta_all_delay_spectra.csv``，取 100 fs 的一条谱线后独立进行 max-abs
归一化。理论参考是无 population relaxation 时的线性 Lorentz 线型：

* 0<->1: ``gamma_01 = gamma_1 / 2``；
* 1<->2: ``gamma_12 = (gamma_1 + gamma_2) / 2``；
* TA 符号参考: ``+L_01/max(L_01) - L_12/max(L_12)``。

最后一项仅用于比较峰位、线宽与符号；它不包含偶极平方或 pump 后人口因子，
也不对理论曲线做拟合。
"""

from __future__ import annotations

import argparse
import csv
import os.path
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


BASE_EXAMPLE_NAME = "ta_three_level_intrinsic_response_phase_cycling_demo_no_relaxation"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / BASE_EXAMPLE_NAME
DEFAULT_SPECTRA_CSV = DEFAULT_OUTPUT_DIR / "data" / "ta_all_delay_spectra.csv"


def normalized_maxabs(values: np.ndarray, *, name: str) -> np.ndarray:
	"""使用自身最大绝对值归一化；零信号直接报错。"""
	array = np.asarray(values, dtype = float)
	if not np.all(np.isfinite(array)):
		raise ValueError(f"{name} contains non-finite values.")
	scale = float(np.max(np.abs(array)))
	if scale == 0.0:
		raise ValueError(f"{name} is identically zero and cannot be normalized.")
	return array / scale


def lorentzian_energy(energy_eV: np.ndarray, *, center_eV: float, gamma_eV: float) -> np.ndarray:
	"""返回峰值为 1/gamma 的能量域 Lorentz 线型。"""
	if gamma_eV <= 0.0:
		raise ValueError("gamma_eV must be positive.")
	delta_eV = np.asarray(energy_eV, dtype = float) - float(center_eV)
	return float(gamma_eV) / (delta_eV ** 2 + float(gamma_eV) ** 2)


def read_delay_spectrum(path: Path, *, delay_fs: float, signal_column: str) -> tuple[np.ndarray, np.ndarray, float]:
	"""严格读取指定 delay 的光谱；delay 不存在时直接报错。"""
	if not path.exists():
		raise FileNotFoundError(
			f"Missing spectra CSV: {path}\n"
			"Run ta_three_level_intrinsic_response_phase_cycling_demo.py first."
		)
	rows: list[tuple[float, float, float]] = []
	with path.open(newline = "", encoding = "utf-8") as handle:
		reader = csv.DictReader(handle)
		if reader.fieldnames is None or signal_column not in reader.fieldnames:
			raise ValueError(f"CSV does not contain signal column {signal_column!r}.")
		for row in reader:
			row_delay = float(row["delay_fs"])
			if np.isclose(row_delay, delay_fs, rtol = 0.0, atol = 1.0e-9):
				rows.append((float(row["energy_eV"]), float(row[signal_column]), row_delay))
	if not rows:
		raise ValueError(f"Requested delay {delay_fs:g} fs is absent from {path}.")
	actual_delays = {item[2] for item in rows}
	if len(actual_delays) != 1:
		raise ValueError(f"Ambiguous matched delays: {sorted(actual_delays)}")
	rows.sort(key = lambda item: item[0])
	energy = np.asarray([item[0] for item in rows], dtype = float)
	signal = np.asarray([item[1] for item in rows], dtype = float)
	if np.any(np.diff(energy) <= 0.0):
		raise ValueError("Selected spectrum must have strictly increasing energy values.")
	return energy, signal, float(rows[0][2])


def restrict_energy_range(energy_eV: np.ndarray, signal: np.ndarray, *, energy_min_eV: float, energy_max_eV: float) -> tuple[np.ndarray, np.ndarray]:
	"""只保留基础 demo 的报告能量窗口，再进行单独归一化。"""
	if energy_min_eV >= energy_max_eV:
		raise ValueError("energy_min_eV must be smaller than energy_max_eV.")
	mask = (energy_eV >= energy_min_eV) & (energy_eV <= energy_max_eV)
	if not np.any(mask):
		raise ValueError("The requested energy window contains no spectrum points.")
	return np.asarray(energy_eV[mask], dtype = float), np.asarray(signal[mask], dtype = float)


def make_theory_reference(
	energy_eV: np.ndarray,
	*,
	energy_01_eV: float,
	energy_12_eV: float,
	gamma_1_fs_inv: float,
	gamma_2_fs_inv: float,
	hbar_eV_fs: float = 0.6582119569,
) -> dict[str, np.ndarray | float]:
	"""按 projector pure dephasing 约定构造两条理论线型。"""
	if gamma_1_fs_inv < 0.0 or gamma_2_fs_inv < 0.0:
		raise ValueError("Pure-dephasing rates must be non-negative.")
	# L_n=sqrt(gamma_n)|n><n| 对 rho_ij 的衰减率为 (gamma_i+gamma_j)/2。
	gamma_01_fs_inv = 0.5 * gamma_1_fs_inv
	gamma_12_fs_inv = 0.5 * (gamma_1_fs_inv + gamma_2_fs_inv)
	gamma_01_eV = hbar_eV_fs * gamma_01_fs_inv
	gamma_12_eV = hbar_eV_fs * gamma_12_fs_inv
	line_01 = lorentzian_energy(energy_eV, center_eV = energy_01_eV, gamma_eV = gamma_01_eV)
	line_12 = lorentzian_energy(energy_eV, center_eV = energy_12_eV, gamma_eV = gamma_12_eV)
	# 两条线各自归一化到单位峰高，再按符号组合；这样不把不同 gamma
	# 导致的 1/gamma 原始峰高差误解释为振幅权重。
	# 正负号与基础 demo 的 absorption = omega*Im[P/E] 约定对齐。
	signed_ta_reference = normalized_maxabs(line_01, name = "Lorentz 0-1") - normalized_maxabs(
		line_12, name = "Lorentz 1-2"
	)
	return {
		"line_01": line_01,
		"line_12": line_12,
		"signed_ta_reference": signed_ta_reference,
		"gamma_01_fs_inv": gamma_01_fs_inv,
		"gamma_12_fs_inv": gamma_12_fs_inv,
		"gamma_01_eV": gamma_01_eV,
		"gamma_12_eV": gamma_12_eV,
	}


def write_comparison_csv(path: Path, *, energy_eV: np.ndarray, signal: np.ndarray, signal_norm: np.ndarray, theory: dict[str, np.ndarray | float]) -> Path:
	"""保存 100 fs 绘图使用的原始数据、归一化数据与理论线型。"""
	path.parent.mkdir(parents = True, exist_ok = True)
	line_01 = normalized_maxabs(np.asarray(theory["line_01"]), name = "Lorentz 0-1")
	line_12 = normalized_maxabs(np.asarray(theory["line_12"]), name = "Lorentz 1-2")
	signed = normalized_maxabs(np.asarray(theory["signed_ta_reference"]), name = "signed Lorentz reference")
	with path.open("w", newline = "", encoding = "utf-8") as handle:
		writer = csv.DictWriter(
			handle,
			fieldnames = (
				"energy_eV", "wavelength_nm", "TA_signal_raw", "TA_signal_normalized",
				"Lorentz_01_normalized", "Lorentz_12_normalized", "signed_TA_Lorentz_normalized",
			),
		)
		writer.writeheader()
		for index, energy in enumerate(energy_eV):
			writer.writerow({
				"energy_eV": float(energy),
				"wavelength_nm": float(1239.8419843320026 / energy),
				"TA_signal_raw": float(signal[index]),
				"TA_signal_normalized": float(signal_norm[index]),
				"Lorentz_01_normalized": float(line_01[index]),
				"Lorentz_12_normalized": float(line_12[index]),
				"signed_TA_Lorentz_normalized": float(signed[index]),
			})
	return path


def plot_comparison(path: Path, *, energy_eV: np.ndarray, signal_norm: np.ndarray, theory: dict[str, np.ndarray | float], delay_fs: float, signal_column: str, dpi: int) -> Path:
	"""绘制数值 100 fs 谱和无弛豫 Lorentz 参考。"""
	line_01 = normalized_maxabs(np.asarray(theory["line_01"]), name = "Lorentz 0-1")
	line_12 = normalized_maxabs(np.asarray(theory["line_12"]), name = "Lorentz 1-2")
	signed = normalized_maxabs(np.asarray(theory["signed_ta_reference"]), name = "signed Lorentz reference")
	fig1, ax_main = plt.subplots(
		1, 1, figsize = (6, 4), sharex = True,

	)
	fig2, ax_lines = plt.subplots(
		1, 1, figsize = (6, 4), sharex = True,

	)
	ax_main.plot(energy_eV, signal_norm, color = "black", linewidth = 2.0, label = f"{signal_column}, normalized")
	ax_main.plot(energy_eV, signed, color = "C3", linewidth = 1.8, linestyle = "--", label = "signed TA Lorentz reference")
	ax_main.axhline(0.0, color = "black", linewidth = 0.8, alpha = 0.5)
	ax_main.set_ylabel("Independently normalized signal")
	ax_main.set_title(f"TA spectrum at {delay_fs:g} fs: numerical result vs. Lorentz reference")
	ax_main.legend(loc = "best")
	ax_main.grid(alpha = 0.25)

	ax_lines.plot(energy_eV, line_01, color = "C0", label = "+ Lorentz 0↔1 (GSB/SE reference)")
	ax_lines.plot(energy_eV, -line_12, color = "C2", label = "- Lorentz 1↔2 (ESA reference)")
	ax_lines.axhline(0.0, color = "black", linewidth = 0.8, alpha = 0.5)
	ax_lines.set(xlabel = "Probe photon energy (eV)", ylabel = "Normalized line shape")
	ax_lines.legend(loc = "best")
	ax_lines.grid(alpha = 0.25)
	fig1.tight_layout()
	fig2.tight_layout()
	path.parent.mkdir(parents = True, exist_ok = True)
	fig1.savefig(path, dpi = dpi)
	fig2.savefig(os.path.join(os.path.dirname(path),"lorentz_theory.png"))
	plt.close(fig1)
	plt.close(fig2)
	return path


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description = __doc__)
	parser.add_argument("--spectra-csv", type = Path, default = DEFAULT_SPECTRA_CSV)
	parser.add_argument("--output-dir", type = Path, default = DEFAULT_OUTPUT_DIR)
	parser.add_argument("--delay-fs", type = float, default = 100.0)
	parser.add_argument("--signal-column", default = "TA_phase_avg")
	parser.add_argument("--energy-min-eV", type = float, default = 1.35)
	parser.add_argument("--energy-max-eV", type = float, default = 1.90)
	parser.add_argument("--energy-01-eV", type = float, default = 1.55)
	parser.add_argument("--energy-12-eV", type = float, default = 1.70)
	parser.add_argument("--gamma-1-fs-inv", type = float, default = 1.0 / 120.0)
	parser.add_argument("--gamma-2-fs-inv", type = float, default = 1.0 / 100.0)
	parser.add_argument("--dpi", type = int, default = 180)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	energy, signal, actual_delay = read_delay_spectrum(args.spectra_csv, delay_fs = args.delay_fs, signal_column = args.signal_column)
	energy, signal = restrict_energy_range(
		energy, signal, energy_min_eV = args.energy_min_eV, energy_max_eV = args.energy_max_eV,
	)
	signal_norm = normalized_maxabs(signal, name = args.signal_column)
	theory = make_theory_reference(
		energy,
		energy_01_eV = args.energy_01_eV,
		energy_12_eV = args.energy_12_eV,
		gamma_1_fs_inv = args.gamma_1_fs_inv,
		gamma_2_fs_inv = args.gamma_2_fs_inv,
	)
	data_path = args.output_dir / "data" / f"ta_delay_{actual_delay:g}fs_lorentz_linear_comparison.csv"
	figure_path = args.output_dir / "figures" / "plot" / f"ta_delay_{actual_delay:g}fs_lorentz_linear_comparison.png"
	write_comparison_csv(data_path, energy_eV = energy, signal = signal, signal_norm = signal_norm, theory = theory)
	plot_comparison(figure_path, energy_eV = energy, signal_norm = signal_norm, theory = theory, delay_fs = actual_delay, signal_column = args.signal_column, dpi = args.dpi)
	print(f"delay = {actual_delay:g} fs")
	print(f"gamma_01 = {float(theory['gamma_01_fs_inv']):.6g} fs^-1; gamma_12 = {float(theory['gamma_12_fs_inv']):.6g} fs^-1")
	print(f"comparison CSV: {data_path}")
	print(f"comparison plot: {figure_path}")


if __name__ == "__main__":
	main()
