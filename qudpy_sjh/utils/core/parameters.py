"""参数数据结构。

本模块只定义数据容器，不做单位换算、不构造 Hamiltonian，也不调用
QuTiP。用户侧物理系统使用 `NLevelPhysicalParams`；solver 内部参数使用
`NLevelSolverParams`。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
	from qudpy_sjh.utils.fields import FieldPhyRoot


@dataclass(frozen = True)
class RelaxationChannel:
	"""population relaxation 通道。

	物理约定：`C_{to <- from} = sqrt(rate) |to><from|`。
	可以用 `T1_fs` 或 `rate_fs_inv` 指定速率；二者都给出时，
	normalizer 会优先使用 `rate_fs_inv`。
	"""

	name: str
	from_level: int
	to_level: int
	T1_fs: float | None = None
	rate_fs_inv: float | None = None


@dataclass(frozen = True)
class PureDephasingChannel:
	"""level projector pure dephasing 通道。

	物理约定：`C_level^phi = sqrt(rate) |level><level|`。
	可以用 `Tphi_fs` 或 `rate_fs_inv` 指定速率；二者都给出时，
	normalizer 会优先使用 `rate_fs_inv`。
	"""

	name: str
	level: int
	Tphi_fs: float | None = None
	rate_fs_inv: float | None = None


@dataclass(frozen = True)
class NLevelPhysicalParams:
	"""用户侧 N-level 物理系统输入。

	该类描述一个 N 能级体系及其外加光场输入。所有普通输入均使用
	真实物理单位：能量用 eV，偶极矩用 Debye，时间用 fs。输入光场
	只能由 `field` 对象描述，`NLevelPhysicalParams` 不再构造默认
	GaussianCarrierFieldPhysical。

	`dipole_matrix_D` 表示沿选定 optical polarization 投影后的
	跃迁偶极矩矩阵，单位 Debye。矩阵元素可以为正、负、零或复数。

	two-level system 只是普通的 N=2 system。核心层不再把二能级
	作为特殊的标量模型处理。
	"""

	# 各能级能量，单位 eV。
	# 第 i 个元素对应第 i 个 basis state 的能量。
	energies_eV: tuple[float, ...]

	# 跃迁偶极矩矩阵，单位 Debye。
	# 这是沿选定 optical polarization 投影后的偶极矩矩阵。
	# 元素 mu_ij 可以为正、负、零或复数。
	# 若 abs(mu_ij) 近似为 0，则该 i<->j transition 没有直接
	# electric-dipole coupling，但该能级差仍然存在。
	dipole_matrix_D: tuple[tuple[complex, ...], ...]

	# 模拟起始时间，单位 fs。
	t_start_fs: float

	# 模拟结束时间，单位 fs。
	t_end_fs: float

	# 时间步长，单位 fs。
	dt_fs: float

	# 唯一参与求解的输入光场对象。
	# field metadata 由 field 对象自身导出；本 dataclass 不保存
	# field_MV_per_cm、laser_energy_eV 或 pulse_* 这类顶层光场标量。
	field: FieldPhyRoot

	# 可选 basis state 名称。
	# 若为 None，则默认使用 "0", "1", ..., "N-1"。
	basis: tuple[str, ...] | None = None

	# population relaxation 通道列表。
	# 每个通道描述一个 Lindblad relaxation channel。
	# 同一组 from_level -> to_level 可以存在多个物理通道，
	# 例如 radiative 与 nonradiative relaxation。
	relaxation_channels: tuple[RelaxationChannel, ...] = ()

	# pure dephasing 通道列表。
	# 每个通道描述一个不改变 population 的 dephasing channel。
	# 同一能级也可以有多个 dephasing 来源。
	pure_dephasing_channels: tuple[PureDephasingChannel, ...] = ()

	# solver 表示模式。
	# 当前主线是 "lab_exact"；"rwa" 是默认禁用的 legacy mode。
	solver_mode: str = "lab_exact"

	# 用户自定义输入说明。
	# 只用于 meta.json / debug_meta.json 记录。
	# 不参与归一化、Hamiltonian 构造或求解。
	input_description: str | None = None

	# 用户自定义 metadata。
	# 只用于记录用户原始备注，不作为规范化系统描述。
	# 规范化系统信息应从 energies_eV、dipole_matrix_D、
	# field、dissipation 等字段自动生成。
	input_metadata: dict[str, Any] | None = None

	@property
	def dimension(self) -> int:
		"""体系维度，即能级数量 N。"""
		return len(self.energies_eV)

	@property
	def energy_gap_eV(self) -> float:
		"""0->1 能隙，单位 eV。

		这是 N=2 教学示例中常用的兼容属性。核心模型仍以
		`energies_eV` 为准。对 N>2 system，该属性只表示第 0
		和第 1 个能级之间的能量差，不代表体系唯一能隙。
		"""
		if self.dimension < 2:
			raise ValueError("energy_gap_eV requires at least two levels.")
		return float(self.energies_eV[1] - self.energies_eV[0])


@dataclass
class SolverParams:
	"""归一化后的 solver 参数摘要。

	这里包含 fs^-1 和 code-unit 两套量，供 solver 构造内部参数以及
	`debug_meta.json` 调试使用。普通用户不应直接构造这个对象。
	"""

	time_scale_fs: float
	energies_fs_inv: np.ndarray
	energies_code: np.ndarray
	dipole_matrix_D: np.ndarray
	coupling_matrix_fs_inv: np.ndarray
	coupling_matrix_code: np.ndarray
	relaxation_channels_fs_inv: tuple[dict[str, Any], ...]
	pure_dephasing_channels_fs_inv: tuple[dict[str, Any], ...]
	relaxation_channels_code: tuple[dict[str, Any], ...]
	pure_dephasing_channels_code: tuple[dict[str, Any], ...]
	omega_L_fs_inv: float | None
	omega_L: float | None
	t_start: float
	t_end: float
	dt: float
	tlist: np.ndarray
	pulse_center: float | None = None
	pulse_sigma: float | None = None
	pulse_center_fs: float | None = None
	pulse_sigma_fs: float | None = None

	@property
	def omega_eg_fs_inv(self) -> float:
		return float(self.energies_fs_inv[1] - self.energies_fs_inv[0]) if len(self.energies_fs_inv) >= 2 else 0.0

	@property
	def omega_eg(self) -> float:
		return self.omega_eg_fs_inv * self.time_scale_fs

	@property
	def detuning_fs_inv(self) -> float | None:
		if self.omega_L_fs_inv is None:
			return None
		return self.omega_eg_fs_inv - self.omega_L_fs_inv

	@property
	def detuning(self) -> float | None:
		detuning_fs_inv = self.detuning_fs_inv
		return None if detuning_fs_inv is None else detuning_fs_inv * self.time_scale_fs

	@property
	def rabi_fs_inv(self) -> float:
		if self.coupling_matrix_fs_inv.shape[0] < 2:
			return 0.0
		return float(self.coupling_matrix_fs_inv[0, 1].real)

	@property
	def rabi(self) -> float:
		return self.rabi_fs_inv * self.time_scale_fs

	@property
	def gamma1_fs_inv(self) -> float:
		for channel in self.relaxation_channels_fs_inv:
			if channel.get("from_level") == 1 and channel.get("to_level") == 0:
				return float(channel["rate_fs_inv"])
		return 0.0

	@property
	def gamma_phi_fs_inv(self) -> float:
		if len(self.pure_dephasing_channels_fs_inv) == 1:
			return float(self.pure_dephasing_channels_fs_inv[0]["rate_fs_inv"])
		if len(self.pure_dephasing_channels_fs_inv) >= 2:
			rates = [float(item["rate_fs_inv"]) for item in self.pure_dephasing_channels_fs_inv[:2]]
			return 0.5 * sum(rates)
		return 0.0

	@property
	def gamma2_fs_inv(self) -> float:
		return self.gamma_phi_fs_inv + 0.5 * self.gamma1_fs_inv

	@property
	def gamma1(self) -> float:
		return self.gamma1_fs_inv * self.time_scale_fs

	@property
	def gamma_phi(self) -> float:
		return self.gamma_phi_fs_inv * self.time_scale_fs

	@property
	def gamma2(self) -> float:
		return self.gamma2_fs_inv * self.time_scale_fs


@dataclass(frozen = True)
class NLevelSolverParams:
	"""内部 N-level solver 参数。

	这里的矩阵、频率、时间和速率已经是 solver code unit。普通用户侧
	示例应先构造 `NLevelPhysicalParams`，再经 `ParaNormalizer` 转换。
	"""

	t_start: float = 0.0
	t_final: float = 120.0
	t_end: float | None = None
	dt: float = 0.01
	hbar: float = 1.0
	energies: tuple[float, ...] = (0.0, 1.0)
	dipole_matrix: tuple[tuple[complex, ...], ...] = ((0.0, 1.0), (1.0, 0.0))
	coupling_matrix: tuple[tuple[complex, ...], ...] | None = None
	omega_drive: float = 1.0
	relaxation_channels: tuple[dict[str, Any], ...] = ()
	pure_dephasing_channels: tuple[dict[str, Any], ...] = ()
	field: Any | None = None
	tlist: object | None = None
	times_fs: object | None = None
	pulse_center: float | None = None
	pulse_sigma: float | None = None
	basis: tuple[str, ...] | None = None
	detuning: float = 0.0


def as_complex_matrix(value) -> np.ndarray:
	return np.asarray(value, dtype = np.complex128)


__all__ = [
	"NLevelPhysicalParams",
	"RelaxationChannel",
	"PureDephasingChannel",
	"SolverParams",
	"NLevelSolverParams",
	"as_complex_matrix",
]
