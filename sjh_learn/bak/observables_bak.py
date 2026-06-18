"""analysis 层的谱学 observable 工具。

这些函数只从已经求解完成的 density matrix trajectory 和物理参数计算后处理量；
它们不属于 solver/core，也不应被 `DynamicsResult` 用来自动追加物理 observable。
"""

from __future__ import annotations

import numpy as np

from sjh_learn.utils.constants import DEBYE_TO_C_M, EPSILON0_F_PER_M, FS_TO_S, HBAR_J_S


FS_INV_TO_S_INV = 1.0 / FS_TO_S


def _as_density_trajectory(rho_t) -> np.ndarray:
    rho = np.asarray(rho_t, dtype=np.complex128)
    if rho.ndim != 3 or rho.shape[1] != rho.shape[2]:
        raise ValueError("rho_t 必须是 shape=(T, N, N) 的 density-matrix trajectory。")
    return rho


def _as_dipole_matrix(dipole_matrix_D, dimension: int) -> np.ndarray:
    mu = np.asarray(dipole_matrix_D, dtype=np.complex128)
    if mu.shape != (dimension, dimension):
        raise ValueError("dipole_matrix_D 必须是 shape=(N, N)，并且与 rho_t 的 N 一致。")
    return mu


def dipole_expectation_D(rho_t, dipole_matrix_D) -> np.ndarray:
    """计算单个量子体系的偶极矩期望值，单位 Debye。

    物理约定为 `p(t) = Tr[rho(t) mu] = sum_ij rho_ij(t) mu_ji`。实现使用
    `np.einsum("tij,ji->t", rho_t, dipole_matrix_D)`，避免逐时间点构造
    `rho @ mu` 的中间矩阵。这里必须使用用户侧物理偶极矩矩阵
    `dipole_matrix_D`，不能使用已经乘过场强和 code-unit 归一化因子的
    `coupling_matrix_code`。
    """

    rho = _as_density_trajectory(rho_t)
    mu = _as_dipole_matrix(dipole_matrix_D, rho.shape[1])
    return np.einsum("tij,ji->t", rho, mu)


def polarization_C_per_m2(rho_t, dipole_matrix_D, number_density_m3: float) -> np.ndarray:
    """计算宏观 polarization，单位 C/m^2。

    `number_density_m3` 的单位是 `m^-3`，必须显式给出。单体系偶极矩先由
    Debye 转换为 `C*m`，再乘以 number density，得到 `P(t)` 的 `C/m^2`。
    """

    density = float(number_density_m3)
    if density < 0:
        raise ValueError("number_density_m3 不能为负。")
    return density * dipole_expectation_D(rho_t, dipole_matrix_D) * DEBYE_TO_C_M


def chi_two_level_linear(
    omega_fs_inv,
    omega_eg_fs_inv: float,
    mu_ge_D: float,
    gamma2_fs_inv: float,
    number_density_m3: float,
    population_difference: float = 1.0,
) -> np.ndarray:
    """two-level analytic linear-response susceptibility 教学参考公式。

    这是 analysis 层的 analytic/teaching helper，不是 core two-level API，也不是
    `DynamicsResult` 的一部分。输入角频率使用 `fs^-1`，函数内部转换到 `s^-1`；
    `mu_ge_D` 使用 Debye，并转换为 `C*m`。
    """

    omega_s_inv = np.asarray(omega_fs_inv, dtype=float) * FS_INV_TO_S_INV
    omega_eg_s_inv = float(omega_eg_fs_inv) * FS_INV_TO_S_INV
    gamma2_s_inv = float(gamma2_fs_inv) * FS_INV_TO_S_INV
    if gamma2_s_inv < 0:
        raise ValueError("gamma2_fs_inv 不能为负。")
    density = float(number_density_m3)
    if density < 0:
        raise ValueError("number_density_m3 不能为负。")
    mu_C_m = float(abs(mu_ge_D)) * DEBYE_TO_C_M
    prefactor = density * (mu_C_m**2) / (EPSILON0_F_PER_M * HBAR_J_S)
    denominator = omega_eg_s_inv - omega_s_inv - 1j * gamma2_s_inv
    return prefactor * float(population_difference) / denominator


__all__ = [
    "DEBYE_TO_C_M",
    "EPSILON0_F_PER_M",
    "FS_INV_TO_S_INV",
    "HBAR_J_S",
    "dipole_expectation_D",
    "polarization_C_per_m2",
    "chi_two_level_linear",
]
