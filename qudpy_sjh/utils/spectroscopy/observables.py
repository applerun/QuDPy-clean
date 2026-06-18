"""密度矩阵轨迹的谱学 observable。"""

from __future__ import annotations

import numpy as np

from qudpy_sjh.utils.constants import DEBYE_TO_C_M


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

    物理约定为 `p(t)=Tr[rho(t) mu]=sum_ij rho_ij(t) mu_ji`。这里使用用户侧
    物理偶极矩 `dipole_matrix_D`，不能使用已经乘过场强和 code-unit
    归一化因子的 `coupling_matrix_code`。
    """

    rho = _as_density_trajectory(rho_t)
    mu = _as_dipole_matrix(dipole_matrix_D, rho.shape[1])
    return np.einsum("tij,ji->t", rho, mu)


def polarization_C_per_m2(rho_t, dipole_matrix_D, number_density_m3: float) -> np.ndarray:
    """计算宏观 polarization，单位 C/m^2。

    `number_density_m3` 的单位是 `m^-3`。单体系偶极矩先由 Debye 转成
    `C*m`，再乘 number density 得到 `C/m^2`。
    """

    density = float(number_density_m3)
    if density < 0:
        raise ValueError("number_density_m3 不能为负。")
    return density * dipole_expectation_D(rho_t, dipole_matrix_D) * DEBYE_TO_C_M


__all__ = ["dipole_expectation_D", "polarization_C_per_m2"]
