"""解析线性响应参考公式。"""

from __future__ import annotations

import numpy as np

from qudpy_sjh.utils.constants import EPSILON0_F_PER_M, FS_TO_S, HBAR_J_S, DEBYE_TO_C_M

FS_INV_TO_S_INV = 1.0 / FS_TO_S


def gamma2_fs_inv_from_T1_Tphi(T1_fs: float | None, Tphi_fs: float | None) -> float:
    gamma2 = 0.0
    if T1_fs is not None:
        gamma2 += 0.5 / float(T1_fs)
    if Tphi_fs is not None:
        gamma2 += 1.0 / float(Tphi_fs)
    return gamma2


def chi_two_level_linear(
        omega_fs_inv,
        omega_eg_fs_inv: float,
        mu_ge_D: complex,
        gamma2_fs_inv: float,
        number_density_m3: float,
        population_difference: float = 1.0,
) -> np.ndarray:
    """two-level analytic linear-response susceptibility 参考公式。

    输入角频率使用 `fs^-1`，函数内部转换为 `s^-1`；`mu_ge_D` 使用 Debye，
    并以 `|mu|` 转换为 `C*m`。这是线性响应参考，不是 solver 主线。
    """

    omega_s_inv = np.asarray(omega_fs_inv, dtype = float) * FS_INV_TO_S_INV
    omega_eg_s_inv = float(omega_eg_fs_inv) * FS_INV_TO_S_INV
    gamma2_s_inv = float(gamma2_fs_inv) * FS_INV_TO_S_INV
    if gamma2_s_inv < 0:
        raise ValueError("gamma2_fs_inv 不能为负。")
    density = float(number_density_m3)
    if density < 0:
        raise ValueError("number_density_m3 不能为负。")
    mu_C_m = float(abs(mu_ge_D)) * DEBYE_TO_C_M
    prefactor = density * (mu_C_m ** 2) / (EPSILON0_F_PER_M * HBAR_J_S)
    denominator = omega_eg_s_inv - omega_s_inv - 1j * gamma2_s_inv
    return prefactor * float(population_difference) / denominator


__all__ = [
    "EPSILON0_F_PER_M",
    "FS_INV_TO_S_INV",
    "HBAR_J_S",
    "chi_two_level_linear",
    "gamma2_fs_inv_from_T1_Tphi",
]
