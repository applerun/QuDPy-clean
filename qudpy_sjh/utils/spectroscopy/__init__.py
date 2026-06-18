"""QuDPy 谱学后处理入口。

本包只处理已完成模拟的 `rho(t)`、输入场 `E(t)` / drive、polarization 和
频域响应；不负责构造 Hamiltonian，也不引入实验数据分析依赖。
"""

from .observables import dipole_expectation_D, polarization_C_per_m2
from .absorption_spectra import (
    apply_time_window,
    diagnose_uniform_time_axis,
    lab_frame_absorption_response,
    lab_frame_fft_response_legacy,
    rwa_fft_response,
    safe_complex_ratio,
)
from .theory import EPSILON0_F_PER_M, chi_two_level_linear, gamma2_fs_inv_from_T1_Tphi

__all__ = [
    "DEBYE_TO_C_M",
    "EPSILON0_F_PER_M",
    "apply_time_window",
    "chi_two_level_linear",
    "diagnose_uniform_time_axis",
    "dipole_expectation_D",
    "gamma2_fs_inv_from_T1_Tphi",
    "lab_frame_absorption_response",
    "lab_frame_fft_response_legacy",
    "polarization_C_per_m2",
    "rwa_fft_response",
    "safe_complex_ratio",
]
