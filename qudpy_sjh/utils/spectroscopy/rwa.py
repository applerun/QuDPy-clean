"""legacy RWA diagnostic 后处理。

RWA solver 默认禁用；本模块只保留从已有 RWA-like trajectory 做诊断比较的
后处理函数，不应成为 core 主流程依赖。
"""

from __future__ import annotations

import numpy as np

from qudpy_sjh.utils.core.normalization import ParaNormalizer
from qudpy_sjh.utils.constants import DEBYE_TO_C_M
from .absorption_spectra import lab_frame_fft_response_legacy


def _normalize_for_shape(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y)
    if np.iscomplexobj(y):
        y = np.real(y)
    y = y.astype(float)
    finite = np.isfinite(y)
    if not np.any(finite):
        return y
    scale = np.nanmax(np.abs(y[finite]))
    if scale == 0.0:
        return y
    return y / scale


def reconstruct_rwa_lab_polarization_C_per_m2(
    *,
    t_fs: np.ndarray,
    rho12_rwa: np.ndarray,
    dipole_matrix_D,
    number_density_m3: float,
    laser_energy_eV: float,
    carrier_sign: int,
) -> np.ndarray:
    """从 RWA 慢变量 coherence 诊断性重构 lab-frame polarization。

    这里只重构 off-diagonal dipole 贡献；diagonal permanent dipole 贡献需要
    population 项，主要落在 baseband 附近，不在这个 legacy diagnostic 中处理。
    """

    if carrier_sign not in (-1, 1):
        raise ValueError("carrier_sign must be -1 or +1.")

    t_fs = np.asarray(t_fs, dtype=float)
    rho12_rwa = np.asarray(rho12_rwa, dtype=np.complex128)
    dipole_matrix = np.asarray(dipole_matrix_D, dtype=np.complex128)
    mu_01_D = dipole_matrix[0, 1]
    mu_10_D = dipole_matrix[1, 0]
    omega_L_fs_inv = float(laser_energy_eV) * ParaNormalizer.EV_TO_FS_INV

    rho01_lab_like = rho12_rwa * np.exp(1j * int(carrier_sign) * omega_L_fs_inv * t_fs)
    rho10_lab_like = np.conjugate(rho01_lab_like)
    polarization = (
        float(number_density_m3)
        * DEBYE_TO_C_M
        * (rho01_lab_like * mu_10_D + rho10_lab_like * mu_01_D)
    )
    polarization = np.real_if_close(polarization, tol=1000)
    if np.iscomplexobj(polarization):
        max_imag = float(np.max(np.abs(np.imag(polarization))))
        max_real = float(np.max(np.abs(np.real(polarization))))
        if max_imag > 1e-8 * max(1.0, max_real):
            raise ValueError(
                "Reconstructed RWA polarization has a non-negligible imaginary part. "
                f"max_imag={max_imag:.3e}, max_real={max_real:.3e}"
            )
        polarization = np.real(polarization)
    return np.asarray(polarization, dtype=float)


def normalized_correlation(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    finite = np.isfinite(a) & np.isfinite(b)
    if np.count_nonzero(finite) < 3:
        return -np.inf
    aa = _normalize_for_shape(a[finite])
    bb = _normalize_for_shape(b[finite])
    aa = aa - np.mean(aa)
    bb = bb - np.mean(bb)
    denom = float(np.sqrt(np.sum(aa * aa) * np.sum(bb * bb)))
    if denom == 0.0:
        return -np.inf
    return float(np.sum(aa * bb) / denom)


def choose_rwa_reconstructed_p_over_e_response(
    *,
    t_fs: np.ndarray,
    E_t: np.ndarray,
    rho12_rwa_t: np.ndarray,
    rho12_lab_t: np.ndarray,
    dipole_matrix_D,
    number_density_m3: float,
    laser_energy_eV: float,
    lab_reference_response: dict[str, np.ndarray],
    energy_window_eV: tuple[float, float] = (1.4, 1.7),
    e_min: float | None = None,
    e_max: float | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray], str]:
    """尝试两个 carrier sign，选择更接近 lab `-omega Im[P/E]` 的重构。"""

    if e_min is not None or e_max is not None:
        energy_window_eV = (
            energy_window_eV[0] if e_min is None else float(e_min),
            energy_window_eV[1] if e_max is None else float(e_max),
        )

    best_score = -np.inf
    best_P = None
    best_response = None
    best_label = "unknown carrier sign"
    reference = lab_reference_response["neg_omega_im_P_over_E"]

    for sign in (-1, 1):
        P_recon = reconstruct_rwa_lab_polarization_C_per_m2(
            t_fs=t_fs,
            rho12_rwa=rho12_rwa_t,
            dipole_matrix_D=dipole_matrix_D,
            number_density_m3=number_density_m3,
            laser_energy_eV=laser_energy_eV,
            carrier_sign=sign,
        )
        response = lab_frame_fft_response_legacy(
            t_fs=t_fs,
            E_MV_per_cm=E_t,
            P_C_per_m2=P_recon,
            rhoij=rho12_lab_t,
            window="hann",
            subtract_mean=True,
            rel_threshold=1e-5,
            zero_padding_factor=4,
        )
        band = (response["energy_eV"] >= energy_window_eV[0]) & (response["energy_eV"] <= energy_window_eV[1])
        score = normalized_correlation(response["neg_omega_im_P_over_E"][band], reference[band])
        if score > best_score:
            best_score = score
            best_P = P_recon
            best_response = response
            best_label = f"rho12_RWA * exp({sign:+d} i omega_L t), corr={score:.3f}"

    if best_P is None or best_response is None:
        raise RuntimeError("Failed to reconstruct RWA lab-frame polarization.")
    return best_P, best_response, best_label


__all__ = [
    "choose_rwa_reconstructed_p_over_e_response",
    "normalized_correlation",
    "reconstruct_rwa_lab_polarization_C_per_m2",
]
