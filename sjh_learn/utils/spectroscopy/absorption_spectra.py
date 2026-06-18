"""从时间域模拟信号计算谱学响应。"""

from __future__ import annotations

import numpy as np

from sjh_learn.utils.core.normalization import ParaNormalizer


def apply_time_window(values: np.ndarray, window: str | None) -> np.ndarray:
    if window is None or window == "none":
        return values
    if window == "hann":
        return values * np.hanning(values.size)
    raise ValueError("window must be None, 'none', or 'hann'.")


def safe_complex_ratio(
    numerator: np.ndarray,
    denominator: np.ndarray,
    *,
    rel_threshold: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    denominator_abs = np.abs(denominator)
    max_denominator = float(np.max(denominator_abs))
    if max_denominator == 0.0:
        raise ValueError("The input spectrum is identically zero.")
    valid = denominator_abs > rel_threshold * max_denominator
    ratio = np.full_like(numerator, np.nan + 1j * np.nan, dtype=np.complex128)
    ratio[valid] = numerator[valid] / denominator[valid]
    return ratio, valid


def diagnose_uniform_time_axis(
    time_fs,
    *,
    rtol: float = 1e-7,
    atol: float = 1e-9,
) -> dict[str, object]:
    """Summarize whether a time axis is uniformly sampled in femtoseconds."""

    time = np.asarray(time_fs, dtype=float)
    diffs = np.diff(time)
    median_dt = None if diffs.size == 0 else float(np.median(diffs))
    is_uniform = bool(
        diffs.size > 0
        and median_dt is not None
        and np.allclose(diffs, median_dt, rtol=float(rtol), atol=float(atol))
    )
    return {
        "n": int(time.size),
        "t0_fs": None if time.size == 0 else float(time[0]),
        "t1_fs": None if time.size == 0 else float(time[-1]),
        "min_dt_fs": None if diffs.size == 0 else float(np.min(diffs)),
        "max_dt_fs": None if diffs.size == 0 else float(np.max(diffs)),
        "median_dt_fs": median_dt,
        "is_uniform": is_uniform,
    }


def _uniform_dt_fs(t_fs: np.ndarray) -> float:
    diagnostics = diagnose_uniform_time_axis(t_fs, rtol=1e-5, atol=1e-10)
    if not diagnostics["is_uniform"]:
        raise ValueError(
            "FFT requires a uniformly sampled time axis with at least two points. "
            f"time_axis={diagnostics}"
        )
    return float(diagnostics["median_dt_fs"])


def _fft_size(n_samples: int, zero_padding_factor: int) -> int:
    n_fft_target = int(n_samples * zero_padding_factor)
    return 1 << int(np.ceil(np.log2(max(n_fft_target, n_samples))))


def lab_frame_absorption_response(
    *,
    time_fs=None,
    polarization_C_per_m2=None,
    field=None,
    t_fs=None,
    P_C_per_m2=None,
    E_MV_per_cm=None,
    window: str | None = "hann",
    subtract_mean: bool = True,
    rel_threshold: float = 1e-6,
    zero_padding_factor: int = 4,
    return_intermediates: bool = False,
    include_legacy_alias: bool = False,
) -> dict[str, np.ndarray]:
    """Compute the lab-frame absorption-only response from ``P(t)`` and ``E(t)``.

    The main spectrum is ``absorption = omega * Im[P(omega) / E(omega)]``.
    ``neg_omega_im_P_over_E`` is intentionally not a primary output; callers
    that still need the old sign convention can request it as an alias.
    """

    if time_fs is None:
        time_fs = t_fs
    if polarization_C_per_m2 is None:
        polarization_C_per_m2 = P_C_per_m2
    if field is None:
        field = E_MV_per_cm
    if time_fs is None or polarization_C_per_m2 is None or field is None:
        raise ValueError("time_fs, polarization_C_per_m2, and field are required.")

    time = np.asarray(time_fs, dtype=float)
    diagnostics = diagnose_uniform_time_axis(time)
    if not diagnostics["is_uniform"]:
        raise ValueError(
            "lab_frame_absorption_response requires a uniformly sampled time axis. "
            f"time_axis={diagnostics}"
        )
    dt_fs = float(diagnostics["median_dt_fs"])

    field_values = np.asarray(field, dtype=float)
    polarization = np.asarray(polarization_C_per_m2, dtype=np.complex128)
    if time.shape != field_values.shape or time.shape != polarization.shape:
        raise ValueError("time_fs, field, and polarization_C_per_m2 must have the same shape.")

    E_signal = field_values.astype(np.complex128)
    P_signal = polarization.astype(np.complex128)
    if subtract_mean:
        E_signal = E_signal - np.mean(E_signal)
        P_signal = P_signal - np.mean(P_signal)

    n_fft = _fft_size(time.size, zero_padding_factor)
    E_fft = np.fft.fft(apply_time_window(E_signal, window), n=n_fft)
    P_fft = np.fft.fft(apply_time_window(P_signal, window), n=n_fft)
    freq_fs_inv = np.fft.fftfreq(n_fft, d=dt_fs)
    omega_fs_inv = 2.0 * np.pi * freq_fs_inv
    energy_eV = omega_fs_inv / ParaNormalizer.EV_TO_FS_INV

    P_over_E, valid_E = safe_complex_ratio(P_fft, E_fft, rel_threshold=rel_threshold)
    mask = (freq_fs_inv > 0) & valid_E
    omega = omega_fs_inv[mask]
    p_over_e = P_over_E[mask]
    absorption = omega * np.imag(p_over_e)
    response: dict[str, object] = {
        "omega_fs_inv": omega,
        "energy_eV": energy_eV[mask],
        "absorption": absorption,
        "omega_im_P_over_E": absorption,
        "metadata": {
            "time_axis": diagnostics,
            "absorption_definition": "absorption = omega * Im[P(omega) / E(omega)]",
        },
    }
    if include_legacy_alias:
        response["neg_omega_im_P_over_E"] = -absorption
        response["metadata"]["legacy_alias"] = (
            "neg_omega_im_P_over_E is retained only on request; "
            "new code should use absorption or omega_im_P_over_E."
        )
    if return_intermediates:
        response.update(
            {
                "E_omega": E_fft[mask],
                "P_omega": P_fft[mask],
                "P_over_E": p_over_e,
                "abs_E_omega": np.abs(E_fft[mask]),
            }
        )
    return response


def lab_frame_fft_response_legacy(
    *,
    t_fs: np.ndarray,
    E_MV_per_cm: np.ndarray,
    P_C_per_m2: np.ndarray,
    rhoij: np.ndarray | None = None,
    rho12: np.ndarray | None = None,
    window: str | None = "hann",
    subtract_mean: bool = True,
    rel_threshold: float = 1e-6,
    zero_padding_factor: int = 4,
) -> dict[str, np.ndarray]:
    """计算 lab-frame pulse 响应谱。

    `neg_omega_im_P_over_E` 保留当前数值行为：在 `np.fft.fft` 的正频符号
    约定下使用 `-omega * Im[P/E]` 作为 absorption-like spectrum。
    """

    if rhoij is None:
        if rho12 is None:
            raise ValueError("rhoij is required.")
        rhoij = rho12

    t_fs = np.asarray(t_fs, dtype=float)
    E_MV_per_cm = np.asarray(E_MV_per_cm, dtype=float)
    P_C_per_m2 = np.asarray(P_C_per_m2, dtype=np.complex128)
    rho_signal = np.asarray(rhoij, dtype=np.complex128)

    dt_fs = _uniform_dt_fs(t_fs)
    E_signal = E_MV_per_cm.astype(np.complex128)
    P_signal = P_C_per_m2.astype(np.complex128)
    if subtract_mean:
        E_signal = E_signal - np.mean(E_signal)
        P_signal = P_signal - np.mean(P_signal)
        rho_signal = rho_signal - np.mean(rho_signal)

    n_fft = _fft_size(t_fs.size, zero_padding_factor)
    E_fft = np.fft.fft(apply_time_window(E_signal, window), n=n_fft)
    P_fft = np.fft.fft(apply_time_window(P_signal, window), n=n_fft)
    rho_fft = np.fft.fft(apply_time_window(rho_signal, window), n=n_fft)

    freq_fs_inv = np.fft.fftfreq(n_fft, d=dt_fs)
    omega_fs_inv = 2.0 * np.pi * freq_fs_inv
    energy_eV = omega_fs_inv / ParaNormalizer.EV_TO_FS_INV

    P_over_E, valid_E = safe_complex_ratio(P_fft, E_fft, rel_threshold=rel_threshold)
    rho_over_E, _ = safe_complex_ratio(rho_fft, E_fft, rel_threshold=rel_threshold)
    mask = (freq_fs_inv > 0) & valid_E

    response = {
        "omega_fs_inv": omega_fs_inv[mask],
        "energy_eV": energy_eV[mask],
        "E_fft": E_fft[mask],
        "P_fft": P_fft[mask],
        "rhoij_fft": rho_fft[mask],
        "P_over_E": P_over_E[mask],
        "rhoij_over_E": rho_over_E[mask],
        "abs_E_fft": np.abs(E_fft[mask]),
        "abs_rhoij_over_E": np.abs(rho_over_E[mask]),
        "im_rhoij_over_E": np.imag(rho_over_E[mask]),
        "omega_im_rhoij_over_E": omega_fs_inv[mask] * np.imag(rho_over_E[mask]),
        "neg_omega_im_P_over_E": -omega_fs_inv[mask] * np.imag(P_over_E[mask]),
    }
    # 兼容旧 benchmark 脚本中的 rho12 命名。
    response["rho12_fft"] = response["rhoij_fft"]
    response["rho12_over_E"] = response["rhoij_over_E"]
    response["abs_rho12_over_E"] = response["abs_rhoij_over_E"]
    response["im_rho12_over_E"] = response["im_rhoij_over_E"]
    response["omega_im_rho12_over_E"] = response["omega_im_rhoij_over_E"]
    return response


def rwa_fft_response(
    *,
    t_fs: np.ndarray,
    g_fs_inv: np.ndarray,
    rho12_rwa: np.ndarray,
    laser_energy_eV: float,
    window: str | None = "hann",
    subtract_mean: bool = False,
    rel_threshold: float = 1e-6,
    zero_padding_factor: int = 4,
) -> dict[str, np.ndarray]:
    """legacy RWA diagnostic 响应谱；不运行 RWA solver。"""

    t_fs = np.asarray(t_fs, dtype=float)
    g_signal = np.asarray(g_fs_inv, dtype=np.complex128)
    rho_signal = np.asarray(rho12_rwa, dtype=np.complex128)
    dt_fs = _uniform_dt_fs(t_fs)

    if subtract_mean:
        g_signal = g_signal - np.mean(g_signal)
        rho_signal = rho_signal - np.mean(rho_signal)

    n_fft = _fft_size(t_fs.size, zero_padding_factor)
    g_fft = np.fft.fft(apply_time_window(g_signal, window), n=n_fft)
    rho_fft = np.fft.fft(apply_time_window(rho_signal, window), n=n_fft)

    freq_offset_fs_inv = np.fft.fftfreq(n_fft, d=dt_fs)
    omega_offset_fs_inv = 2.0 * np.pi * freq_offset_fs_inv
    energy_eV = laser_energy_eV + omega_offset_fs_inv / ParaNormalizer.EV_TO_FS_INV

    rho_over_g, valid_g = safe_complex_ratio(rho_fft, g_fft, rel_threshold=rel_threshold)
    idx = np.where(valid_g)[0]
    return {
        "omega_offset_fs_inv": omega_offset_fs_inv[idx],
        "energy_eV": energy_eV[idx],
        "g_fft": g_fft[idx],
        "rho12_rwa_over_g": rho_over_g[idx],
        "abs_g_fft": np.abs(g_fft[idx]),
        "abs_rho12_rwa_over_g": np.abs(rho_over_g[idx]),
        "im_rho12_rwa_over_g": np.imag(rho_over_g[idx]),
        "omega_im_rho12_rwa_over_g": omega_offset_fs_inv[idx] * np.imag(rho_over_g[idx]),
    }


# 旧名称兼容。
apply_window = apply_time_window
safe_complex_divide = safe_complex_ratio
fft_pulse_response = lab_frame_fft_response_legacy
fft_rwa_response = rwa_fft_response


__all__ = [
    "apply_time_window",
    "apply_window",
    "diagnose_uniform_time_axis",
    "fft_pulse_response",
    "fft_rwa_response",
    "lab_frame_absorption_response",
    "lab_frame_fft_response_legacy",
    "rwa_fft_response",
    "safe_complex_divide",
    "safe_complex_ratio",
]
