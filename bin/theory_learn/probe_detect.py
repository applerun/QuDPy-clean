#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
probe_detect.py

Purpose
-------
Demonstrate the difference between:

1. time-domain square-law products:
       FFT[Eout(t)^2]
       FFT[Eprobe(t) Esig(t)]
       FFT[Eprobe(t) Esig(t) g(t)]

2. spectrometer-like field readout:
       |FFT[Eout(t)]|^2
       |FFT[Eprobe(t) + Esig(t)]|^2

The key point is that an ordinary spectrometer first decomposes the field
in frequency and then measures intensity:

       I(omega) = |Eout(omega)|^2

not:

       FFT[ |Eout(t)|^2 ]

This script also compares the ordinary readout with a gated readout:

       I_gated(omega) = |FFT[g(t) Eout(t)]|^2

where g(t) is a broad active window around the probe.

Parameters are chosen to mimic an extreme pump-probe simulation:
- Eprobe(t) is only a few fs wide.
- The active window is +/- 500 fs around the probe.
- Esig(t), representing a long coherent polarization-emitted tail, lasts > 1000 fs.

Run
---
python probe_detect.py
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 0. User parameters
# ============================================================

OUTDIR = Path("probe_detect_outputs")
OUTDIR.mkdir(exist_ok=True)

# Time grid
dt = 0.05  # fs
t_min = -1500.0  # fs
t_max = 2500.0   # fs

# Optical carrier
# f0 in cycles/fs. 0.375 cycles/fs is roughly 800 nm.
f0 = 0.375
w0 = 2.0 * np.pi * f0

# Very short probe pulse
probe_center = 0.0      # fs
probe_sigma = 2.5       # fs, Gaussian sigma; FWHM = 2.355*sigma ~ 5.9 fs
probe_amp = 1.0

# Long weak coherent emitted tail / FID-like signal
sig_start = 8.0         # fs, starts shortly after probe center
sig_tau = 1400.0        # fs, long decay; coherent tail extends beyond 1000 fs
sig_amp = 0.04          # weak amplitude
sig_phase = 0.0         # change this to test quadrature effects
sig_detune = 0.0        # cycles/fs; use e.g. 0.002 to shift Esig carrier slightly

# Active window / gate around the probe
gate_left = -500.0      # fs
gate_right = 500.0      # fs

# Frequency windows for plotting
carrier_plot_half_width = 0.05   # cycles/fs, near f0
broad_freq_max = 1.2             # cycles/fs, for time-domain product spectra


# ============================================================
# 1. Helpers
# ============================================================

def integrate_trapz(y, x):
    """Compatible trapezoidal integration for different NumPy versions."""
    if hasattr(np, "trapezoid"):
        return np.trapezoid(y, x)
    return np.trapz(y, x)


def fft_shift(x, dt_value):
    """
    FFT with shift and dt scaling.

    Convention:
    NumPy FFT places exp(+i 2π f0 t) at +f0.
    """
    return np.fft.fftshift(np.fft.fft(np.fft.ifftshift(x))) * dt_value


def sci_y(ax):
    """Use scientific notation for y-axis when appropriate."""
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))


def safe_ratio(a, b):
    if b == 0:
        return np.nan
    return a / b


# ============================================================
# 2. Construct time grid and fields
# ============================================================

t = np.arange(t_min, t_max, dt)
N = t.size

freq = np.fft.fftshift(np.fft.fftfreq(N, d=dt))  # cycles/fs

mask_carrier = (freq > f0 - carrier_plot_half_width) & (freq < f0 + carrier_plot_half_width)
mask_broad_pos = (freq > 0.0) & (freq < broad_freq_max)

# Probe: analytic complex field
Eprobe_env = probe_amp * np.exp(-0.5 * ((t - probe_center) / probe_sigma) ** 2)
Eprobe = Eprobe_env * np.exp(+1j * w0 * t)

# Long coherent signal tail
H = (t >= sig_start).astype(float)
Esig_env = sig_amp * H * np.exp(-(t - sig_start) / sig_tau)
Esig_carrier = 2.0 * np.pi * (f0 + sig_detune)
Esig = Esig_env * np.exp(+1j * (Esig_carrier * (t - sig_start) + sig_phase))

# Output field
Eout = Eprobe + Esig

# Real physical fields for time-domain square-law products
Eprobe_r = np.real(Eprobe)
Esig_r = np.real(Esig)
Eout_r = np.real(Eout)

# Gate / active window
gate = ((t >= gate_left) & (t <= gate_right)).astype(float)

Eprobe_g = gate * Eprobe
Esig_g = gate * Esig
Eout_g = gate * Eout

Eprobe_g_r = np.real(Eprobe_g)
Esig_g_r = np.real(Esig_g)
Eout_g_r = np.real(Eout_g)


# ============================================================
# 3. FFTs of fields
# ============================================================

Eprobe_w = fft_shift(Eprobe, dt)
Esig_w = fft_shift(Esig, dt)
Eout_w = fft_shift(Eout, dt)

Eprobe_g_w = fft_shift(Eprobe_g, dt)
Esig_g_w = fft_shift(Esig_g, dt)
Eout_g_w = fft_shift(Eout_g, dt)


# ============================================================
# 4. Figure 1
# Time-domain fields, gate, and field spectra
# ============================================================

fig1, axes = plt.subplots(2, 3, figsize=(16, 8))

# Panel 1: very short probe and gate
axes[0, 0].plot(t, Eprobe_r, label="Re[Eprobe(t)]")
axes[0, 0].plot(t, Eprobe_env, linestyle="--", linewidth=1, label="probe envelope")
axes[0, 0].plot(t, -Eprobe_env, linestyle="--", linewidth=1)
axes[0, 0].plot(t, gate * np.max(Eprobe_env), linestyle=":", linewidth=1.5, label="gate scaled")
axes[0, 0].set_xlim(-650, 650)
axes[0, 0].set_title("Figure 1-1: short probe and ±500 fs gate")
axes[0, 0].set_xlabel("time (fs)")
axes[0, 0].set_ylabel("field")
axes[0, 0].legend()
sci_y(axes[0, 0])

# Panel 2: long Esig and gated Esig
axes[0, 1].plot(t, Esig_r, label="Re[Esig(t)]")
axes[0, 1].plot(t, Esig_g_r, linestyle="--", label="Re[g(t) Esig(t)]")
axes[0, 1].plot(t, Esig_env, linestyle=":", linewidth=1, label="Esig envelope")
axes[0, 1].plot(t, -Esig_env, linestyle=":", linewidth=1)
axes[0, 1].axvline(gate_right, linestyle=":", linewidth=1)
axes[0, 1].set_xlim(-100, 1800)
axes[0, 1].set_title("Figure 1-2: long coherent tail and gated tail")
axes[0, 1].set_xlabel("time (fs)")
axes[0, 1].set_ylabel("field")
axes[0, 1].legend()
sci_y(axes[0, 1])

# Panel 3: gate and normalized envelopes
max_probe_env = np.max(Eprobe_env)
max_sig_env = np.max(Esig_env) if np.max(Esig_env) != 0 else 1.0
axes[0, 2].plot(t, gate, label="gate g(t)")
axes[0, 2].plot(t, Eprobe_env / max_probe_env, linestyle="--", label="probe envelope, scaled")
axes[0, 2].plot(t, Esig_env / max_sig_env, linestyle=":", label="Esig envelope, scaled")
axes[0, 2].set_xlim(-650, 1800)
axes[0, 2].set_ylim(-0.1, 1.2)
axes[0, 2].set_title("Figure 1-3: gate length versus response tail")
axes[0, 2].set_xlabel("time (fs)")
axes[0, 2].set_ylabel("scaled amplitude")
axes[0, 2].legend()

# Panel 4: probe spectrum
axes[1, 0].plot(freq[mask_carrier], np.abs(Eprobe_w)[mask_carrier], label="|Eprobe(ω)|")
axes[1, 0].plot(freq[mask_carrier], np.abs(Eprobe_g_w)[mask_carrier], linestyle="--", label="|gEprobe(ω)|")
axes[1, 0].set_title("Figure 1-4: probe field spectrum")
axes[1, 0].set_xlabel("frequency (cycles/fs)")
axes[1, 0].set_ylabel("raw spectral amplitude")
axes[1, 0].legend()
sci_y(axes[1, 0])

# Panel 5: Esig spectrum versus gated Esig spectrum
axes[1, 1].plot(freq[mask_carrier], np.abs(Esig_w)[mask_carrier], label="|Esig(ω)|")
axes[1, 1].plot(freq[mask_carrier], np.abs(Esig_g_w)[mask_carrier], linestyle="--", label="|gEsig(ω)|")
axes[1, 1].set_title("Figure 1-5: tail spectrum changed by gate")
axes[1, 1].set_xlabel("frequency (cycles/fs)")
axes[1, 1].set_ylabel("raw spectral amplitude")
axes[1, 1].legend()
sci_y(axes[1, 1])

# Panel 6: Eout spectrum versus gated Eout spectrum
axes[1, 2].plot(freq[mask_carrier], np.abs(Eout_w)[mask_carrier], label="|Eout(ω)|")
axes[1, 2].plot(freq[mask_carrier], np.abs(Eout_g_w)[mask_carrier], linestyle="--", label="|gEout(ω)|")
axes[1, 2].set_title("Figure 1-6: output spectrum changed by gate")
axes[1, 2].set_xlabel("frequency (cycles/fs)")
axes[1, 2].set_ylabel("raw spectral amplitude")
axes[1, 2].legend()
sci_y(axes[1, 2])

fig1.tight_layout()
fig1.savefig(OUTDIR / "figure1_time_freq_with_gate_long_tail.png", dpi=300)


# ============================================================
# 5. Figure 2
# Time-domain square-law products, then FFT
#
# These are intentionally NOT ordinary spectrometer readouts.
# They answer a different question:
# first multiply fields in time, then Fourier transform the product.
# ============================================================

I_time = Eout_r ** 2
cross_time = Eprobe_r * Esig_r
cross_time_g = Eprobe_r * Esig_r * gate

I_time_w = fft_shift(I_time, dt)
cross_time_w = fft_shift(cross_time, dt)
cross_time_g_w = fft_shift(cross_time_g, dt)

fig2, axes = plt.subplots(1, 3, figsize=(16, 4.8))

axes[0].plot(freq[mask_broad_pos], np.abs(I_time_w)[mask_broad_pos])
axes[0].set_title("Figure 2-1: |FFT[Eout(t)^2]|")
axes[0].set_xlabel("frequency (cycles/fs)")
axes[0].set_ylabel("raw amplitude")
sci_y(axes[0])

axes[1].plot(freq[mask_broad_pos], np.abs(cross_time_w)[mask_broad_pos])
axes[1].set_title("Figure 2-2: |FFT[Eprobe(t) Esig(t)]|")
axes[1].set_xlabel("frequency (cycles/fs)")
axes[1].set_ylabel("raw amplitude")
sci_y(axes[1])

axes[2].plot(freq[mask_broad_pos], np.abs(cross_time_g_w)[mask_broad_pos])
axes[2].set_title("Figure 2-3: |FFT[Eprobe Esig × g(t)]|")
axes[2].set_xlabel("frequency (cycles/fs)")
axes[2].set_ylabel("raw amplitude")
sci_y(axes[2])

fig2.tight_layout()
fig2.savefig(OUTDIR / "figure2_time_domain_square_law_products_long_tail.png", dpi=300)


# ============================================================
# ============================================================
# 6. Figure 3
# Ordinary spectrometer readout versus gated spectrometer readout
#
# Row 1: ordinary full-field readout
# Row 2: gated readout
#
# Column 1: ΔI(ω)
# Column 2: heterodyne term
# Column 3: weak |Esig|² term
#
# Ordinary:
#     I(ω) = |FFT[Eout(t)]|²
#
# Gated:
#     Ig(ω) = |FFT[g(t) Eout(t)]|²
# ============================================================

# ---------- ordinary full-field readout ----------

I_probe_w = np.abs(Eprobe_w) ** 2
I_out_w = np.abs(Eout_w) ** 2

delta_I = I_out_w - I_probe_w

heterodyne_complex = np.conj(Eprobe_w) * Esig_w
heterodyne_real = 2.0 * np.real(heterodyne_complex)
heterodyne_abs = 2.0 * np.abs(heterodyne_complex)

sig2_w = np.abs(Esig_w) ** 2


# ---------- gated readout ----------

I_probe_g_w = np.abs(Eprobe_g_w) ** 2
I_out_g_w = np.abs(Eout_g_w) ** 2

delta_I_g = I_out_g_w - I_probe_g_w

heterodyne_g_complex = np.conj(Eprobe_g_w) * Esig_g_w
heterodyne_g_real = 2.0 * np.real(heterodyne_g_complex)
heterodyne_g_abs = 2.0 * np.abs(heterodyne_g_complex)

sig2_g_w = np.abs(Esig_g_w) ** 2


# ---------- combined figure ----------

fig3, axes = plt.subplots(2, 3, figsize=(17, 8), sharex="col")

# Row 1, Col 1: ordinary ΔI
axes[0, 0].plot(freq[mask_carrier], delta_I[mask_carrier])
axes[0, 0].set_title("Ordinary: ΔI(ω) = |Eout|² - |Eprobe|²")
axes[0, 0].set_ylabel("raw spectral intensity change")
sci_y(axes[0, 0])

# Row 1, Col 2: ordinary heterodyne
axes[0, 1].plot(
    freq[mask_carrier],
    heterodyne_real[mask_carrier],
    label="2 Re[Eprobe*(ω) Esig(ω)]",
)
axes[0, 1].plot(
    freq[mask_carrier],
    heterodyne_abs[mask_carrier],
    linestyle="--",
    label="2 |Eprobe*(ω) Esig(ω)|",
)
axes[0, 1].set_title("Ordinary: spectral heterodyne term")
axes[0, 1].set_ylabel("raw heterodyne signal")
axes[0, 1].legend()
sci_y(axes[0, 1])

# Row 1, Col 3: ordinary |Esig|²
axes[0, 2].plot(freq[mask_carrier], sig2_w[mask_carrier])
axes[0, 2].set_title("Ordinary: |Esig(ω)|²")
axes[0, 2].set_ylabel("raw weak-signal intensity")
sci_y(axes[0, 2])


# Row 2, Col 1: gated ΔI
axes[1, 0].plot(freq[mask_carrier], delta_I_g[mask_carrier])
axes[1, 0].set_title("Gated: ΔIg(ω) = |gEout|² - |gEprobe|²")
axes[1, 0].set_xlabel("frequency (cycles/fs)")
axes[1, 0].set_ylabel("raw spectral intensity change")
sci_y(axes[1, 0])

# Row 2, Col 2: gated heterodyne
axes[1, 1].plot(
    freq[mask_carrier],
    heterodyne_g_real[mask_carrier],
    label="2 Re[Eprobe_g*(ω) Esig_g(ω)]",
)
axes[1, 1].plot(
    freq[mask_carrier],
    heterodyne_g_abs[mask_carrier],
    linestyle="--",
    label="2 |Eprobe_g*(ω) Esig_g(ω)|",
)
axes[1, 1].set_title("Gated: spectral heterodyne term")
axes[1, 1].set_xlabel("frequency (cycles/fs)")
axes[1, 1].set_ylabel("raw heterodyne signal")
axes[1, 1].legend()
sci_y(axes[1, 1])

# Row 2, Col 3: gated |Esig|²
axes[1, 2].plot(freq[mask_carrier], sig2_g_w[mask_carrier])
axes[1, 2].set_title("Gated: |Esig_g(ω)|²")
axes[1, 2].set_xlabel("frequency (cycles/fs)")
axes[1, 2].set_ylabel("raw weak-signal intensity")
sci_y(axes[1, 2])

fig3.tight_layout()
fig3.savefig(
    OUTDIR / "figure3_ordinary_vs_gated_spectrometer_readout_raw_long_tail.png",
    dpi=300,
)


# ============================================================
# 7. Figure 4
# Ordinary readout versus gated readout
#
# Ordinary:
#     I(ω) = |FFT[Eout(t)]|²
#
# Gated:
#     Ig(ω) = |FFT[g(t) Eout(t)]|²
#
# This directly shows how cutting a long tail changes the spectrum,
# even when the gate is much wider than the probe pulse.
# ============================================================

I_probe_g_w = np.abs(Eprobe_g_w) ** 2
I_out_g_w = np.abs(Eout_g_w) ** 2

delta_I_g = I_out_g_w - I_probe_g_w

heterodyne_g_complex = np.conj(Eprobe_g_w) * Esig_g_w
heterodyne_g_real = 2.0 * np.real(heterodyne_g_complex)
heterodyne_g_abs = 2.0 * np.abs(heterodyne_g_complex)

sig2_g_w = np.abs(Esig_g_w) ** 2

fig4, axes = plt.subplots(2, 2, figsize=(14, 9))

# Panel 1: time gate and signal capture
axes[0, 0].plot(t, Esig_env, label="Esig envelope")
axes[0, 0].plot(t, gate * Esig_env, linestyle="--", label="g(t) Esig envelope")
axes[0, 0].plot(t, gate * np.max(Esig_env), linestyle=":", label="gate scaled")
axes[0, 0].axvline(gate_right, linestyle=":", linewidth=1)
axes[0, 0].set_xlim(-100, 1800)
axes[0, 0].set_title("Figure 4-1: time gate cuts a long tail")
axes[0, 0].set_xlabel("time (fs)")
axes[0, 0].set_ylabel("amplitude")
axes[0, 0].legend()
sci_y(axes[0, 0])

# Panel 2: delta I comparison
axes[0, 1].plot(freq[mask_carrier], delta_I[mask_carrier], label="ordinary full field")
axes[0, 1].plot(freq[mask_carrier], delta_I_g[mask_carrier],
                linestyle="--", label="gated field")
axes[0, 1].set_title("Figure 4-2: ΔI(ω), ordinary vs gated")
axes[0, 1].set_xlabel("frequency (cycles/fs)")
axes[0, 1].set_ylabel("raw spectral intensity change")
axes[0, 1].legend()
sci_y(axes[0, 1])

# Panel 3: heterodyne comparison
axes[1, 0].plot(freq[mask_carrier], heterodyne_real[mask_carrier],
                label="ordinary 2 Re[Eprobe* Esig]")
axes[1, 0].plot(freq[mask_carrier], heterodyne_g_real[mask_carrier],
                linestyle="--", label="gated 2 Re[Eprobe_g* Esig_g]")
axes[1, 0].plot(freq[mask_carrier], heterodyne_abs[mask_carrier],
                linestyle=":", label="ordinary 2 |Eprobe* Esig|")
axes[1, 0].plot(freq[mask_carrier], heterodyne_g_abs[mask_carrier],
                linestyle="-.", label="gated 2 |Eprobe_g* Esig_g|")
axes[1, 0].set_title("Figure 4-3: heterodyne term changed by gate")
axes[1, 0].set_xlabel("frequency (cycles/fs)")
axes[1, 0].set_ylabel("raw heterodyne signal")
axes[1, 0].legend()
sci_y(axes[1, 0])

# Panel 4: weak |Esig|^2 term comparison
axes[1, 1].plot(freq[mask_carrier], sig2_w[mask_carrier], label="ordinary |Esig(ω)|²")
axes[1, 1].plot(freq[mask_carrier], sig2_g_w[mask_carrier],
                linestyle="--", label="gated |Esig_g(ω)|²")
axes[1, 1].set_title("Figure 4-4: weak-signal term changed by gate")
axes[1, 1].set_xlabel("frequency (cycles/fs)")
axes[1, 1].set_ylabel("raw weak-signal intensity")
axes[1, 1].legend()
sci_y(axes[1, 1])

fig4.tight_layout()
fig4.savefig(OUTDIR / "figure4_gate_effect_raw_long_tail.png", dpi=300)


# ============================================================
# 8. Diagnostics
# ============================================================

# Time-domain overlap
time_overlap_abs = integrate_trapz(np.abs(Eprobe_r * Esig_r), t)
time_overlap_signed = integrate_trapz(Eprobe_r * Esig_r, t)

probe_energy_real = integrate_trapz(Eprobe_r ** 2, t)
sig_energy_real = integrate_trapz(Esig_r ** 2, t)

probe_energy_complex = integrate_trapz(np.abs(Eprobe) ** 2, t)
sig_energy_complex = integrate_trapz(np.abs(Esig) ** 2, t)
sig_g_energy_complex = integrate_trapz(np.abs(Esig_g) ** 2, t)

after_gate_mask = t > gate_right
if np.any(after_gate_mask):
    sig_after_gate_energy = integrate_trapz(np.abs(Esig[after_gate_mask]) ** 2, t[after_gate_mask])
else:
    sig_after_gate_energy = 0.0

# Spectral magnitudes
delta_I_size = np.max(np.abs(delta_I[mask_carrier]))
delta_I_g_size = np.max(np.abs(delta_I_g[mask_carrier]))

heterodyne_size = np.max(np.abs(heterodyne_real[mask_carrier]))
heterodyne_abs_size = np.max(np.abs(heterodyne_abs[mask_carrier]))
heterodyne_g_size = np.max(np.abs(heterodyne_g_real[mask_carrier]))
heterodyne_g_abs_size = np.max(np.abs(heterodyne_g_abs[mask_carrier]))

sig2_size = np.max(np.abs(sig2_w[mask_carrier]))
sig2_g_size = np.max(np.abs(sig2_g_w[mask_carrier]))

# Decomposition check
decomposition_error = np.max(np.abs(delta_I - (heterodyne_real + sig2_w)))
decomposition_scale = np.max(np.abs(delta_I)) if np.max(np.abs(delta_I)) != 0 else 1.0

print()
print("Saved figures:")
print(f"  {OUTDIR / 'figure1_time_freq_with_gate_long_tail.png'}")
print(f"  {OUTDIR / 'figure2_time_domain_square_law_products_long_tail.png'}")
print(f"  {OUTDIR / 'figure3_ordinary_vs_gated_spectrometer_readout_raw_long_tail.png'}")

print()
print("Basic parameters:")
print(f"  dt                                    = {dt:.3f} fs")
print(f"  time range                            = [{t_min:.1f}, {t_max:.1f}] fs")
print(f"  f0                                    = {f0:.6f} cycles/fs")
print(f"  probe_sigma                           = {probe_sigma:.3f} fs")
print(f"  probe FWHM                            = {2.355 * probe_sigma:.3f} fs")
print(f"  sig_start                             = {sig_start:.3f} fs")
print(f"  sig_tau                               = {sig_tau:.3f} fs")
print(f"  gate                                  = [{gate_left:.1f}, {gate_right:.1f}] fs")

print()
print("Time-domain diagnostics:")
print(f"  ∫ |Eprobe(t) Esig(t)| dt              = {time_overlap_abs:.3e}")
print(f"  ∫ Eprobe(t) Esig(t) dt                = {time_overlap_signed:.3e}")
print(f"  ∫ Re[Eprobe(t)]² dt                   = {probe_energy_real:.3e}")
print(f"  ∫ Re[Esig(t)]² dt                     = {sig_energy_real:.3e}")
print(f"  ∫ |Eprobe(t)|² dt                     = {probe_energy_complex:.3e}")
print(f"  ∫ |Esig(t)|² dt                       = {sig_energy_complex:.3e}")
print(f"  ∫ |g(t) Esig(t)|² dt                  = {sig_g_energy_complex:.3e}")
print(f"  gated Esig energy / total Esig energy = {safe_ratio(sig_g_energy_complex, sig_energy_complex):.3e}")
print(f"  ∫_after_gate |Esig(t)|² dt            = {sig_after_gate_energy:.3e}")
print(f"  after-gate Esig energy / total        = {safe_ratio(sig_after_gate_energy, sig_energy_complex):.3e}")

print()
print("Ordinary spectrometer readout, near carrier:")
print(f"  max |ΔI(ω)|                            = {delta_I_size:.3e}")
print(f"  max |2 Re[Eprobe*(ω) Esig(ω)]|         = {heterodyne_size:.3e}")
print(f"  max 2 |Eprobe*(ω) Esig(ω)|             = {heterodyne_abs_size:.3e}")
print(f"  max |Esig(ω)|²                         = {sig2_size:.3e}")
print(f"  heterodyne_abs / |Esig|²               = {safe_ratio(heterodyne_abs_size, sig2_size):.3e}")

print()
print("Gated spectrometer readout, near carrier:")
print(f"  max |ΔI_g(ω)|                          = {delta_I_g_size:.3e}")
print(f"  max |2 Re[Eprobe_g*(ω) Esig_g(ω)]|      = {heterodyne_g_size:.3e}")
print(f"  max 2 |Eprobe_g*(ω) Esig_g(ω)|          = {heterodyne_g_abs_size:.3e}")
print(f"  max |Esig_g(ω)|²                       = {sig2_g_size:.3e}")

print()
print("Gate effect ratios:")
print(f"  max |ΔI_g| / max |ΔI|                  = {safe_ratio(delta_I_g_size, delta_I_size):.3e}")
print(f"  gated heterodyne_abs / ordinary        = {safe_ratio(heterodyne_g_abs_size, heterodyne_abs_size):.3e}")
print(f"  gated |Esig|² / ordinary |Esig|²        = {safe_ratio(sig2_g_size, sig2_size):.3e}")

print()
print("Identity check:")
print("  ΔI(ω) should equal 2 Re[Eprobe*(ω) Esig(ω)] + |Esig(ω)|²")
print(f"  max absolute decomposition error       = {decomposition_error:.3e}")
print(f"  relative decomposition error           = {safe_ratio(decomposition_error, decomposition_scale):.3e}")