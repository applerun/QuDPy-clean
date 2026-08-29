"""Convenience builders for structured carrier-envelope fields."""

from __future__ import annotations

import warnings
from typing import Any, Iterable

import numpy as np

from ..field_series import FieldPhySeries
from ..lab_fields import _metadata_copy
from qudpy_sjh.utils.constants import EV_TO_FS_INV

from .carrier_envelope_field import CarrierEnvelopeField
from .carrier_spec import CarrierSpec
from .envelope_spec import ConstantEnvelopeSpec, EnvelopeSpec, GaussianEnvelopeSpec, SechEnvelopeSpec
from .multi_carrier_envelope_field import MultiCarrierEnvelopeField


def _validated_1d_array(values: Iterable[float], *, name: str) -> np.ndarray:
    array = np.asarray(tuple(values), dtype=float)
    if array.ndim != 1 or array.size < 2:
        raise ValueError(f"{name} must be a one-dimensional array with at least two values.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


def _default_validation_time(
    envelope: EnvelopeSpec,
    *,
    max_omega_fs_inv: float,
) -> np.ndarray:
    rates = tuple(
        float(rate)
        for rate in envelope.normalization_rate_candidates_fs_inv
        if np.isfinite(float(rate)) and float(rate) > 0.0
    )
    if not rates:
        raise ValueError(
            "normalization='peak_field' requires validation_time_fs when the envelope "
            "does not provide a finite characteristic rate."
    )
    half_width_fs = 8.0 / min(rates)
    center_fs = float(envelope.center_fs)
    max_omega = float(max_omega_fs_inv)
    max_dt_fs = np.pi / (8.0 * max_omega) if max_omega > 0.0 else 2.0 * half_width_fs
    sample_count = max(16385, int(np.ceil((2.0 * half_width_fs) / max_dt_fs)) + 1)
    if sample_count > 262145:
        raise ValueError(
            "The automatic validation grid would exceed 262145 samples. "
            "Provide an explicit uniformly sampled validation_time_fs grid."
        )
    if sample_count % 2 == 0:
        sample_count += 1
    return np.linspace(
        center_fs - half_width_fs,
        center_fs + half_width_fs,
        sample_count,
    )


def _validated_fft_time(
    values: Iterable[float] | None,
    *,
    envelope: EnvelopeSpec,
    max_omega_fs_inv: float,
) -> np.ndarray:
    if values is None:
        return _default_validation_time(envelope, max_omega_fs_inv=max_omega_fs_inv)
    time_fs = _validated_1d_array(values, name="validation_time_fs")
    if time_fs.size < 16:
        raise ValueError("validation_time_fs must contain at least 16 samples.")
    dt = np.diff(time_fs)
    if np.any(dt <= 0.0):
        raise ValueError("validation_time_fs must be strictly increasing.")
    if not np.allclose(dt, dt[0], rtol=1.0e-7, atol=1.0e-12):
        raise ValueError("validation_time_fs must be uniformly sampled for FFT diagnostics.")
    return time_fs


def _normalized_rms(candidate: np.ndarray, reference: np.ndarray) -> float:
    denominator = float(np.sqrt(np.mean(np.abs(reference) ** 2)))
    if denominator == 0.0:
        return 0.0 if np.allclose(candidate, 0.0) else float("inf")
    return float(np.sqrt(np.mean(np.abs(candidate - reference) ** 2)) / denominator)


def _build_diagnostics(
    *,
    field: MultiCarrierEnvelopeField,
    time_fs: np.ndarray,
    target_omega_fs_inv: np.ndarray,
    target_amplitude: np.ndarray,
) -> dict[str, Any]:
    positive_field = np.asarray(field.positive_frequency_E_MV_per_cm(time_fs), dtype=complex)
    requested_envelope = np.abs(np.asarray(field.envelope.value(time_fs), dtype=float))
    actual_envelope = np.abs(positive_field)
    requested_envelope = requested_envelope / float(np.max(requested_envelope))
    actual_envelope = actual_envelope / float(np.max(actual_envelope))

    dt_fs = float(time_fs[1] - time_fs[0])
    n_fft_target = max(4 * time_fs.size, time_fs.size)
    n_fft = 1 << int(np.ceil(np.log2(n_fft_target)))
    fft_values = np.fft.fft(positive_field, n=n_fft)
    omega_fs_inv = 2.0 * np.pi * np.fft.fftfreq(n_fft, d=dt_fs)
    positive = omega_fs_inv >= 0.0
    reconstructed = np.interp(
        target_omega_fs_inv,
        omega_fs_inv[positive],
        np.abs(fft_values[positive]),
    )
    reconstructed = reconstructed / float(np.max(reconstructed))
    normalized_target = target_amplitude / float(np.max(target_amplitude))

    return {
        "spectral_normalized_rms_error": _normalized_rms(reconstructed, normalized_target),
        "temporal_normalized_rms_error": _normalized_rms(actual_envelope, requested_envelope),
        "validation_time_min_fs": float(time_fs[0]),
        "validation_time_max_fs": float(time_fs[-1]),
        "validation_time_count": int(time_fs.size),
        "fft_size": int(n_fft),
        "fourier_consistency_note": (
            "The requested spectrum and shared temporal envelope are not independent. "
            "Errors quantify the finite-carrier field actually produced; the builder does not force-fit either target."
        ),
    }


def make_multi_carrier_field_from_spectrum(
    *,
    spectrum_axis: Iterable[float],
    spectrum_values: Iterable[float],
    envelope: EnvelopeSpec,
    carrier_count: int,
    E0_MV_per_cm: float,
    spectrum_axis_kind: str = "energy_eV",
    spectrum_values_kind: str = "intensity",
    spectral_phase_rad: Iterable[float] | None = None,
    spectral_support: tuple[float, float] | None = None,
    normalization: str = "peak_field",
    global_phase_rad: float = 0.0,
    validation_time_fs: Iterable[float] | None = None,
    diagnostics_warning_threshold: float = 0.25,
    name: str = "multi_carrier_field_from_spectrum",
    metadata: dict[str, Any] | None = None,
) -> MultiCarrierEnvelopeField:
    """Build a shared-envelope multi-carrier approximation to a target spectrum.

    The input spectrum may be arbitrarily sampled. Carrier locations are uniform
    over ``spectral_support`` (or the complete input axis when omitted).
    ``spectrum_values_kind`` explicitly distinguishes intensity from field
    amplitude; intensity values are square-rooted before interpolation.

    ``normalization='peak_field'`` scales the dimensionless carrier weights so
    that the numerical peak of ``abs(E_positive(t))`` on the validation grid is
    ``abs(E0_MV_per_cm)``. The field keeps one shared ``EnvelopeSpec`` and one
    global experimental phase; intrinsic spectral phases remain per carrier.
    """

    if not isinstance(envelope, EnvelopeSpec):
        raise TypeError("envelope must be an EnvelopeSpec instance.")
    count = int(carrier_count)
    if count < 1 or count != carrier_count:
        raise ValueError("carrier_count must be a positive integer.")
    if not np.isfinite(float(E0_MV_per_cm)):
        raise ValueError("E0_MV_per_cm must be finite.")
    if not np.isfinite(float(global_phase_rad)):
        raise ValueError("global_phase_rad must be finite.")
    threshold = float(diagnostics_warning_threshold)
    if not np.isfinite(threshold) or threshold < 0.0:
        raise ValueError("diagnostics_warning_threshold must be finite and non-negative.")

    axis = _validated_1d_array(spectrum_axis, name="spectrum_axis")
    values = _validated_1d_array(spectrum_values, name="spectrum_values")
    if values.size != axis.size:
        raise ValueError("spectrum_values must have the same length as spectrum_axis.")
    if np.any(values < 0.0):
        raise ValueError("spectrum_values must be non-negative.")

    axis_kind = str(spectrum_axis_kind).strip()
    if axis_kind not in {"energy_eV", "omega_fs_inv"}:
        raise ValueError("spectrum_axis_kind must be 'energy_eV' or 'omega_fs_inv'.")
    values_kind = str(spectrum_values_kind).strip()
    if values_kind == "intensity":
        field_amplitude = np.sqrt(values)
    elif values_kind == "field_amplitude":
        field_amplitude = values.copy()
    else:
        raise ValueError("spectrum_values_kind must be 'intensity' or 'field_amplitude'.")
    if not np.any(field_amplitude > 0.0):
        raise ValueError("spectrum_values must contain at least one positive value.")

    if spectral_phase_rad is None:
        phase = np.zeros_like(axis)
    else:
        phase = _validated_1d_array(spectral_phase_rad, name="spectral_phase_rad")
        if phase.size != axis.size:
            raise ValueError("spectral_phase_rad must have the same length as spectrum_axis.")

    order = np.argsort(axis)
    axis = axis[order]
    field_amplitude = field_amplitude[order]
    phase = np.unwrap(phase[order])
    if np.any(np.diff(axis) <= 0.0):
        raise ValueError("spectrum_axis values must be unique.")

    if spectral_support is None:
        support_min = float(axis[0])
        support_max = float(axis[-1])
    else:
        support_min, support_max = (float(x) for x in spectral_support)
        if not np.isfinite(support_min) or not np.isfinite(support_max):
            raise ValueError("spectral_support values must be finite.")
        if support_min >= support_max:
            raise ValueError("spectral_support must satisfy min < max.")
        if support_min < axis[0] or support_max > axis[-1]:
            raise ValueError("spectral_support must lie within spectrum_axis.")

    sampled_axis = (
        np.array([(support_min + support_max) / 2.0])
        if count == 1
        else np.linspace(support_min, support_max, count)
    )
    raw_weights = np.interp(sampled_axis, axis, field_amplitude)
    sampled_phase = np.interp(sampled_axis, axis, phase)
    if not np.any(raw_weights > 0.0):
        raise ValueError("The selected spectral support contains no positive field amplitude.")

    if axis_kind == "energy_eV":
        sampled_energy_eV = sampled_axis
        sampled_omega_fs_inv = sampled_axis * EV_TO_FS_INV
        target_omega_fs_inv = axis * EV_TO_FS_INV
    else:
        sampled_omega_fs_inv = sampled_axis
        sampled_energy_eV = sampled_axis / EV_TO_FS_INV
        target_omega_fs_inv = axis

    carriers = tuple(
        CarrierSpec(
            omega_fs_inv=float(omega),
            phase_rad=float(component_phase),
            label=f"carrier_{index}",
            metadata={
                "laser_energy_eV": float(energy),
                "sampled_spectrum_axis_value": float(axis_value),
                "spectrum_axis_kind": axis_kind,
            },
        )
        for index, (omega, energy, axis_value, component_phase) in enumerate(
            zip(sampled_omega_fs_inv, sampled_energy_eV, sampled_axis, sampled_phase)
        )
    )

    if normalization != "peak_field":
        raise ValueError("normalization must be 'peak_field'.")
    time_fs = _validated_fft_time(
        validation_time_fs,
        envelope=envelope,
        max_omega_fs_inv=float(np.max(sampled_omega_fs_inv)),
    )
    raw_field = MultiCarrierEnvelopeField.from_carriers(
        E0_MV_per_cm=1.0,
        carriers=carriers,
        amplitudes=raw_weights,
        envelope=envelope,
        global_phase_rad=0.0,
    )
    raw_peak = float(np.max(np.abs(raw_field.positive_frequency_E_MV_per_cm(time_fs))))
    if not np.isfinite(raw_peak) or raw_peak <= 0.0:
        raise ValueError("Unable to normalize a field with zero or non-finite peak amplitude.")
    applied_scaling_factor = 1.0 / raw_peak
    normalized_weights = raw_weights * applied_scaling_factor

    builder_metadata = {
        "builder": "make_multi_carrier_field_from_spectrum",
        "spectrum_axis_kind": axis_kind,
        "spectrum_values_kind": values_kind,
        "carrier_sampling": "uniform over spectral_support",
        "carrier_count": count,
        "spectral_support": [support_min, support_max],
        "normalization": normalization,
        "raw_weights": raw_weights.tolist(),
        "applied_scaling_factor": float(applied_scaling_factor),
        "normalized_weights": normalized_weights.tolist(),
        "sampled_spectrum_axis": sampled_axis.tolist(),
        "sampled_energy_eV": sampled_energy_eV.tolist(),
        "sampled_omega_fs_inv": sampled_omega_fs_inv.tolist(),
        "sampled_spectral_phase_rad": sampled_phase.tolist(),
    }
    diagnostic_field = MultiCarrierEnvelopeField.from_carriers(
        E0_MV_per_cm=float(E0_MV_per_cm),
        carriers=carriers,
        amplitudes=normalized_weights,
        envelope=envelope,
        global_phase_rad=float(global_phase_rad),
    )
    diagnostic_mask = (axis >= support_min) & (axis <= support_max)
    diagnostics = _build_diagnostics(
        field=diagnostic_field,
        time_fs=time_fs,
        target_omega_fs_inv=target_omega_fs_inv[diagnostic_mask],
        target_amplitude=field_amplitude[diagnostic_mask],
    )
    builder_metadata["diagnostics"] = diagnostics
    if max(
        diagnostics["spectral_normalized_rms_error"],
        diagnostics["temporal_normalized_rms_error"],
    ) > threshold:
        warnings.warn(
            "The requested spectrum and temporal envelope are not closely reproduced by "
            f"the finite shared-envelope field (spectral error={diagnostics['spectral_normalized_rms_error']:.3g}, "
            f"temporal error={diagnostics['temporal_normalized_rms_error']:.3g}).",
            UserWarning,
            stacklevel=2,
        )
    field_metadata = _metadata_copy(metadata)
    field_metadata["spectrum_builder"] = builder_metadata
    return MultiCarrierEnvelopeField.from_carriers(
        E0_MV_per_cm=float(E0_MV_per_cm),
        carriers=carriers,
        amplitudes=normalized_weights,
        envelope=envelope,
        global_phase_rad=float(global_phase_rad),
        name=name,
        metadata=field_metadata,
    )


def make_carrier_envelope_field(
    *,
    E0_MV_per_cm: float,
    carrier: CarrierSpec,
    envelope,
    name: str = "carrier_envelope_field",
    metadata: dict[str, Any] | None = None,
) -> CarrierEnvelopeField:
    return CarrierEnvelopeField(
        E0_MV_per_cm=float(E0_MV_per_cm),
        carrier=carrier,
        envelope=envelope,
        name=name,
        metadata=_metadata_copy(metadata),
    )


def make_gaussian_carrier_envelope_field(
    *,
    E0_MV_per_cm: float,
    laser_energy_eV: float,
    center_fs: float,
    sigma_fs: float,
    phase_rad: float = 0.0,
    envelope_amplitude: float = 1.0,
    name: str = "gaussian_carrier_envelope_field",
    metadata: dict[str, Any] | None = None,
) -> CarrierEnvelopeField:
    return CarrierEnvelopeField(
        E0_MV_per_cm=float(E0_MV_per_cm),
        carrier=CarrierSpec.from_energy_eV(
            float(laser_energy_eV),
            phase_rad=float(phase_rad),
            metadata={"laser_energy_eV": float(laser_energy_eV)},
        ),
        envelope=GaussianEnvelopeSpec(
            center_fs=float(center_fs),
            sigma_fs=float(sigma_fs),
            amplitude=float(envelope_amplitude),
        ),
        name=name,
        metadata=_metadata_copy(metadata),
    )


def make_sech_carrier_envelope_field(
    *,
    E0_MV_per_cm: float,
    laser_energy_eV: float,
    center_fs: float,
    tau_fs: float,
    phase_rad: float = 0.0,
    envelope_amplitude: float = 1.0,
    name: str = "sech_carrier_envelope_field",
    metadata: dict[str, Any] | None = None,
) -> CarrierEnvelopeField:
    return CarrierEnvelopeField(
        E0_MV_per_cm=float(E0_MV_per_cm),
        carrier=CarrierSpec.from_energy_eV(
            float(laser_energy_eV),
            phase_rad=float(phase_rad),
            metadata={"laser_energy_eV": float(laser_energy_eV)},
        ),
        envelope = SechEnvelopeSpec(
            width_fs = float(tau_fs),
            center_fs = float(center_fs),
            amplitude = float(envelope_amplitude),
        ),
        name=name,
        metadata=_metadata_copy(metadata),
    )


def make_constant_carrier_envelope_field(
    *,
    E0_MV_per_cm: float,
    laser_energy_eV: float,
    phase_rad: float = 0.0,
    center_fs: float = 0.0,
    envelope_amplitude: float = 1.0,
    name: str = "constant_carrier_envelope_field",
    metadata: dict[str, Any] | None = None,
) -> CarrierEnvelopeField:
    return CarrierEnvelopeField(
        E0_MV_per_cm=float(E0_MV_per_cm),
        carrier=CarrierSpec.from_energy_eV(
            float(laser_energy_eV),
            phase_rad=float(phase_rad),
            metadata={"laser_energy_eV": float(laser_energy_eV)},
        ),
        envelope=ConstantEnvelopeSpec(
            center_fs=float(center_fs),
            amplitude=float(envelope_amplitude),
        ),
        name=name,
        metadata=_metadata_copy(metadata),
    )


def make_pump_probe_field_series(
    *,
    pump_field: CarrierEnvelopeField,
    probe_field: CarrierEnvelopeField,
    name: str = "pump_probe_field_series",
    metadata: dict[str, Any] | None = None,
) -> FieldPhySeries:
    """Create a pump+probe linear field series.

    Pump/probe role is represented here at the series/case level, not inside
    CarrierEnvelopeField itself.
    """

    return FieldPhySeries(
        fields=(pump_field, probe_field),
        sub_field_names=("pump", "probe"),
        name=name,
        metadata=_metadata_copy(metadata),
    )


__all__ = [
    "make_carrier_envelope_field",
    "make_multi_carrier_field_from_spectrum",
    "make_gaussian_carrier_envelope_field",
    "make_sech_carrier_envelope_field",
    "make_constant_carrier_envelope_field",
    "make_pump_probe_field_series",
]
