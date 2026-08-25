"""Reusable polarization and detector readout operations.

The canonical boundary is::

    DynamicsResult -> PolarizationResult -> ReadoutPlan -> ReadoutResult

Readout fields may reference a named interaction subfield or be supplied as an
external ``FieldPhyRoot``.  External fields are sampled only during readout and
never enter the Hamiltonian.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field as dataclass_field
from typing import Any

import numpy as np

from qudpy_sjh.utils.core import DynamicsResult, ParaNormalizer
from qudpy_sjh.utils.fields import FieldPhyRoot
from qudpy_sjh.utils.spectroscopy import (
    apply_time_window,
    diagnose_uniform_time_axis,
    lab_frame_absorption_response,
    polarization_C_per_m2,
)


_READOUT_MODES = {"polarization", "absorption_like", "full", "weak"}
_DETECTOR_MODES = {"full", "weak"}
_WINDOWS = {None, "none", "hann"}


def _copy_metadata(metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    return dict(metadata or {})


def _json_array(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_array(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_array(item) for item in value]
    if isinstance(value, complex):
        return {"real": float(value.real), "imag": float(value.imag)}
    return value


def _array_range(array: np.ndarray) -> tuple[float, float] | None:
    values = np.asarray(array, dtype=float)
    if values.size == 0:
        return None
    return float(np.min(values)), float(np.max(values))


def _validate_field_reference(value: str) -> str:
    name = str(value).strip()
    if not name:
        raise ValueError("readout field reference must not be empty.")
    return name


@dataclass(frozen=True)
class PolarizationResult:
    """Physical polarization trajectory derived from one dynamics result."""

    time_fs: np.ndarray
    polarization_C_per_m2: np.ndarray
    metadata: dict[str, Any] = dataclass_field(default_factory=dict)

    def __post_init__(self) -> None:
        time = np.asarray(self.time_fs, dtype=float)
        polarization = np.asarray(self.polarization_C_per_m2, dtype=np.complex128)
        if time.ndim != 1 or polarization.ndim != 1:
            raise ValueError("time_fs and polarization_C_per_m2 must be one-dimensional.")
        if time.shape != polarization.shape:
            raise ValueError("time_fs and polarization_C_per_m2 must have matching shapes.")
        if time.size < 2:
            raise ValueError("PolarizationResult requires at least two time samples.")
        if not np.all(np.isfinite(time)):
            raise ValueError("time_fs must contain only finite values.")
        object.__setattr__(self, "time_fs", time)
        object.__setattr__(self, "polarization_C_per_m2", polarization)
        object.__setattr__(self, "metadata", _copy_metadata(self.metadata))

    def to_dict(self, *, include_arrays: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "class": self.__class__.__name__,
            "n_time_points": int(self.time_fs.size),
            "time_range_fs": _array_range(self.time_fs),
            "max_abs_polarization_C_per_m2": float(np.max(np.abs(self.polarization_C_per_m2))),
            "metadata": dict(self.metadata),
        }
        if include_arrays:
            payload["time_fs"] = self.time_fs.tolist()
            payload["polarization_C_per_m2"] = _json_array(self.polarization_C_per_m2)
        return payload


def compute_polarization_result(
    result: DynamicsResult,
    *,
    number_density_m3: float,
) -> PolarizationResult:
    """Convert saved dynamics into physical ``P(t)`` without executing a solver."""

    if not isinstance(result, DynamicsResult):
        raise TypeError("result must be a DynamicsResult instance.")
    density = float(number_density_m3)
    if density <= 0.0:
        raise ValueError("number_density_m3 must be > 0.")
    if result.physical_params is None:
        raise ValueError("DynamicsResult.physical_params is required to compute polarization.")
    if result.times_fs is None:
        raise ValueError("DynamicsResult.times_fs is required to compute polarization.")
    polarization = polarization_C_per_m2(
        result.density_array(),
        result.physical_params.dipole_matrix_D,
        density,
    )
    return PolarizationResult(
        time_fs=np.asarray(result.times_fs, dtype=float),
        polarization_C_per_m2=polarization,
        metadata={
            "source": "DynamicsResult density trajectory and physical dipole matrix",
            "number_density_m3": density,
        },
    )


def select_interaction_readout_field(
    interaction_field: FieldPhyRoot,
    field_name: str | None,
) -> FieldPhyRoot:
    """Resolve the total interaction field or one named interaction subfield."""

    if not isinstance(interaction_field, FieldPhyRoot):
        raise TypeError("interaction_field must be a FieldPhyRoot instance.")
    if field_name is None:
        return interaction_field
    name = _validate_field_reference(field_name)
    try:
        selected = interaction_field[name]  # type: ignore[index]
    except KeyError as exc:
        raise KeyError(f"readout field reference {name!r} was not found in the interaction field.") from exc
    except TypeError as exc:
        raise TypeError(
            f"readout field reference {name!r} requires an interaction-field container "
            "with named subfields."
        ) from exc
    if not isinstance(selected, FieldPhyRoot):
        raise TypeError("resolved readout field must be a FieldPhyRoot instance.")
    return selected


def resolve_readout_field(
    readout_field: FieldPhyRoot | str | None,
    *,
    interaction_field: FieldPhyRoot | None,
) -> tuple[FieldPhyRoot, str]:
    """Resolve one unambiguous readout-field source.

    ``FieldPhyRoot`` means an external/direct field, ``str`` means a named
    interaction subfield, and ``None`` means the total interaction field.
    """

    if isinstance(readout_field, FieldPhyRoot):
        return readout_field, "external_field"
    if isinstance(readout_field, str):
        if interaction_field is None:
            raise ValueError("interaction_field is required for a named readout field reference.")
        name = _validate_field_reference(readout_field)
        return select_interaction_readout_field(interaction_field, name), f"interaction_subfield:{name}"
    if readout_field is not None:
        raise TypeError("readout_field must be a FieldPhyRoot, str, or None.")
    if interaction_field is None:
        raise ValueError("interaction_field is required when readout_field=None.")
    return select_interaction_readout_field(interaction_field, None), "total_interaction_field"


def coherent_detector_terms(
    readout_field_omega: np.ndarray,
    signal_field_omega: np.ndarray,
    *,
    mode: str,
) -> dict[str, np.ndarray]:
    """Evaluate full or weak coherent-detector algebra on aligned spectra."""

    detector_mode = str(mode).strip()
    if detector_mode not in _DETECTOR_MODES:
        raise ValueError("mode must be 'full' or 'weak'.")
    readout = np.asarray(readout_field_omega, dtype=np.complex128)
    signal = np.asarray(signal_field_omega, dtype=np.complex128)
    if readout.shape != signal.shape:
        raise ValueError("readout_field_omega and signal_field_omega must have matching shapes.")
    readout_intensity = np.abs(readout) ** 2
    interference = 2.0 * np.real(np.conjugate(readout) * signal)
    signal_intensity = np.abs(signal) ** 2
    detector_intensity = readout_intensity + interference
    if detector_mode == "full":
        detector_intensity = detector_intensity + signal_intensity
    return {
        "detector_intensity": detector_intensity,
        "readout_intensity": readout_intensity,
        "interference_term": interference,
        "signal_intensity": signal_intensity,
    }


@dataclass
class ReadoutResult:
    """Lightweight detector/readout output, independent of solver state."""

    mode: str
    time_fs: np.ndarray | None = None
    polarization_C_per_m2: np.ndarray | None = None
    readout_field_MV_per_cm: np.ndarray | None = None
    spectrum: dict[str, Any] | None = None
    metadata: dict[str, Any] = dataclass_field(default_factory=dict)

    def to_dict(self, *, include_arrays: bool = False) -> dict[str, Any]:
        polarization = None if self.polarization_C_per_m2 is None else np.asarray(self.polarization_C_per_m2)
        readout_field = None if self.readout_field_MV_per_cm is None else np.asarray(self.readout_field_MV_per_cm)
        time = None if self.time_fs is None else np.asarray(self.time_fs, dtype=float)
        spectrum = self.spectrum or {}
        energy = np.asarray(spectrum.get("energy_eV", []), dtype=float)
        payload: dict[str, Any] = {
            "class": self.__class__.__name__,
            "mode": self.mode,
            "n_time_points": None if time is None else int(time.size),
            "max_abs_polarization": None if polarization is None else float(np.max(np.abs(polarization))),
            "max_abs_readout_field_MV_per_cm": (
                None if readout_field is None else float(np.max(np.abs(readout_field)))
            ),
            "readout_field_name": self.metadata.get("readout_field_name"),
            "readout_field_source": self.metadata.get("readout_field_source"),
            "spectrum": {
                "n_points": int(energy.size),
                "energy_range_eV": _array_range(energy),
            } if self.spectrum is not None else None,
            "metadata": dict(self.metadata),
        }
        if include_arrays:
            payload["time_fs"] = None if time is None else time.tolist()
            payload["polarization_C_per_m2"] = None if polarization is None else _json_array(polarization)
            payload["readout_field_MV_per_cm"] = None if readout_field is None else readout_field.tolist()
            payload["spectrum_full"] = _json_array(self.spectrum) if self.spectrum is not None else None
        return payload


@dataclass(frozen=True)
class ReadoutPlan:
    """Executable polarization-to-observable plan.

    ``readout_field`` has one union-like meaning: a ``FieldPhyRoot`` is an
    external/direct field, a string references a named interaction subfield,
    and ``None`` selects the total interaction field.  Polarization mode does
    not use a readout field.
    """

    mode: str
    readout_field: FieldPhyRoot | str | None = None
    window: str | None = "hann"
    subtract_mean: bool = True
    rel_threshold: float = 1.0e-6
    zero_padding_factor: int = 4
    emitted_field_scale: float = 1.0
    return_intermediates: bool = True
    metadata: dict[str, Any] = dataclass_field(default_factory=dict)

    def __post_init__(self) -> None:
        mode = str(self.mode).strip()
        if mode not in _READOUT_MODES:
            raise ValueError(f"Unsupported readout mode: {self.mode!r}. Expected one of {sorted(_READOUT_MODES)}.")
        if self.window not in _WINDOWS:
            raise ValueError("window must be None, 'none', or 'hann'.")
        if float(self.rel_threshold) <= 0.0:
            raise ValueError("rel_threshold must be > 0.")
        if self.zero_padding_factor != int(self.zero_padding_factor) or int(self.zero_padding_factor) < 1:
            raise ValueError("zero_padding_factor must be a positive integer.")
        scale = float(self.emitted_field_scale)
        if not np.isfinite(scale):
            raise ValueError("emitted_field_scale must be finite.")
        readout_field = self.readout_field
        if isinstance(readout_field, str):
            readout_field = _validate_field_reference(readout_field)
        elif readout_field is not None and not isinstance(readout_field, FieldPhyRoot):
            raise TypeError("readout_field must be a FieldPhyRoot, str, or None.")
        if mode == "polarization" and readout_field is not None:
            raise ValueError("polarization mode does not use readout_field.")
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "readout_field", readout_field)
        object.__setattr__(self, "subtract_mean", bool(self.subtract_mean))
        object.__setattr__(self, "rel_threshold", float(self.rel_threshold))
        object.__setattr__(self, "zero_padding_factor", int(self.zero_padding_factor))
        object.__setattr__(self, "emitted_field_scale", scale)
        object.__setattr__(self, "return_intermediates", bool(self.return_intermediates))
        object.__setattr__(self, "metadata", _copy_metadata(self.metadata))

    def _field_metadata(self) -> dict[str, Any]:
        if isinstance(self.readout_field, FieldPhyRoot):
            return {"source": "external_field", "field": self.readout_field.to_dict()}
        if isinstance(self.readout_field, str):
            return {"source": "interaction_subfield", "field_name": self.readout_field}
        return {"source": "total_interaction_field"}

    def to_dict(self) -> dict[str, Any]:
        return {
            "class": self.__class__.__name__,
            "mode": self.mode,
            "readout_field": None if self.mode == "polarization" else self._field_metadata(),
            "window": self.window,
            "subtract_mean": bool(self.subtract_mean),
            "rel_threshold": float(self.rel_threshold),
            "zero_padding_factor": int(self.zero_padding_factor),
            "emitted_field_scale": float(self.emitted_field_scale),
            "return_intermediates": bool(self.return_intermediates),
            "metadata": dict(self.metadata),
        }

    def execute(
        self,
        polarization: PolarizationResult,
        *,
        interaction_field: FieldPhyRoot | None = None,
    ) -> ReadoutResult:
        """Execute detector physics without invoking or owning a solver."""

        if not isinstance(polarization, PolarizationResult):
            raise TypeError("polarization must be a PolarizationResult instance.")
        metadata = dict(self.metadata)
        metadata.update(
            {
                "readout_plan": self.to_dict(),
                "readout_field_name": self.readout_field if isinstance(self.readout_field, str) else None,
                "readout_field_source": None,
            }
        )
        if self.mode == "polarization":
            return ReadoutResult(
                mode=self.mode,
                time_fs=polarization.time_fs,
                polarization_C_per_m2=polarization.polarization_C_per_m2,
                metadata=metadata,
            )

        field, source = resolve_readout_field(
            self.readout_field,
            interaction_field=interaction_field,
        )
        field_values = np.asarray(field(polarization.time_fs), dtype=float)
        if field_values.shape != polarization.time_fs.shape:
            raise ValueError("readout field values must align with the polarization time axis.")
        metadata["readout_field_source"] = source

        if self.mode == "absorption_like":
            response = lab_frame_absorption_response(
                time_fs=polarization.time_fs,
                polarization_C_per_m2=polarization.polarization_C_per_m2,
                field=field_values,
                window=self.window,
                subtract_mean=self.subtract_mean,
                rel_threshold=self.rel_threshold,
                zero_padding_factor=self.zero_padding_factor,
                return_intermediates=self.return_intermediates,
            )
            response["absorption_like_response"] = np.asarray(response["absorption"])
            response["metadata"]["canonical_quantity"] = "absorption_like_response"
            return ReadoutResult(
                mode=self.mode,
                time_fs=polarization.time_fs,
                polarization_C_per_m2=polarization.polarization_C_per_m2,
                readout_field_MV_per_cm=field_values,
                spectrum=response,
                metadata=metadata,
            )

        spectrum = self._coherent_detector_spectrum(polarization, field_values)
        return ReadoutResult(
            mode=self.mode,
            time_fs=polarization.time_fs,
            polarization_C_per_m2=polarization.polarization_C_per_m2,
            readout_field_MV_per_cm=field_values,
            spectrum=spectrum,
            metadata=metadata,
        )

    def _coherent_detector_spectrum(
        self,
        polarization: PolarizationResult,
        field_values: np.ndarray,
    ) -> dict[str, Any]:
        diagnostics = diagnose_uniform_time_axis(polarization.time_fs)
        if not diagnostics["is_uniform"]:
            raise ValueError(
                "coherent detector readout requires a uniformly sampled time axis. "
                f"time_axis={diagnostics}"
            )
        dt_fs = float(diagnostics["median_dt_fs"])
        readout_signal = np.asarray(field_values, dtype=np.complex128)
        polarization_signal = np.asarray(polarization.polarization_C_per_m2, dtype=np.complex128)
        if self.subtract_mean:
            readout_signal = readout_signal - np.mean(readout_signal)
            polarization_signal = polarization_signal - np.mean(polarization_signal)
        n_samples = polarization.time_fs.size
        n_fft_target = int(n_samples * self.zero_padding_factor)
        n_fft = 1 << int(np.ceil(np.log2(max(n_fft_target, n_samples))))
        readout_fft = np.fft.fft(apply_time_window(readout_signal, self.window), n=n_fft)
        polarization_fft = np.fft.fft(apply_time_window(polarization_signal, self.window), n=n_fft)
        frequency_fs_inv = np.fft.fftfreq(n_fft, d=dt_fs)
        omega_fs_inv = 2.0 * np.pi * frequency_fs_inv
        positive = frequency_fs_inv > 0.0
        omega = omega_fs_inv[positive]
        readout_omega = readout_fft[positive]
        polarization_omega = polarization_fft[positive]
        signal_omega = 1j * self.emitted_field_scale * omega * polarization_omega
        terms = coherent_detector_terms(readout_omega, signal_omega, mode=self.mode)
        response: dict[str, Any] = {
            "omega_fs_inv": omega,
            "energy_eV": omega / ParaNormalizer.EV_TO_FS_INV,
            **terms,
            "metadata": {
                "time_axis": diagnostics,
                "emitted_field_definition": "E_signal(omega) = i * emitted_field_scale * omega * P(omega)",
                "detector_definition": (
                    "I = |E_readout + E_signal|^2"
                    if self.mode == "full"
                    else "I_weak = |E_readout|^2 + 2 Re[conj(E_readout) E_signal]"
                ),
                "fft_convention": "numpy fft: exp(-2*pi*i*k*n/N); positive frequencies retained",
                "spectrum_alignment": "P(t) and readout field sampled on the same time_fs grid; no interpolation",
            },
        }
        if self.return_intermediates:
            response.update(
                {
                    "E_readout_omega": readout_omega,
                    "P_omega": polarization_omega,
                    "E_signal_omega": signal_omega,
                }
            )
        return response


__all__ = [
    "PolarizationResult",
    "ReadoutPlan",
    "ReadoutResult",
    "coherent_detector_terms",
    "compute_polarization_result",
    "resolve_readout_field",
    "select_interaction_readout_field",
]
