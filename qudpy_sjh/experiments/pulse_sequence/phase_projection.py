"""Pure named-axis Fourier projection over precomputed phase-dependent data.

This module owns phase-domain validation and Fourier mathematics only.  It does
not import or execute solvers, single-run plans, readout plans, or experiment
recipes.  Explicit nonuniform phase values are accepted for an equal-weight
phase sum; that operation is not a general nonuniform Fourier inversion.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from itertools import product
from typing import Any

import numpy as np

from qudpy_sjh.experiments.pulse_sequence.pulse_sequence import validate_phase_tag


PhaseVector = dict[str, float]
TargetPhaseVector = dict[str, int]

PHASE_PROJECTION_CONVENTION = "exp_plus_i_m_phi"
PHASE_PROJECTION_CONVENTION_VERSION = 1
TARGET_PHASE_VECTOR_SEMANTICS = "physical_phase_order_vector_m"
_LEGACY_PHASE_PROJECTION_CONVENTION = "legacy_exp_minus_i_target_phi"


def _stable_unique_phase_tags(tags: Sequence[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for tag in tags:
        text = validate_phase_tag(tag, allow_none=False)
        assert text is not None
        if text in seen:
            continue
        seen.add(text)
        ordered.append(text)
    return tuple(ordered)


def _integer_coefficient(value: int | float, *, tag: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"target coefficient for {tag!r} must be an integer, not bool.")
    try:
        coefficient = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"target coefficient for {tag!r} must be an integer. Got {value!r}.") from exc
    if not np.isfinite(coefficient):
        raise ValueError(f"target coefficient for {tag!r} must be finite.")
    rounded = int(round(coefficient))
    if not np.isclose(coefficient, rounded, rtol=0.0, atol=1.0e-12):
        raise ValueError(f"target coefficient for {tag!r} must be an integer. Got {value!r}.")
    return rounded


def normalize_target_phase_vector(
    target_phase_vector: Mapping[str, int | float],
    *,
    known_tags: Sequence[str] | None = None,
    fill_missing_with_zero: bool = True,
) -> TargetPhaseVector:
    """Normalize a physical integer phase-order vector ``m``."""

    if not isinstance(target_phase_vector, Mapping):
        raise TypeError("target_phase_vector must be a mapping from phase tag to integer order.")
    data: TargetPhaseVector = {}
    for key, value in target_phase_vector.items():
        tag = validate_phase_tag(key, allow_none=False)
        assert tag is not None
        data[tag] = _integer_coefficient(value, tag=tag)

    if known_tags is None:
        return data

    tags = _stable_unique_phase_tags(tuple(known_tags))
    known = set(tags)
    unknown = sorted(set(data) - known)
    if unknown:
        raise ValueError(f"target_phase_vector contains unknown phase tags: {unknown}")
    if fill_missing_with_zero:
        for tag in tags:
            data.setdefault(tag, 0)
    return data


@dataclass(frozen=True)
class PhaseGrid:
    """Cartesian phase sampling for arbitrary tags and finite phase values."""

    phases_by_tag: dict[str, tuple[float, ...]]

    def __post_init__(self) -> None:
        normalized: dict[str, tuple[float, ...]] = {}
        for key, values in self.phases_by_tag.items():
            tag = validate_phase_tag(key, allow_none=False)
            assert tag is not None
            if tag in normalized:
                raise ValueError(f"PhaseGrid contains duplicate normalized phase tag: {tag!r}.")
            phases = tuple(float(value) for value in values)
            if not phases:
                raise ValueError(f"PhaseGrid tag {tag!r} must contain at least one phase.")
            if not all(np.isfinite(phase) for phase in phases):
                raise ValueError(f"PhaseGrid phases for tag {tag!r} must be finite.")
            normalized[tag] = phases
        object.__setattr__(self, "phases_by_tag", normalized)

    @property
    def tags(self) -> tuple[str, ...]:
        return tuple(self.phases_by_tag)

    def iter_phase_vectors(self) -> Iterator[PhaseVector]:
        if not self.tags:
            yield {}
            return
        phase_lists = [self.phases_by_tag[tag] for tag in self.tags]
        for phases in product(*phase_lists):
            yield {tag: float(phase) for tag, phase in zip(self.tags, phases)}

    def __len__(self) -> int:
        total = 1
        for tag in self.tags:
            total *= len(self.phases_by_tag[tag])
        return int(total)

    def to_dict(self) -> dict[str, Any]:
        return {
            "class": self.__class__.__name__,
            "tags": list(self.tags),
            "phases_by_tag": {tag: list(phases) for tag, phases in self.phases_by_tag.items()},
            "n_phase_cases": len(self),
        }


def _phase_step_count(value: Any, *, tag: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"n_steps for {tag!r} must be a positive integer, not bool.")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"n_steps for {tag!r} must be a positive integer.") from exc
    if not np.isfinite(numeric) or not numeric.is_integer() or numeric < 1:
        raise ValueError(f"n_steps for {tag!r} must be a positive integer. Got {value!r}.")
    return int(numeric)


def build_uniform_phase_grid(
    phase_tags: Sequence[str],
    *,
    n_steps: int | Mapping[str, int] = 4,
) -> PhaseGrid:
    """Build uniform grids with one shared N or an explicit N per phase tag."""

    tags = _stable_unique_phase_tags(tuple(phase_tags))
    if isinstance(n_steps, Mapping):
        normalized_steps: dict[str, Any] = {}
        for key, value in n_steps.items():
            tag = validate_phase_tag(key, allow_none=False)
            assert tag is not None
            if tag in normalized_steps:
                raise ValueError(f"n_steps contains duplicate normalized phase tag: {tag!r}.")
            normalized_steps[tag] = value
        missing = sorted(set(tags) - set(normalized_steps))
        extra = sorted(set(normalized_steps) - set(tags))
        if missing or extra:
            raise ValueError(
                "n_steps mapping keys must exactly match phase_tags. "
                f"Missing: {missing}; extra: {extra}."
            )
        steps_by_tag = {
            tag: _phase_step_count(normalized_steps[tag], tag=tag)
            for tag in tags
        }
    else:
        shared_steps = _phase_step_count(n_steps, tag="all phase tags")
        steps_by_tag = {tag: shared_steps for tag in tags}

    return PhaseGrid(
        {
            tag: tuple(2.0 * np.pi * index / steps_by_tag[tag] for index in range(steps_by_tag[tag]))
            for tag in tags
        }
    )


def _validate_projection_sign(sign: int, *, warn_legacy: bool) -> int:
    if isinstance(sign, (bool, np.bool_)) or sign not in {-1, 1}:
        raise ValueError("sign must be +1 or -1.")
    value = int(sign)
    if warn_legacy and value == -1:
        warnings.warn(
            "sign=-1 is a deprecated legacy phase-projection convention. "
            "Canonical target_phase_vector semantics use exp(+i*m*phi) with sign=+1.",
            DeprecationWarning,
            stacklevel=3,
        )
    return value


def phase_projection_convention_metadata(*, sign: int = 1) -> dict[str, Any]:
    """Return canonical or temporary legacy convention metadata."""

    value = _validate_projection_sign(sign, warn_legacy=False)
    if value == 1:
        return {
            "phase_projection_convention": PHASE_PROJECTION_CONVENTION,
            "phase_projection_convention_version": PHASE_PROJECTION_CONVENTION_VERSION,
            "target_phase_vector_semantics": TARGET_PHASE_VECTOR_SEMANTICS,
        }
    return {
        "phase_projection_convention": _LEGACY_PHASE_PROJECTION_CONVENTION,
        "phase_projection_convention_version": 0,
        "target_phase_vector_semantics": "legacy_target_interpreted_with_projection_sign",
    }


def _phase_projection_weight(
    phase_vector: Mapping[str, float],
    target_phase_vector: Mapping[str, int],
    *,
    sign: int,
) -> complex:
    phase_sum = 0.0
    for tag, coefficient in target_phase_vector.items():
        integer = _integer_coefficient(coefficient, tag=tag)
        if integer == 0:
            continue
        if tag not in phase_vector:
            raise ValueError(f"phase_vector is missing non-zero target tag: {tag!r}")
        phase_sum += float(integer) * float(phase_vector[tag])
    return complex(np.exp(sign * 1j * phase_sum))


def phase_projection_weight(
    phase_vector: Mapping[str, float],
    target_phase_vector: Mapping[str, int],
    *,
    sign: int = 1,
) -> complex:
    """Return ``exp(+i*m dot phi)`` for one phase case."""

    value = _validate_projection_sign(sign, warn_legacy=True)
    return _phase_projection_weight(phase_vector, target_phase_vector, sign=value)


def fourier_project_phase_cases(
    values: np.ndarray,
    phase_vectors: Sequence[Mapping[str, float]],
    target_phase_vector: Mapping[str, int],
    *,
    phase_axis: int = 0,
    normalize: bool = True,
    sign: int = 1,
) -> np.ndarray:
    """Authoritative equal-weight Fourier sum over one flattened case axis."""

    array = np.asarray(values)
    projection_sign = _validate_projection_sign(sign, warn_legacy=True)
    if array.ndim == 0:
        raise ValueError("values must have at least one phase axis.")
    axis = int(phase_axis)
    if axis < 0:
        axis += array.ndim
    if axis < 0 or axis >= array.ndim:
        raise ValueError(f"phase_axis is out of bounds for values.ndim={array.ndim}: {phase_axis}")
    if not phase_vectors:
        raise ValueError("phase_vectors must contain at least one phase case.")
    if array.shape[axis] != len(phase_vectors):
        raise ValueError(
            "values phase_axis length must match len(phase_vectors). "
            f"Got {array.shape[axis]} and {len(phase_vectors)}."
        )

    moved = np.moveaxis(array, axis, 0).astype(np.complex128, copy=False)
    weights = np.asarray(
        [
            _phase_projection_weight(phase_vector, target_phase_vector, sign=projection_sign)
            for phase_vector in phase_vectors
        ],
        dtype=np.complex128,
    )
    weighted = moved * weights.reshape((-1,) + (1,) * (moved.ndim - 1))
    projected = np.sum(weighted, axis=0)
    if normalize:
        projected = projected / float(len(phase_vectors))
    return projected


def _normalize_axis_names(axis_names: Sequence[str], *, ndim: int) -> tuple[str, ...]:
    names = tuple(str(name).strip() for name in axis_names)
    if len(names) != ndim:
        raise ValueError(f"len(axis_names) must equal data.ndim ({ndim}); got {len(names)}.")
    if any(not name for name in names):
        raise ValueError("axis_names must not contain empty names.")
    if len(set(names)) != len(names):
        raise ValueError("axis_names must be unique.")
    return names


def _normalize_axis_values(
    axis_values: Mapping[str, Any],
    *,
    axis_names: tuple[str, ...],
    data_shape: tuple[int, ...],
) -> dict[str, np.ndarray]:
    values: dict[str, np.ndarray] = {}
    for raw_name, raw_values in axis_values.items():
        name = str(raw_name).strip()
        if not name:
            raise ValueError("axis_values keys must not be empty.")
        if name in values:
            raise ValueError(f"axis_values contains duplicate normalized key: {name!r}.")
        if name not in axis_names:
            raise ValueError(f"axis_values contains unknown axis name: {name!r}.")
        array = np.asarray(raw_values)
        expected = data_shape[axis_names.index(name)]
        if array.ndim != 1 or array.size != expected:
            raise ValueError(
                f"axis_values[{name!r}] must be one-dimensional with length {expected}."
            )
        values[name] = array
    return values


def _normalize_phase_axes(
    phase_grid: PhaseGrid,
    phase_axes: Mapping[str, str] | None,
) -> dict[str, str]:
    if not phase_grid.tags:
        raise ValueError("phase_grid must define at least one phase tag for projection.")
    if phase_axes is None:
        return {tag: f"phase:{tag}" for tag in phase_grid.tags}
    if not isinstance(phase_axes, Mapping):
        raise TypeError("phase_axes must be a mapping from phase tag to ndarray axis name.")
    normalized: dict[str, str] = {}
    for raw_tag, raw_axis in phase_axes.items():
        tag = validate_phase_tag(raw_tag, allow_none=False)
        assert tag is not None
        if tag in normalized:
            raise ValueError(f"phase_axes contains duplicate normalized phase tag: {tag!r}.")
        axis_name = str(raw_axis).strip()
        if not axis_name:
            raise ValueError(f"phase axis name for tag {tag!r} must not be empty.")
        normalized[tag] = axis_name
    missing = sorted(set(phase_grid.tags) - set(normalized))
    extra = sorted(set(normalized) - set(phase_grid.tags))
    if missing or extra:
        raise ValueError(
            "phase_axes keys must exactly match phase_grid.tags. "
            f"Missing: {missing}; extra: {extra}."
        )
    if len(set(normalized.values())) != len(normalized):
        raise ValueError("phase_axes must map phase tags to distinct ndarray axes.")
    return {tag: normalized[tag] for tag in phase_grid.tags}


def _normalize_targets(
    targets: Mapping[str, Mapping[str, int | float]],
    *,
    phase_tags: tuple[str, ...],
) -> dict[str, TargetPhaseVector]:
    if not isinstance(targets, Mapping) or not targets:
        raise ValueError("targets must be a non-empty mapping of target name to phase-order vector.")
    normalized: dict[str, TargetPhaseVector] = {}
    for raw_name, target in targets.items():
        if not isinstance(raw_name, str):
            raise TypeError("target names must be strings.")
        name = raw_name.strip()
        if not name:
            raise ValueError("target names must not be empty.")
        if name in normalized:
            raise ValueError(f"targets contains duplicate normalized name: {name!r}.")
        normalized[name] = normalize_target_phase_vector(
            target,
            known_tags=phase_tags,
            fill_missing_with_zero=True,
        )
    return normalized


def project_phase_orders(
    data: np.ndarray,
    *,
    axis_names: Sequence[str],
    axis_values: Mapping[str, Any],
    phase_grid: PhaseGrid,
    targets: Mapping[str, Mapping[str, int | float]],
    phase_axes: Mapping[str, str] | None = None,
    normalize: bool = True,
) -> dict[str, Any]:
    """Project precomputed named-axis data onto one or more physical orders.

    ``PhaseGrid`` is authoritative for phase values.  ``axis_values`` must
    contain an equal realization for every mapped phase axis.  All phase axes
    are removed; remaining axes preserve their relative input order.
    """

    array = np.asarray(data)
    names = _normalize_axis_names(axis_names, ndim=array.ndim)
    if not isinstance(axis_values, Mapping):
        raise TypeError("axis_values must be a mapping from axis name to 1D values.")
    values = _normalize_axis_values(axis_values, axis_names=names, data_shape=array.shape)
    if not isinstance(phase_grid, PhaseGrid):
        raise TypeError("phase_grid must be a PhaseGrid instance.")
    tag_to_axis = _normalize_phase_axes(phase_grid, phase_axes)

    phase_axis_indices: list[int] = []
    for tag in phase_grid.tags:
        axis_name = tag_to_axis[tag]
        if axis_name not in names:
            raise ValueError(f"phase axis {axis_name!r} for tag {tag!r} is missing from axis_names.")
        if axis_name not in values:
            raise ValueError(f"axis_values must contain phase axis {axis_name!r} for tag {tag!r}.")
        actual = np.asarray(values[axis_name], dtype=float)
        expected = np.asarray(phase_grid.phases_by_tag[tag], dtype=float)
        if actual.shape != expected.shape or not np.allclose(actual, expected, rtol=0.0, atol=1.0e-12):
            raise ValueError(
                f"axis_values[{axis_name!r}] does not match PhaseGrid values for tag {tag!r}."
            )
        phase_axis_indices.append(names.index(axis_name))

    normalized_targets = _normalize_targets(targets, phase_tags=phase_grid.tags)
    remaining_indices = [index for index in range(array.ndim) if index not in phase_axis_indices]
    remaining_names = tuple(names[index] for index in remaining_indices)
    moved = np.moveaxis(array, phase_axis_indices, tuple(range(len(phase_axis_indices))))
    phase_shape = tuple(len(phase_grid.phases_by_tag[tag]) for tag in phase_grid.tags)
    if moved.shape[: len(phase_shape)] != phase_shape:
        raise ValueError(
            "data phase-axis shape does not match PhaseGrid. "
            f"Expected {phase_shape}, got {moved.shape[:len(phase_shape)]}."
        )
    payload_shape = tuple(array.shape[index] for index in remaining_indices)
    flattened = moved.reshape((len(phase_grid), *payload_shape))
    phase_vectors = tuple(phase_grid.iter_phase_vectors())
    projected = {
        name: fourier_project_phase_cases(
            flattened,
            phase_vectors,
            target,
            phase_axis=0,
            normalize=bool(normalize),
            sign=1,
        )
        for name, target in normalized_targets.items()
    }
    remaining_values = {
        name: values[name]
        for name in remaining_names
        if name in values
    }
    convention = phase_projection_convention_metadata(sign=1)
    return {
        "projected": projected,
        "axis_names": remaining_names,
        "axis_values": remaining_values,
        "targets": normalized_targets,
        "metadata": {
            **convention,
            "normalization": {
                "enabled": bool(normalize),
                "n_phase_cases": len(phase_grid),
                "divisor": len(phase_grid) if normalize else 1,
            },
            "phase_grid": phase_grid.to_dict(),
            "phase_axes": dict(tag_to_axis),
            "targets": {name: dict(target) for name, target in normalized_targets.items()},
            "remaining_axis_names": list(remaining_names),
            "nonuniform_phase_note": (
                "Explicit phase values use an equal-weight sum; standard exact discrete "
                "orthogonality is guaranteed only for the intended complete uniform grids."
            ),
        },
    }


__all__ = [
    "PHASE_PROJECTION_CONVENTION",
    "PHASE_PROJECTION_CONVENTION_VERSION",
    "TARGET_PHASE_VECTOR_SEMANTICS",
    "PhaseGrid",
    "build_uniform_phase_grid",
    "fourier_project_phase_cases",
    "normalize_target_phase_vector",
    "phase_projection_convention_metadata",
    "phase_projection_weight",
    "project_phase_orders",
]
