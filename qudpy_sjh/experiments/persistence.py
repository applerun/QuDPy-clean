"""Lightweight persistence for canonical projected phase-order mappings."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from qudpy_sjh.experiments.pulse_sequence.phase_projection import (
    PHASE_PROJECTION_CONVENTION,
    PHASE_PROJECTION_CONVENTION_VERSION,
    TARGET_PHASE_VECTOR_SEMANTICS,
    PhaseGrid,
    normalize_target_phase_vector,
)
from qudpy_sjh.utils.serialization import write_json


PROJECTED_RESULT_SCHEMA = "qudpy_projected_phase_orders"
PROJECTED_RESULT_SCHEMA_VERSION = 1


def _projected_paths(path: str | Path) -> tuple[Path, Path]:
    supplied = Path(path)
    if supplied.suffix.lower() in {".npz", ".json"}:
        base = supplied.with_suffix("")
    else:
        base = supplied
    return Path(f"{base}.npz"), Path(f"{base}.json")


def _axis_names(value: Any) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError("projected result axis_names must be a sequence of names.")
    names = tuple(str(name).strip() for name in value)
    if any(not name for name in names):
        raise ValueError("projected result axis_names must not contain empty names.")
    if len(set(names)) != len(names):
        raise ValueError("projected result axis_names must be unique.")
    return names


def _phase_grid_from_metadata(metadata: Mapping[str, Any]) -> PhaseGrid:
    payload = metadata.get("phase_grid")
    if not isinstance(payload, Mapping):
        raise ValueError("projected result metadata must contain a phase_grid mapping.")
    phases_by_tag = payload.get("phases_by_tag")
    if not isinstance(phases_by_tag, Mapping):
        raise ValueError("projected result metadata.phase_grid must contain phases_by_tag.")
    return PhaseGrid(
        {
            str(tag): tuple(float(phase) for phase in phases)
            for tag, phases in phases_by_tag.items()
        }
    )


def _validate_projected_result(result: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(result, Mapping):
        raise TypeError("result must be the lightweight mapping returned by project_phase_orders.")
    projected_raw = result.get("projected")
    targets_raw = result.get("targets")
    axis_values_raw = result.get("axis_values")
    metadata_raw = result.get("metadata")
    if not isinstance(projected_raw, Mapping) or not projected_raw:
        raise ValueError("projected result must contain a non-empty 'projected' mapping.")
    if not isinstance(targets_raw, Mapping):
        raise ValueError("projected result must contain a 'targets' mapping.")
    if not isinstance(axis_values_raw, Mapping):
        raise ValueError("projected result must contain an 'axis_values' mapping.")
    if not isinstance(metadata_raw, Mapping):
        raise ValueError("projected result must contain a 'metadata' mapping.")

    names = _axis_names(result.get("axis_names"))
    metadata = dict(metadata_raw)
    expected_convention = {
        "phase_projection_convention": PHASE_PROJECTION_CONVENTION,
        "phase_projection_convention_version": PHASE_PROJECTION_CONVENTION_VERSION,
        "target_phase_vector_semantics": TARGET_PHASE_VECTOR_SEMANTICS,
    }
    for key, expected in expected_convention.items():
        if metadata.get(key) != expected:
            raise ValueError(f"projected result metadata {key!r} must equal {expected!r}.")
    normalization = metadata.get("normalization")
    if not isinstance(normalization, Mapping) or "enabled" not in normalization:
        raise ValueError("projected result metadata must contain normalization details.")

    grid = _phase_grid_from_metadata(metadata)
    projected: dict[str, np.ndarray] = {}
    normalized_targets: dict[str, dict[str, int]] = {}
    target_names: list[str] = []
    for raw_name, raw_array in projected_raw.items():
        if not isinstance(raw_name, str) or not raw_name.strip():
            raise ValueError("projected channel names must be non-empty strings.")
        name = raw_name.strip()
        if name in projected:
            raise ValueError(f"duplicate normalized projected channel name: {name!r}.")
        array = np.asarray(raw_array)
        if array.ndim != len(names):
            raise ValueError(
                f"projected channel {name!r} ndim must equal len(axis_names)={len(names)}."
            )
        if raw_name not in targets_raw:
            raise ValueError(f"projected channel {raw_name!r} has no target phase-order vector.")
        target = normalize_target_phase_vector(
            targets_raw[raw_name],
            known_tags=grid.tags,
            fill_missing_with_zero=True,
        )
        if dict(targets_raw[raw_name]) != target:
            raise ValueError(
                f"target {raw_name!r} must explicitly contain every PhaseGrid tag with integer orders."
            )
        projected[name] = array
        normalized_targets[name] = target
        target_names.append(name)
    extra_targets = sorted(set(targets_raw) - set(projected_raw))
    if extra_targets:
        raise ValueError(f"targets contains channels absent from projected arrays: {extra_targets}")

    axis_values: dict[str, np.ndarray] = {}
    reference_shape = next(iter(projected.values())).shape
    for raw_name, raw_values in axis_values_raw.items():
        name = str(raw_name).strip()
        if name not in names:
            raise ValueError(f"axis_values contains unknown remaining axis: {name!r}.")
        values = np.asarray(raw_values)
        expected_length = reference_shape[names.index(name)]
        if values.ndim != 1 or values.size != expected_length:
            raise ValueError(
                f"axis_values[{name!r}] must be one-dimensional with length {expected_length}."
            )
        axis_values[name] = values
    for name, array in projected.items():
        if array.shape != reference_shape:
            raise ValueError(
                f"all projected channels must share one shape; {name!r} has {array.shape}, "
                f"expected {reference_shape}."
            )

    return {
        "projected": projected,
        "axis_names": names,
        "axis_values": axis_values,
        "targets": normalized_targets,
        "metadata": metadata,
        "target_names": target_names,
    }


def save_projected_result(
    result: Mapping[str, Any],
    path: str | Path,
) -> dict[str, Path]:
    """Save one canonical projected mapping as compressed NPZ plus strict JSON."""

    canonical = _validate_projected_result(result)
    npz_path, json_path = _projected_paths(path)
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    projected_keys: dict[str, str] = {}
    axis_keys: dict[str, str] = {}
    for index, name in enumerate(canonical["target_names"]):
        key = f"projected_{index:04d}"
        arrays[key] = canonical["projected"][name]
        projected_keys[name] = key
    for index, name in enumerate(canonical["axis_names"]):
        if name not in canonical["axis_values"]:
            continue
        key = f"axis_{index:04d}"
        arrays[key] = canonical["axis_values"][name]
        axis_keys[name] = key
    np.savez_compressed(npz_path, **arrays)
    write_json(
        json_path,
        {
            "schema": PROJECTED_RESULT_SCHEMA,
            "schema_version": PROJECTED_RESULT_SCHEMA_VERSION,
            "npz_file": npz_path.name,
            "axis_names": list(canonical["axis_names"]),
            "axis_array_keys": axis_keys,
            "projected_array_keys": projected_keys,
            "targets": canonical["targets"],
            "metadata": canonical["metadata"],
        },
    )
    return {"npz": npz_path, "json": json_path}


def load_projected_result(path: str | Path) -> dict[str, Any]:
    """Load and validate a canonical lightweight projected mapping."""

    default_npz_path, json_path = _projected_paths(path)
    manifest = json.loads(json_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != PROJECTED_RESULT_SCHEMA:
        raise ValueError(f"Unsupported projected result schema: {manifest.get('schema')!r}.")
    if manifest.get("schema_version") != PROJECTED_RESULT_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported projected result schema version: {manifest.get('schema_version')!r}."
        )
    npz_name = manifest.get("npz_file")
    npz_path = default_npz_path if not isinstance(npz_name, str) else json_path.parent / npz_name
    projected_keys = manifest.get("projected_array_keys")
    axis_keys = manifest.get("axis_array_keys")
    if not isinstance(projected_keys, Mapping) or not isinstance(axis_keys, Mapping):
        raise ValueError("Projected result manifest is missing array-key mappings.")
    with np.load(npz_path, allow_pickle=False) as archive:
        projected = {
            str(name): np.asarray(archive[str(key)])
            for name, key in projected_keys.items()
        }
        axis_values = {
            str(name): np.asarray(archive[str(key)])
            for name, key in axis_keys.items()
        }
    result = {
        "projected": projected,
        "axis_names": tuple(manifest.get("axis_names", ())),
        "axis_values": axis_values,
        "targets": manifest.get("targets", {}),
        "metadata": manifest.get("metadata", {}),
    }
    canonical = _validate_projected_result(result)
    return {
        "projected": canonical["projected"],
        "axis_names": canonical["axis_names"],
        "axis_values": canonical["axis_values"],
        "targets": canonical["targets"],
        "metadata": canonical["metadata"],
    }


__all__ = [
    "PROJECTED_RESULT_SCHEMA",
    "PROJECTED_RESULT_SCHEMA_VERSION",
    "load_projected_result",
    "save_projected_result",
]
