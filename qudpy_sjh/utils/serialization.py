"""Shared JSON serialization helpers for QuDPy.

The conversion deliberately keeps JSON output human-readable.  Runtime-only
objects are represented by descriptive metadata instead of being pickled.
"""

from __future__ import annotations

import json
import math
from dataclasses import fields as dataclass_fields, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np


def _is_qobj(value: Any) -> bool:
    value_type = type(value)
    return value_type.__name__ == "Qobj" and value_type.__module__.startswith("qutip")


def _complex_matrix_to_json(value: Any) -> list[list[dict[str, float]]]:
    array = np.asarray(value, dtype=np.complex128)
    return [
        [{"real": float(item.real), "imag": float(item.imag)} for item in row]
        for row in array
    ]


def json_safe(value: Any) -> Any:
    """Convert ``value`` recursively into data accepted by ``json.dumps``."""

    value_type_name = type(value).__name__
    if value_type_name == "ParaNormalizer":
        return {
            "class": "ParaNormalizer",
            "note": "runtime object omitted from JSON metadata",
        }
    if value_type_name == "NLevelPhysicalParams":
        if hasattr(value, "grouped_params"):
            return json_safe(value.grouped_params)
        payload = {
            item.name: getattr(value, item.name)
            for item in dataclass_fields(value)
            if item.name != "field"
        }
        field_value = getattr(value, "field", None)
        payload["field"] = None if field_value is None else json_safe(field_value)
        return json_safe(payload)
    if _is_qobj(value):
        return {
            "qobj_shape": list(value.shape),
            "data": _complex_matrix_to_json(value.full()),
        }
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return json_safe(value.to_dict())
    if is_dataclass(value):
        return json_safe(
            {item.name: getattr(value, item.name) for item in dataclass_fields(value)}
        )
    if isinstance(value, complex):
        return json_safe({"real": float(value.real), "imag": float(value.imag)})
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, Path):
        return str(value)
    if callable(value):
        return {"callable_serialized": False, "repr": repr(value)}
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return {
        "type": f"{value.__class__.__module__}.{value.__class__.__name__}",
        "repr": repr(value),
    }


def write_json(
    path: str | Path,
    payload: Any,
    *,
    indent: int = 2,
    ensure_ascii: bool = False,
) -> Path:
    """Write ``payload`` as UTF-8 JSON after applying :func:`json_safe`."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            json_safe(payload),
            indent=indent,
            ensure_ascii=ensure_ascii,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    return output


__all__ = ["json_safe", "write_json"]
