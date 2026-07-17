"""Adapters between systems definitions and core physical params.

本模块只把 matter-side `NLevelSystem` 映射到已有
`NLevelPhysicalParams` 容器；它不生成 field、pulse sequence、readout、
checkpoint，也不调用 solver。
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping

import numpy as np

from qudpy_sjh.systems.core import NLevelSystem
from qudpy_sjh.utils.core import NLevelPhysicalParams, PureDephasingChannel, RelaxationChannel


def _transition_dephasing_records(system: NLevelSystem) -> list[dict[str, Any]]:
    return [
        {
            "from": left,
            "to": right,
            "gamma_fs_inv": float(gamma),
        }
        for (left, right), gamma in system.transition_dephasing_fs_inv.items()
    ]


def _initial_state_summary(system: NLevelSystem) -> dict[str, Any]:
    initial = system.initial_state
    if initial is None:
        return {
            "present": False,
            "policy": "NLevelPhysicalParams has no initial_state field; solver default rho0 is unchanged.",
        }
    array = np.asarray(initial)
    return {
        "present": True,
        "shape": tuple(array.shape),
        "policy": "NLevelPhysicalParams has no initial_state field; value is metadata-only in this adapter.",
    }


def _split_dissipation(system: NLevelSystem) -> tuple[tuple[RelaxationChannel, ...], tuple[PureDephasingChannel, ...], list[dict[str, Any]]]:
    relaxation: list[RelaxationChannel] = []
    pure_dephasing: list[PureDephasingChannel] = []
    unmapped: list[dict[str, Any]] = []
    for index, channel in enumerate(system.dissipation):
        if isinstance(channel, RelaxationChannel):
            relaxation.append(channel)
        elif isinstance(channel, PureDephasingChannel):
            pure_dephasing.append(channel)
        else:
            unmapped.append(
                {
                    "index": int(index),
                    "type": f"{channel.__class__.__module__}.{channel.__class__.__name__}",
                    "policy": "not mapped to NLevelPhysicalParams channel fields",
                }
            )
    return tuple(relaxation), tuple(pure_dephasing), unmapped


def _system_adapter_metadata(
    system: NLevelSystem,
    *,
    transition_dephasing_policy: str,
    dissipation_policy: str,
    unmapped_dissipation: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "source": "NLevelSystem",
        "system_name": system.name,
        "system_dimension": system.dimension,
        "generator": system.metadata.get("generator"),
        "transition_dephasing_policy": transition_dephasing_policy,
        "dissipation_policy": dissipation_policy,
        "initial_state": _initial_state_summary(system),
        "transition_dephasing_fs_inv": _transition_dephasing_records(system),
        "unmapped_dissipation": unmapped_dissipation,
    }


def _merge_metadata(
    *,
    system: NLevelSystem,
    existing: Mapping[str, Any] | None = None,
    user: Mapping[str, Any] | None = None,
    transition_dephasing_policy: str,
    dissipation_policy: str,
    unmapped_dissipation: list[dict[str, Any]],
) -> dict[str, Any]:
    merged = dict(existing or {})
    for key, value in dict(user or {}).items():
        if key in merged and merged[key] != value:
            raise ValueError(f"input_metadata key collision: {key!r}.")
        merged[key] = value

    adapter_payload = _system_adapter_metadata(
        system,
        transition_dephasing_policy=transition_dephasing_policy,
        dissipation_policy=dissipation_policy,
        unmapped_dissipation=unmapped_dissipation,
    )
    if "system_adapter" in merged and merged["system_adapter"] != adapter_payload:
        if "previous_system_adapter" in merged:
            raise ValueError("input_metadata key collision: 'previous_system_adapter'.")
        merged["previous_system_adapter"] = merged["system_adapter"]
    merged["system_adapter"] = adapter_payload

    if "system" in merged and merged["system"] != system.to_dict(include_arrays=False):
        if "previous_system" in merged:
            raise ValueError("input_metadata key collision: 'previous_system'.")
        merged["previous_system"] = merged["system"]
    merged["system"] = system.to_dict(include_arrays=False)
    return merged


def _matter_kwargs_from_system(system: NLevelSystem) -> dict[str, Any]:
    return {
        "basis": tuple(system.basis),
        "energies_eV": tuple(float(item) for item in system.energies_eV),
        "dipole_matrix_D": tuple(tuple(complex(value) for value in row) for row in system.dipole_matrix_D),
    }


def _require_system(system: NLevelSystem) -> None:
    if not isinstance(system, NLevelSystem):
        raise TypeError("system must be an NLevelSystem instance.")


def make_base_physical_params_from_system(
    system: NLevelSystem,
    *,
    field: Any,
    t_start_fs: float,
    t_end_fs: float,
    dt_fs: float,
    solver_mode: str = "lab_exact",
    input_description: str | None = None,
    input_metadata: Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> NLevelPhysicalParams:
    """从 `NLevelSystem` 构造基础 `NLevelPhysicalParams`。

    `field` 和 time grid 必须由调用方显式提供；transition-level
    dephasing 暂不转换成 level pure-dephasing channels，而是记录到
    metadata。
    """

    _require_system(system)
    if field is None:
        raise ValueError("field is required.")
    relaxation_from_system, pure_from_system, unmapped = _split_dissipation(system)
    relaxation_channels = tuple(kwargs.pop("relaxation_channels", ())) + relaxation_from_system
    pure_dephasing_channels = tuple(kwargs.pop("pure_dephasing_channels", ())) + pure_from_system
    metadata = _merge_metadata(
        system=system,
        user=input_metadata,
        transition_dephasing_policy="metadata_only; no compatible transition-level field in NLevelPhysicalParams",
        dissipation_policy="RelaxationChannel/PureDephasingChannel objects are mapped; other objects are metadata-only",
        unmapped_dissipation=unmapped,
    )
    return NLevelPhysicalParams(
        **_matter_kwargs_from_system(system),
        t_start_fs=float(t_start_fs),
        t_end_fs=float(t_end_fs),
        dt_fs=float(dt_fs),
        field=field,
        relaxation_channels=relaxation_channels,
        pure_dephasing_channels=pure_dephasing_channels,
        solver_mode=str(solver_mode),
        input_description=input_description,
        input_metadata=metadata,
        **kwargs,
    )


def update_physical_params_system(
    params: NLevelPhysicalParams,
    system: NLevelSystem,
    *,
    input_description: str | None = None,
    input_metadata: Mapping[str, Any] | None = None,
    preserve_existing_metadata: bool = True,
) -> NLevelPhysicalParams:
    """返回覆写 matter-side system 后的新 `NLevelPhysicalParams`。

    保留 field、time grid、solver mode 以及其它非 matter-side 设置。
    """

    if not isinstance(params, NLevelPhysicalParams):
        raise TypeError("params must be an NLevelPhysicalParams instance.")
    _require_system(system)
    relaxation_from_system, pure_from_system, unmapped = _split_dissipation(system)
    metadata = _merge_metadata(
        system=system,
        existing=params.input_metadata if preserve_existing_metadata else None,
        user=input_metadata,
        transition_dephasing_policy="metadata_only; no compatible transition-level field in NLevelPhysicalParams",
        dissipation_policy="system dissipation replaces existing channel fields when compatible",
        unmapped_dissipation=unmapped,
    )
    description = params.input_description if input_description is None else input_description
    return replace(
        params,
        **_matter_kwargs_from_system(system),
        relaxation_channels=relaxation_from_system,
        pure_dephasing_channels=pure_from_system,
        input_description=description,
        input_metadata=metadata,
    )


with_system_in_physical_params = update_physical_params_system


__all__ = [
    "make_base_physical_params_from_system",
    "update_physical_params_system",
    "with_system_in_physical_params",
]
