"""Generic pulse-sequence single-run dynamics execution.

本模块只负责把一个 `SingleRunFieldPlan` 落到一次具体传播：

    field_plan -> FieldPhySeries -> replace(base_params, field=...) -> run_case

Canonical ``SingleRunPlan.execute()`` stops at dynamics.  The legacy embedded
readout configuration remains only as an explicit compatibility adapter while
callers migrate to ``DynamicsResult -> PolarizationResult -> ReadoutPlan``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field as dataclass_field, replace
from pathlib import Path
from typing import Any

import numpy as np

from qudpy_sjh.experiments.pulse_sequence.pulse_sequence import (
    SingleRunFieldPlan,
    validate_pulse_name,
)
from qudpy_sjh.experiments.readout import (
    ReadoutPlan,
    ReadoutResult,
    compute_polarization_result,
    select_interaction_readout_field,
)
from qudpy_sjh.utils.core import DynamicsResult, NLevelPhysicalParams, ParaNormalizer, run_case
from qudpy_sjh.utils.fields import FieldPhyRoot, FieldPhySeries


_READOUT_MODES = {"none", "polarization", "absorption"}
_WINDOWS = {None, "none", "hann"}


def _copy_metadata(metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    return dict(metadata or {})


@dataclass(frozen=True)
class ReadoutSpec:
    """Legacy embedded single-run readout configuration.

    New code should construct ``ReadoutPlan`` and execute it after dynamics.
    """

    mode: str = "none"
    number_density_m3: float = 1.0e24
    readout_field_name: str | None = None
    window: str | None = "hann"
    subtract_mean: bool = True
    rel_threshold: float = 1.0e-6
    zero_padding_factor: int = 4
    return_intermediates: bool = True
    metadata: dict[str, Any] = dataclass_field(default_factory=dict)

    def __post_init__(self) -> None:
        mode = str(self.mode).strip()
        if mode not in _READOUT_MODES:
            raise ValueError(f"Unsupported readout mode: {self.mode!r}. Expected one of {sorted(_READOUT_MODES)}.")
        density = float(self.number_density_m3)
        if density <= 0.0:
            raise ValueError("number_density_m3 must be > 0.")
        rel_threshold = float(self.rel_threshold)
        if rel_threshold <= 0.0:
            raise ValueError("rel_threshold must be > 0.")
        zero_padding_factor = int(self.zero_padding_factor)
        if zero_padding_factor < 1:
            raise ValueError("zero_padding_factor must be >= 1.")
        if self.window not in _WINDOWS:
            raise ValueError("window must be None, 'none', or 'hann'.")
        field_name = None if self.readout_field_name is None else validate_pulse_name(self.readout_field_name)

        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "number_density_m3", density)
        object.__setattr__(self, "readout_field_name", field_name)
        object.__setattr__(self, "rel_threshold", rel_threshold)
        object.__setattr__(self, "zero_padding_factor", zero_padding_factor)
        object.__setattr__(self, "subtract_mean", bool(self.subtract_mean))
        object.__setattr__(self, "return_intermediates", bool(self.return_intermediates))
        object.__setattr__(self, "metadata", _copy_metadata(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "class": self.__class__.__name__,
            "mode": self.mode,
            "number_density_m3": float(self.number_density_m3),
            "readout_field_name": self.readout_field_name,
            "window": self.window,
            "subtract_mean": bool(self.subtract_mean),
            "rel_threshold": float(self.rel_threshold),
            "zero_padding_factor": int(self.zero_padding_factor),
            "return_intermediates": bool(self.return_intermediates),
            "metadata": dict(self.metadata),
        }


# Compatibility name retained for phase-cycling and TA wrappers until result cleanup.
SingleRunReadoutResult = ReadoutResult


@dataclass(frozen=True)
class SingleRunCheckpointSettings:
    """一次 single-run 的最小 checkpoint 设置。"""

    enabled: bool = False
    checkpoint_path: Path | str | None = None
    force_run: bool = False
    require_existing_for_load: bool = False

    def __post_init__(self) -> None:
        enabled = bool(self.enabled)
        path = None if self.checkpoint_path is None else Path(self.checkpoint_path)
        if enabled and path is None:
            raise ValueError("checkpoint_path is required when checkpoint.enabled=True.")
        object.__setattr__(self, "enabled", enabled)
        object.__setattr__(self, "checkpoint_path", path)
        object.__setattr__(self, "force_run", bool(self.force_run))
        object.__setattr__(self, "require_existing_for_load", bool(self.require_existing_for_load))

    def to_dict(self) -> dict[str, Any]:
        return {
            "class": self.__class__.__name__,
            "enabled": bool(self.enabled),
            "checkpoint_path": None if self.checkpoint_path is None else str(self.checkpoint_path),
            "force_run": bool(self.force_run),
            "require_existing_for_load": bool(self.require_existing_for_load),
        }


def select_readout_field(field: FieldPhyRoot, readout_field_name: str | None) -> FieldPhyRoot:
    """Compatibility wrapper for interaction-field readout selection."""

    name = None if readout_field_name is None else validate_pulse_name(readout_field_name)
    return select_interaction_readout_field(field, name)


def readout_plan_from_spec(readout: ReadoutSpec) -> ReadoutPlan | None:
    """Translate a legacy ``ReadoutSpec`` into the canonical executable plan."""

    if not isinstance(readout, ReadoutSpec):
        raise TypeError("readout must be a ReadoutSpec instance.")
    if readout.mode == "none":
        return None
    mode = "absorption_like" if readout.mode == "absorption" else readout.mode
    return ReadoutPlan(
        mode=mode,
        readout_field=readout.readout_field_name if mode != "polarization" else None,
        window=readout.window,
        subtract_mean=readout.subtract_mean,
        rel_threshold=readout.rel_threshold,
        zero_padding_factor=readout.zero_padding_factor,
        return_intermediates=readout.return_intermediates,
        metadata={
            "compatibility_source": "ReadoutSpec",
            **dict(readout.metadata),
        },
    )


def compute_single_run_readout(
    result: DynamicsResult,
    *,
    readout: ReadoutSpec,
) -> SingleRunReadoutResult | None:
    """Compatibility adapter from saved dynamics and ``ReadoutSpec``."""

    if not isinstance(result, DynamicsResult):
        raise TypeError("result must be a DynamicsResult instance.")
    if not isinstance(readout, ReadoutSpec):
        raise TypeError("readout must be a ReadoutSpec instance.")
    plan = readout_plan_from_spec(readout)
    if plan is None:
        return None
    physical = result.physical_params
    if physical is None:
        raise ValueError("DynamicsResult.physical_params is required for single-run readout.")
    polarization = compute_polarization_result(
        result,
        number_density_m3=readout.number_density_m3,
    )
    canonical = plan.execute(polarization, interaction_field=physical.field)
    canonical.mode = readout.mode
    canonical.metadata.update(
        {
            "legacy_mode": readout.mode,
            "canonical_mode": plan.mode,
            "number_density_m3": float(readout.number_density_m3),
            "readout_field_name": readout.readout_field_name,
            "readout_spec": readout.to_dict(),
            "temporary_compatibility_path": True,
        }
    )
    return canonical


@dataclass
class SingleRunPlan:
    """一次 concrete field configuration 的 dynamics propagation plan.

    ``readout`` is retained only for ``execute_with_legacy_readout()`` while
    existing experiment runners migrate to standalone ``ReadoutPlan`` objects.
    """

    base_params: NLevelPhysicalParams
    field_plan: SingleRunFieldPlan
    normalizer: ParaNormalizer = dataclass_field(default_factory=ParaNormalizer)
    readout: ReadoutSpec | None = None
    checkpoint: SingleRunCheckpointSettings = dataclass_field(default_factory=SingleRunCheckpointSettings)
    case_name: str | None = None
    input_description: str | None = None
    input_metadata: dict[str, Any] = dataclass_field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.base_params, NLevelPhysicalParams):
            raise TypeError("base_params must be a NLevelPhysicalParams instance.")
        if not isinstance(self.field_plan, SingleRunFieldPlan):
            raise TypeError("field_plan must be a SingleRunFieldPlan instance.")
        if not isinstance(self.normalizer, ParaNormalizer):
            raise TypeError("normalizer must be a ParaNormalizer instance.")
        if self.readout is not None and not isinstance(self.readout, ReadoutSpec):
            raise TypeError("readout must be a ReadoutSpec instance or None.")
        if not isinstance(self.checkpoint, SingleRunCheckpointSettings):
            raise TypeError("checkpoint must be a SingleRunCheckpointSettings instance.")
        self.case_name = validate_pulse_name(self.field_plan.case_name if self.case_name is None else self.case_name)
        self.input_metadata = _copy_metadata(self.input_metadata)

    def build_field(self) -> FieldPhySeries:
        return self.field_plan.build_field()

    def make_params(self) -> NLevelPhysicalParams:
        field = self.build_field()
        base_metadata = _copy_metadata(self.base_params.input_metadata)
        base_metadata.update(self.input_metadata)
        base_metadata["single_run_workflow"] = {
            "case_name": self.case_name,
            "field_plan": self.field_plan.to_dict(),
            "phase_vector": dict(self.field_plan.phase_vector),
            "centers_fs": dict(self.field_plan.centers_fs),
            "execution_scope": "dynamics_only",
        }
        return replace(
            self.base_params,
            field=field,
            input_description=self.input_description
            if self.input_description is not None
            else self.base_params.input_description,
            input_metadata=base_metadata,
        )

    def execute(self) -> "SingleRunResult":
        """Execute dynamics only; detector/readout physics is a later stage."""

        params = self.make_params()
        checkpoint_path = self.checkpoint.checkpoint_path
        load_ckp = None
        save_ckp = None
        if self.checkpoint.enabled:
            assert checkpoint_path is not None
            if (
                self.checkpoint.require_existing_for_load
                and not self.checkpoint.force_run
                and not checkpoint_path.exists()
            ):
                raise FileNotFoundError(checkpoint_path)
            load_ckp = checkpoint_path
            save_ckp = checkpoint_path
        dynamics = run_case(
            params,
            normalizer=self.normalizer,
            load_ckp=load_ckp,
            save_ckp=save_ckp,
            force_run=self.checkpoint.force_run,
        )
        return SingleRunResult(
            case_name=self.case_name,
            params=params,
            dynamics_result=dynamics,
            field_metadata=params.field.to_dict(),
            readout=None,
            metadata={
                "single_run_plan": self.execution_dict(),
                "checkpoint": self.checkpoint.to_dict(),
                "execution_scope": "dynamics_only",
            },
        )

    def execute_with_legacy_readout(self) -> "SingleRunResult":
        """Temporary adapter preserving the pre-M2 embedded-readout workflow."""

        result = self.execute()
        readout_spec = ReadoutSpec() if self.readout is None else self.readout
        result.readout = compute_single_run_readout(
            result.dynamics_result,
            readout=readout_spec,
        )
        result.metadata["readout_compatibility"] = {
            "temporary": True,
            "adapter": "SingleRunPlan.execute_with_legacy_readout",
            "readout_spec": readout_spec.to_dict(),
        }
        return result

    def execution_dict(self) -> dict[str, Any]:
        """Serialize only inputs that own or affect dynamics execution."""

        return {
            "class": self.__class__.__name__,
            "case_name": self.case_name,
            "field_plan": self.field_plan.to_dict(),
            "checkpoint": self.checkpoint.to_dict(),
            "input_description": self.input_description,
            "input_metadata": dict(self.input_metadata),
            "execution_scope": "dynamics_only",
        }

    def to_dict(self) -> dict[str, Any]:
        readout = ReadoutSpec() if self.readout is None else self.readout
        return {
            **self.execution_dict(),
            "legacy_readout": readout.to_dict(),
            "legacy_readout_status": "temporary compatibility; ignored by execute()",
        }


@dataclass
class SingleRunResult:
    """一次 generic dynamics execution 的轻量结构化 wrapper。

    ``readout`` remains optional only for temporary compatibility wrappers.
    """

    case_name: str
    params: NLevelPhysicalParams
    dynamics_result: DynamicsResult
    field_metadata: dict[str, Any]
    readout: SingleRunReadoutResult | None = None
    metadata: dict[str, Any] = dataclass_field(default_factory=dict)

    def to_dict(self, *, include_arrays: bool = False) -> dict[str, Any]:
        time = np.asarray(
            self.dynamics_result.times_fs
            if self.dynamics_result.times_fs is not None
            else self.dynamics_result.times,
            dtype=float,
        )
        return {
            "class": self.__class__.__name__,
            "case_name": self.case_name,
            "field_metadata": dict(self.field_metadata),
            "readout": None if self.readout is None else self.readout.to_dict(include_arrays=include_arrays),
            "max_trace_error": float(self.dynamics_result.max_trace_error()),
            "max_hermiticity_error": float(self.dynamics_result.max_hermiticity_error()),
            "time_range_fs": (float(time[0]), float(time[-1])),
            "dimension": int(self.dynamics_result.dimension()),
            "metadata": dict(self.metadata),
        }


__all__ = [
    "ReadoutSpec",
    "SingleRunReadoutResult",
    "SingleRunCheckpointSettings",
    "SingleRunPlan",
    "SingleRunResult",
    "compute_single_run_readout",
    "readout_plan_from_spec",
    "select_readout_field",
]
