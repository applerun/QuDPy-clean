"""Generic pulse-sequence single-run dynamics execution.

本模块只负责把一个 `SingleRunFieldPlan` 落到一次具体传播：

    field_plan -> FieldPhySeries -> replace(base_params, field=...) -> run_case

Canonical ``SingleRunPlan.execute()`` stops at dynamics.  Detector physics is
owned separately by ``ReadoutPlan``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field as dataclass_field, replace
from pathlib import Path
from typing import Any

import numpy as np

from qudpy_sjh.experiments.pulse_sequence.pulse_sequence import SingleRunFieldPlan, validate_pulse_name
from qudpy_sjh.utils.core import DynamicsResult, NLevelPhysicalParams, ParaNormalizer, run_case
from qudpy_sjh.utils.fields import FieldPhySeries


def _copy_metadata(metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    return dict(metadata or {})


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


@dataclass
class SingleRunPlan:
    """一次 concrete field configuration 的 dynamics propagation plan."""

    base_params: NLevelPhysicalParams
    field_plan: SingleRunFieldPlan
    normalizer: ParaNormalizer = dataclass_field(default_factory=ParaNormalizer)
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
            metadata={
                "single_run_plan": self.execution_dict(),
                "checkpoint": self.checkpoint.to_dict(),
                "execution_scope": "dynamics_only",
            },
        )

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
        return self.execution_dict()


@dataclass
class SingleRunResult:
    """一次 generic dynamics execution 的轻量结构化 wrapper。"""

    case_name: str
    params: NLevelPhysicalParams
    dynamics_result: DynamicsResult
    field_metadata: dict[str, Any]
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
            "max_trace_error": float(self.dynamics_result.max_trace_error()),
            "max_hermiticity_error": float(self.dynamics_result.max_hermiticity_error()),
            "time_range_fs": (float(time[0]), float(time[-1])),
            "dimension": int(self.dynamics_result.dimension()),
            "metadata": dict(self.metadata),
        }


__all__ = [
    "SingleRunCheckpointSettings",
    "SingleRunPlan",
    "SingleRunResult",
]
