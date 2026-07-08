"""Generic pulse-sequence single-run execution and readout layer.

本模块只负责把一个 `SingleRunFieldPlan` 落到一次具体传播：

    field_plan -> FieldPhySeries -> replace(base_params, field=...) -> run_case

readout 只提供通用 polarization / absorption-like spectrum，不包含 TA
subtraction、phase average、2DES projection 或任何具体实验 recipe 语义。
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
from qudpy_sjh.utils.core import DynamicsResult, NLevelPhysicalParams, ParaNormalizer, run_case
from qudpy_sjh.utils.fields import FieldPhyRoot, FieldPhySeries
from qudpy_sjh.utils.spectroscopy import lab_frame_absorption_response, polarization_C_per_m2


_READOUT_MODES = {"none", "polarization", "absorption"}
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


@dataclass(frozen=True)
class ReadoutSpec:
    """一次 single-run 的通用 readout 配置。"""

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


@dataclass
class SingleRunReadoutResult:
    """一次 single-run 的通用 readout 结果。"""

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
            "max_abs_readout_field_MV_per_cm": None if readout_field is None else float(np.max(np.abs(readout_field))),
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
    """选择 readout denominator field；None 表示使用 total field。"""

    if not isinstance(field, FieldPhyRoot):
        raise TypeError("field must be a FieldPhyRoot instance.")
    if readout_field_name is None:
        return field
    name = validate_pulse_name(readout_field_name)
    try:
        selected = field[name]  # type: ignore[index]
    except KeyError as exc:
        raise KeyError(f"readout_field_name={name!r} was not found in the concrete field.") from exc
    except TypeError as exc:
        raise TypeError(
            f"readout_field_name={name!r} requires a field container with named subfields."
        ) from exc
    if not isinstance(selected, FieldPhyRoot):
        raise TypeError("selected readout field must be a FieldPhyRoot instance.")
    return selected


def _readout_field_source(readout_field_name: str | None) -> str:
    if readout_field_name is None:
        return "total_field"
    return f"subfield:{readout_field_name}"


def compute_single_run_readout(
    result: DynamicsResult,
    *,
    readout: ReadoutSpec,
) -> SingleRunReadoutResult | None:
    """对一次 `DynamicsResult` 计算通用 readout。"""

    if not isinstance(result, DynamicsResult):
        raise TypeError("result must be a DynamicsResult instance.")
    if not isinstance(readout, ReadoutSpec):
        raise TypeError("readout must be a ReadoutSpec instance.")
    if readout.mode == "none":
        return None
    physical = result.physical_params
    if physical is None:
        raise ValueError("DynamicsResult.physical_params is required for single-run readout.")
    if result.times_fs is None:
        raise ValueError("DynamicsResult.times_fs is required for single-run readout.")

    time_fs = np.asarray(result.times_fs, dtype=float)
    density = result.density_array()
    polarization = polarization_C_per_m2(
        density,
        physical.dipole_matrix_D,
        readout.number_density_m3,
    )
    metadata = {
        "readout_field_name": readout.readout_field_name,
        "readout_field_source": None,
        "number_density_m3": float(readout.number_density_m3),
        "readout_spec": readout.to_dict(),
        **dict(readout.metadata),
    }
    if readout.mode == "polarization":
        return SingleRunReadoutResult(
            mode=readout.mode,
            time_fs=time_fs,
            polarization_C_per_m2=polarization,
            metadata=metadata,
        )

    selected_field = select_readout_field(physical.field, readout.readout_field_name)
    readout_field_values = np.asarray(selected_field(time_fs), dtype=float)
    metadata["readout_field_source"] = _readout_field_source(readout.readout_field_name)
    response = lab_frame_absorption_response(
        time_fs=time_fs,
        polarization_C_per_m2=polarization,
        field=readout_field_values,
        window=readout.window,
        subtract_mean=readout.subtract_mean,
        rel_threshold=readout.rel_threshold,
        zero_padding_factor=readout.zero_padding_factor,
        return_intermediates=readout.return_intermediates,
    )
    return SingleRunReadoutResult(
        mode=readout.mode,
        time_fs=time_fs,
        polarization_C_per_m2=polarization,
        readout_field_MV_per_cm=readout_field_values,
        spectrum=response,
        metadata=metadata,
    )


@dataclass
class SingleRunPlan:
    """一次 concrete field configuration 的传播计划。"""

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
        readout = ReadoutSpec() if self.readout is None else self.readout
        base_metadata.update(self.input_metadata)
        base_metadata["single_run_workflow"] = {
            "case_name": self.case_name,
            "field_plan": self.field_plan.to_dict(),
            "readout": readout.to_dict(),
            "phase_vector": dict(self.field_plan.phase_vector),
            "centers_fs": dict(self.field_plan.centers_fs),
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
        readout_spec = ReadoutSpec() if self.readout is None else self.readout
        readout_result = compute_single_run_readout(dynamics, readout=readout_spec)
        return SingleRunResult(
            case_name=self.case_name,
            params=params,
            dynamics_result=dynamics,
            field_metadata=params.field.to_dict(),
            readout=readout_result,
            metadata={
                "single_run_plan": self.to_dict(),
                "checkpoint": self.checkpoint.to_dict(),
            },
        )

    def to_dict(self) -> dict[str, Any]:
        readout = ReadoutSpec() if self.readout is None else self.readout
        return {
            "class": self.__class__.__name__,
            "case_name": self.case_name,
            "field_plan": self.field_plan.to_dict(),
            "readout": readout.to_dict(),
            "checkpoint": self.checkpoint.to_dict(),
            "input_description": self.input_description,
            "input_metadata": dict(self.input_metadata),
        }


@dataclass
class SingleRunResult:
    """一次 generic single-run execution 的结构化结果。"""

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
    "select_readout_field",
]
