"""Legacy phase-grid execution and projected-result compatibility utilities.

Canonical pure Fourier mathematics lives in ``phase_projection``.  The heavy
runner and result wrappers remain here temporarily for compatibility.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field as dataclass_field, replace
from typing import Any

import numpy as np

from qudpy_sjh.experiments.pulse_sequence.phase_projection import (
    PHASE_PROJECTION_CONVENTION,
    PHASE_PROJECTION_CONVENTION_VERSION,
    TARGET_PHASE_VECTOR_SEMANTICS,
    PhaseGrid,
    PhaseVector,
    TargetPhaseVector,
    _validate_projection_sign,
    build_uniform_phase_grid,
    fourier_project_phase_cases,
    normalize_target_phase_vector,
    phase_projection_convention_metadata,
    phase_projection_weight,
    project_phase_orders,
)
from qudpy_sjh.experiments.pulse_sequence.single_run import (
    SingleRunPlan,
    SingleRunReadoutResult,
    SingleRunResult,
)


def _json_array(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, complex):
        return {"real": float(value.real), "imag": float(value.imag)}
    if isinstance(value, dict):
        return {str(key): _json_array(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_array(item) for item in value]
    return value


@dataclass(frozen=True)
class PhaseProjectionSpec:
    """Legacy runner configuration for one embedded-readout quantity."""

    quantity: str
    phase_axis: int = 0
    normalize: bool = True
    sign: int = 1
    require_matching_shape: bool = True
    metadata: dict[str, Any] = dataclass_field(default_factory=dict)

    def __post_init__(self) -> None:
        quantity = str(self.quantity).strip()
        if not quantity:
            raise ValueError("quantity must not be empty.")
        projection_sign = _validate_projection_sign(self.sign, warn_legacy=True)
        object.__setattr__(self, "quantity", quantity)
        object.__setattr__(self, "phase_axis", int(self.phase_axis))
        object.__setattr__(self, "normalize", bool(self.normalize))
        object.__setattr__(self, "sign", projection_sign)
        object.__setattr__(self, "require_matching_shape", bool(self.require_matching_shape))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "class": self.__class__.__name__,
            "quantity": self.quantity,
            "phase_axis": int(self.phase_axis),
            "normalize": bool(self.normalize),
            "sign": int(self.sign),
            "require_matching_shape": bool(self.require_matching_shape),
            "metadata": dict(self.metadata),
            **phase_projection_convention_metadata(sign=self.sign),
        }


def extract_single_run_quantity(
    result: SingleRunResult,
    quantity: str,
) -> np.ndarray:
    """Legacy runner adapter extracting an array from `SingleRunResult`."""

    if not isinstance(result, SingleRunResult):
        raise TypeError("result must be a SingleRunResult instance.")
    readout = result.readout
    if readout is None:
        raise ValueError("SingleRunResult.readout is required for phase projection.")
    if not isinstance(readout, SingleRunReadoutResult):
        raise TypeError("SingleRunResult.readout must be a SingleRunReadoutResult instance.")

    key = str(quantity).strip()
    if key == "readout.time_fs":
        if readout.time_fs is None:
            raise ValueError("readout.time_fs is not available.")
        return np.asarray(readout.time_fs)
    if key == "readout.polarization_C_per_m2":
        if readout.polarization_C_per_m2 is None:
            raise ValueError("readout.polarization_C_per_m2 is not available.")
        return np.asarray(readout.polarization_C_per_m2)
    if key == "readout.readout_field_MV_per_cm":
        if readout.readout_field_MV_per_cm is None:
            raise ValueError("readout.readout_field_MV_per_cm is not available.")
        return np.asarray(readout.readout_field_MV_per_cm)
    prefix = "readout.spectrum."
    if key.startswith(prefix):
        if readout.spectrum is None:
            raise ValueError("readout.spectrum is not available.")
        spectrum_key = key[len(prefix):]
        if spectrum_key not in readout.spectrum:
            available = sorted(str(item) for item in readout.spectrum)
            raise KeyError(f"spectrum key {spectrum_key!r} is not available. Available keys: {available}")
        return np.asarray(readout.spectrum[spectrum_key])
    raise ValueError(
        "Unsupported quantity. Expected 'readout.time_fs', 'readout.polarization_C_per_m2', "
        "'readout.readout_field_MV_per_cm', or 'readout.spectrum.<key>'. "
        f"Got {quantity!r}."
    )


@dataclass(frozen=True)
class AxisMetadataSpec:
    """Deprecated projected-bundle axis adapter retained for compatibility."""

    name: str
    quantity: str
    source: str = "validate_all_cases"
    rtol: float = 1.0e-9
    atol: float = 1.0e-12
    metadata: dict[str, Any] = dataclass_field(default_factory=dict)

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        quantity = str(self.quantity).strip()
        source = str(self.source).strip()
        if not name:
            raise ValueError("AxisMetadataSpec.name must not be empty.")
        if not quantity:
            raise ValueError("AxisMetadataSpec.quantity must not be empty.")
        if source not in {"first_case", "validate_all_cases"}:
            raise ValueError("AxisMetadataSpec.source must be 'first_case' or 'validate_all_cases'.")
        rtol = float(self.rtol)
        atol = float(self.atol)
        if rtol < 0.0:
            raise ValueError("AxisMetadataSpec.rtol must be >= 0.")
        if atol < 0.0:
            raise ValueError("AxisMetadataSpec.atol must be >= 0.")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "quantity", quantity)
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "rtol", rtol)
        object.__setattr__(self, "atol", atol)
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "class": self.__class__.__name__,
            "name": self.name,
            "quantity": self.quantity,
            "source": self.source,
            "rtol": float(self.rtol),
            "atol": float(self.atol),
            "metadata": dict(self.metadata),
            "compatibility_status": "deprecated; canonical projection uses axis_names/axis_values",
        }


@dataclass
class ProjectedReadoutBundle:
    """Deprecated readout-specific projected wrapper retained through M6."""

    signal_name: str
    signal_quantity: str
    projected_signal: np.ndarray
    axes: dict[str, np.ndarray] = dataclass_field(default_factory=dict)
    phase_result_summary: dict[str, Any] = dataclass_field(default_factory=dict)
    metadata: dict[str, Any] = dataclass_field(default_factory=dict)

    def __post_init__(self) -> None:
        signal_name = str(self.signal_name).strip()
        signal_quantity = str(self.signal_quantity).strip()
        if not signal_name:
            raise ValueError("signal_name must not be empty.")
        if not signal_quantity:
            raise ValueError("signal_quantity must not be empty.")
        axes = {str(name): np.asarray(values) for name, values in self.axes.items()}
        for name in axes:
            if not name.strip():
                raise ValueError("axis name must not be empty.")
        self.signal_name = signal_name
        self.signal_quantity = signal_quantity
        self.projected_signal = np.asarray(self.projected_signal)
        self.axes = axes
        self.phase_result_summary = dict(self.phase_result_summary)
        self.metadata = dict(self.metadata)

    def to_dict(self, *, include_arrays: bool = False) -> dict[str, Any]:
        projected = np.asarray(self.projected_signal)
        payload: dict[str, Any] = {
            "class": self.__class__.__name__,
            "signal_name": self.signal_name,
            "signal_quantity": self.signal_quantity,
            "projected_shape": tuple(projected.shape),
            "projected_dtype": str(projected.dtype),
            "is_complex": bool(np.iscomplexobj(projected)),
            "axis_names": list(self.axes),
            "axis_shapes": {name: tuple(np.asarray(values).shape) for name, values in self.axes.items()},
            "phase_result_summary": dict(self.phase_result_summary),
            "metadata": dict(self.metadata),
            "compatibility_status": (
                "deprecated; canonical output is the lightweight project_phase_orders mapping"
            ),
        }
        if include_arrays:
            payload["projected_signal"] = _json_array(projected)
            payload["axes"] = {
                name: _json_array(np.asarray(values))
                for name, values in self.axes.items()
            }
        return payload


@dataclass
class PhaseCaseRecord:
    """Legacy heavy-runner provenance for one executed phase case."""

    index: int
    case_name: str
    phase_vector: PhaseVector
    single_run_result: SingleRunResult | None = None
    quantity_shape: tuple[int, ...] | None = None
    metadata: dict[str, Any] = dataclass_field(default_factory=dict)

    def to_dict(self, *, include_result: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "class": self.__class__.__name__,
            "index": int(self.index),
            "case_name": self.case_name,
            "phase_vector": dict(self.phase_vector),
            "quantity_shape": None if self.quantity_shape is None else tuple(self.quantity_shape),
            "metadata": dict(self.metadata),
        }
        if include_result and self.single_run_result is not None:
            payload["single_run_result"] = self.single_run_result.to_dict(include_arrays=False)
        return payload


@dataclass
class PhaseCyclingResult:
    """Legacy heavy-runner result retained for compatibility."""

    base_case_name: str
    phase_grid: PhaseGrid
    target_phase_vector: TargetPhaseVector
    projection: PhaseProjectionSpec
    phase_vectors: list[PhaseVector]
    case_records: list[PhaseCaseRecord]
    values: np.ndarray
    projected: np.ndarray
    metadata: dict[str, Any] = dataclass_field(default_factory=dict)

    def to_dict(self, *, include_arrays: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "class": self.__class__.__name__,
            "base_case_name": self.base_case_name,
            "n_phase_cases": len(self.phase_vectors),
            "phase_tags": list(self.phase_grid.tags),
            "target_phase_vector": dict(self.target_phase_vector),
            "quantity": self.projection.quantity,
            "values_shape": tuple(np.asarray(self.values).shape),
            "projected_shape": tuple(np.asarray(self.projected).shape),
            "projection_sign": int(self.projection.sign),
            **phase_projection_convention_metadata(sign=self.projection.sign),
            "normalize": bool(self.projection.normalize),
            "phase_grid": self.phase_grid.to_dict(),
            "projection": self.projection.to_dict(),
            "case_records": [record.to_dict(include_result=False) for record in self.case_records],
            "metadata": dict(self.metadata),
        }
        if include_arrays:
            payload["phase_vectors"] = [dict(item) for item in self.phase_vectors]
            payload["values"] = _json_array(np.asarray(self.values))
            payload["projected"] = _json_array(np.asarray(self.projected))
        return payload


@dataclass
class PhaseCyclingPlan:
    """Legacy orchestration that executes `SingleRunPlan` over a phase grid."""

    base_plan: SingleRunPlan
    phase_grid: PhaseGrid
    target_phase_vector: TargetPhaseVector
    projection: PhaseProjectionSpec
    case_name_template: str = "{base_case_name}_phase_{index:04d}"
    metadata: dict[str, Any] = dataclass_field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.base_plan, SingleRunPlan):
            raise TypeError("base_plan must be a SingleRunPlan instance.")
        if not isinstance(self.phase_grid, PhaseGrid):
            raise TypeError("phase_grid must be a PhaseGrid instance.")
        if not isinstance(self.projection, PhaseProjectionSpec):
            raise TypeError("projection must be a PhaseProjectionSpec instance.")
        if self.base_plan.checkpoint.enabled:
            raise ValueError("PhaseCyclingPlan does not support checkpoint.enabled=True in this milestone.")

        sequence_tags = self.base_plan.field_plan.sequence.phase_tags()
        unknown_grid_tags = sorted(set(self.phase_grid.tags) - set(sequence_tags))
        if unknown_grid_tags:
            raise ValueError(f"phase_grid contains unknown phase tags for base_plan sequence: {unknown_grid_tags}")
        target = normalize_target_phase_vector(
            self.target_phase_vector,
            known_tags=sequence_tags,
            fill_missing_with_zero=True,
        )
        missing_nonzero = sorted(
            tag for tag, coefficient in target.items()
            if coefficient != 0 and tag not in self.phase_grid.tags
        )
        if missing_nonzero:
            raise ValueError(f"target_phase_vector has non-zero tags missing from phase_grid: {missing_nonzero}")

        object.__setattr__(self, "target_phase_vector", target)
        object.__setattr__(self, "case_name_template", str(self.case_name_template))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def base_case_name(self) -> str:
        return self.base_plan.case_name or self.base_plan.field_plan.case_name

    def phase_vectors(self) -> list[PhaseVector]:
        return list(self.phase_grid.iter_phase_vectors())

    def make_case_plan(
        self,
        phase_vector: Mapping[str, float],
        *,
        index: int,
    ) -> SingleRunPlan:
        case_name = self.case_name_template.format(
            base_case_name=self.base_case_name,
            index=int(index),
            phase_vector=dict(phase_vector),
        )
        field_metadata = dict(self.base_plan.field_plan.metadata)
        field_metadata["phase_cycling"] = {
            "base_case_name": self.base_case_name,
            "phase_case_index": int(index),
            "phase_vector": {str(key): float(value) for key, value in phase_vector.items()},
        }
        field_plan = replace(
            self.base_plan.field_plan,
            phase_vector={str(key): float(value) for key, value in phase_vector.items()},
            case_name=case_name,
            metadata=field_metadata,
        )
        input_metadata = dict(self.base_plan.input_metadata)
        input_metadata["phase_cycling"] = {
            "base_case_name": self.base_case_name,
            "phase_case_index": int(index),
            "phase_vector": {str(key): float(value) for key, value in phase_vector.items()},
            "target_phase_vector": dict(self.target_phase_vector),
            "projection": self.projection.to_dict(),
        }
        return replace(
            self.base_plan,
            field_plan=field_plan,
            case_name=case_name,
            input_metadata=input_metadata,
        )

    def execute(
        self,
        *,
        executor: Callable[[SingleRunPlan], SingleRunResult] | None = None,
    ) -> PhaseCyclingResult:
        # Compatibility path: canonical M4 callers project precomputed arrays
        # with project_phase_orders and do not execute this heavy runner.
        run_one = (lambda plan: plan.execute_with_legacy_readout()) if executor is None else executor
        phase_vectors = self.phase_vectors()
        arrays: list[np.ndarray] = []
        records: list[PhaseCaseRecord] = []
        reference_shape: tuple[int, ...] | None = None

        for index, phase_vector in enumerate(phase_vectors):
            case_plan = self.make_case_plan(phase_vector, index=index)
            single_result = run_one(case_plan)
            if not isinstance(single_result, SingleRunResult):
                raise TypeError("phase cycling executor must return a SingleRunResult instance.")
            quantity = np.asarray(extract_single_run_quantity(single_result, self.projection.quantity))
            if reference_shape is None:
                reference_shape = tuple(quantity.shape)
            elif self.projection.require_matching_shape and tuple(quantity.shape) != reference_shape:
                raise ValueError(
                    "All projected quantity arrays must have matching shape. "
                    f"Expected {reference_shape}, got {tuple(quantity.shape)} at phase case {index}."
                )
            arrays.append(quantity)
            records.append(
                PhaseCaseRecord(
                    index=index,
                    case_name=case_plan.case_name,
                    phase_vector=dict(case_plan.field_plan.phase_vector),
                    single_run_result=single_result,
                    quantity_shape=tuple(quantity.shape),
                    metadata={"quantity": self.projection.quantity},
                )
            )

        values = np.stack(arrays, axis=0)
        projected = fourier_project_phase_cases(
            values,
            phase_vectors,
            self.target_phase_vector,
            phase_axis=self.projection.phase_axis,
            normalize=self.projection.normalize,
            sign=self.projection.sign,
        )
        return PhaseCyclingResult(
            base_case_name=self.base_case_name,
            phase_grid=self.phase_grid,
            target_phase_vector=dict(self.target_phase_vector),
            projection=self.projection,
            phase_vectors=phase_vectors,
            case_records=records,
            values=values,
            projected=projected,
            metadata={
                "phase_cycling_plan": self.to_dict(),
                **dict(self.metadata),
            },
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "class": self.__class__.__name__,
            "base_case_name": self.base_case_name,
            "phase_grid": self.phase_grid.to_dict(),
            "target_phase_vector": dict(self.target_phase_vector),
            "projection": self.projection.to_dict(),
            "case_name_template": self.case_name_template,
            "metadata": dict(self.metadata),
        }


def _default_signal_name(quantity: str) -> str:
    text = str(quantity).strip()
    if not text:
        raise ValueError("quantity must not be empty.")
    return text.split(".")[-1]


def _stored_single_run_results(phase_result: PhaseCyclingResult) -> list[SingleRunResult]:
    if not phase_result.case_records:
        raise ValueError("bundle axis extraction requires at least one phase case record.")
    results: list[SingleRunResult] = []
    for record in phase_result.case_records:
        if record.single_run_result is None:
            raise ValueError(
                "bundle axis extraction requires stored single_run_result for every phase case record."
            )
        results.append(record.single_run_result)
    return results


def _extract_axis_from_cases(
    single_run_results: Sequence[SingleRunResult],
    spec: AxisMetadataSpec,
) -> np.ndarray:
    first = np.asarray(extract_single_run_quantity(single_run_results[0], spec.quantity))
    if spec.source == "first_case":
        return first

    for index, result in enumerate(single_run_results[1:], start=1):
        current = np.asarray(extract_single_run_quantity(result, spec.quantity))
        if current.shape != first.shape:
            raise ValueError(
                f"axis metadata mismatch for {spec.name!r} ({spec.quantity}): "
                f"shape {current.shape} at case {index} != {first.shape}."
            )
        if not np.allclose(current, first, rtol=spec.rtol, atol=spec.atol):
            raise ValueError(
                f"axis metadata mismatch for {spec.name!r} ({spec.quantity}) at phase case {index}."
            )
    return first


def build_projected_readout_bundle(
    phase_result: PhaseCyclingResult,
    *,
    signal_name: str | None = None,
    axis_specs: Sequence[AxisMetadataSpec] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> ProjectedReadoutBundle:
    """Build the deprecated readout-specific compatibility bundle.

    axis metadata 只从已有 phase cases 的 `SingleRunResult` 中读取，不做
    Fourier projection，也不重新运行任何 single-run。
    """

    if not isinstance(phase_result, PhaseCyclingResult):
        raise TypeError("phase_result must be a PhaseCyclingResult instance.")
    name = _default_signal_name(phase_result.projection.quantity) if signal_name is None else str(signal_name).strip()
    if not name:
        raise ValueError("signal_name must not be empty.")

    specs = tuple(axis_specs or ())
    for spec in specs:
        if not isinstance(spec, AxisMetadataSpec):
            raise TypeError("axis_specs must contain AxisMetadataSpec instances.")

    axes: dict[str, np.ndarray] = {}
    if specs:
        single_run_results = _stored_single_run_results(phase_result)
        for spec in specs:
            if spec.name in axes:
                raise ValueError(f"duplicate axis metadata name: {spec.name!r}")
            axes[spec.name] = _extract_axis_from_cases(single_run_results, spec)

    bundle_metadata = {
        "axis_specs": [spec.to_dict() for spec in specs],
    }
    bundle_metadata.update(dict(metadata or {}))
    return ProjectedReadoutBundle(
        signal_name=name,
        signal_quantity=phase_result.projection.quantity,
        projected_signal=phase_result.projected,
        axes=axes,
        phase_result_summary=phase_result.to_dict(include_arrays=False),
        metadata=bundle_metadata,
    )


__all__ = [
    "PhaseVector",
    "TargetPhaseVector",
    "PHASE_PROJECTION_CONVENTION",
    "PHASE_PROJECTION_CONVENTION_VERSION",
    "TARGET_PHASE_VECTOR_SEMANTICS",
    "AxisMetadataSpec",
    "PhaseGrid",
    "PhaseProjectionSpec",
    "PhaseCaseRecord",
    "PhaseCyclingPlan",
    "PhaseCyclingResult",
    "ProjectedReadoutBundle",
    "normalize_target_phase_vector",
    "phase_projection_weight",
    "phase_projection_convention_metadata",
    "fourier_project_phase_cases",
    "project_phase_orders",
    "build_uniform_phase_grid",
    "extract_single_run_quantity",
    "build_projected_readout_bundle",
]
