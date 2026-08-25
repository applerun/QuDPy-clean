"""Recipe-first transient-absorption workflow before phase projection.

This module deliberately stops at the complete TA observable ``S(...)``.  It
does not execute generic phase projection, simplify ``PhaseCyclingPlan``, or
define a generic Recipe/Condition framework.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field as dataclass_field
from itertools import product
from typing import Any

import numpy as np

from qudpy_sjh.experiments.pulse_sequence import (
    PhaseGrid,
    PulseSequenceSpec,
    PulseSpec,
    SingleRunFieldPlan,
    SingleRunPlan,
    SingleRunResult,
    normalize_target_phase_vector,
    validate_pulse_name,
)
from qudpy_sjh.experiments.readout import (
    ReadoutPlan,
    ReadoutResult,
    compute_polarization_result,
)
from qudpy_sjh.experiments.ta.ta_recipe_v2 import TADelayCenters
from qudpy_sjh.utils.core import NLevelPhysicalParams, ParaNormalizer
from qudpy_sjh.utils.serialization import json_safe


_DETECTOR_OBSERVABLE = "delta_T_over_T"
_ABSORPTION_COMPATIBILITY_OBSERVABLE = "delta_absorption_like"
_OBSERVABLES = {_DETECTOR_OBSERVABLE, _ABSORPTION_COMPATIBILITY_OBSERVABLE}


def _copy_metadata(metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    return dict(metadata or {})


def _validate_axis(reference: np.ndarray, current: np.ndarray, *, name: str) -> None:
    left = np.asarray(reference)
    right = np.asarray(current)
    if left.shape != right.shape or not np.allclose(left, right, rtol=0.0, atol=1.0e-12):
        raise ValueError(f"Readout axis {name!r} differs between TA condition cases.")


@dataclass
class TAPrePCObservable:
    """Complete recipe-specific TA observable before phase projection."""

    quantity: str
    data: np.ndarray
    difference: np.ndarray
    difference_quantity: str
    axis_names: tuple[str, ...]
    axis_values: dict[str, np.ndarray]
    valid_reference_mask: np.ndarray
    metadata: dict[str, Any] = dataclass_field(default_factory=dict)

    def __post_init__(self) -> None:
        quantity = str(self.quantity).strip()
        difference_quantity = str(self.difference_quantity).strip()
        if not quantity or not difference_quantity:
            raise ValueError("quantity and difference_quantity must not be empty.")
        data = np.asarray(self.data)
        difference = np.asarray(self.difference)
        valid = np.asarray(self.valid_reference_mask, dtype=bool)
        if difference.shape != data.shape or valid.shape != data.shape:
            raise ValueError("difference and valid_reference_mask must match data.shape.")
        names = tuple(str(name).strip() for name in self.axis_names)
        if len(names) != data.ndim or any(not name for name in names):
            raise ValueError("axis_names must provide one non-empty name per data dimension.")
        if len(set(names)) != len(names):
            raise ValueError("axis_names must be unique.")
        values = {str(name): np.asarray(items) for name, items in self.axis_values.items()}
        if set(values) != set(names):
            raise ValueError("axis_values keys must exactly match axis_names.")
        for axis, name in enumerate(names):
            if values[name].ndim != 1 or values[name].size != data.shape[axis]:
                raise ValueError(
                    f"axis_values[{name!r}] must be one-dimensional with length {data.shape[axis]}."
                )
        self.quantity = quantity
        self.data = data
        self.difference = difference
        self.difference_quantity = difference_quantity
        self.axis_names = names
        self.axis_values = values
        self.valid_reference_mask = valid
        self.metadata = _copy_metadata(self.metadata)

    def to_dict(self, *, include_arrays: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "class": self.__class__.__name__,
            "quantity": self.quantity,
            "difference_quantity": self.difference_quantity,
            "data_shape": tuple(self.data.shape),
            "data_dtype": str(self.data.dtype),
            "axis_names": list(self.axis_names),
            "axis_shapes": {name: tuple(values.shape) for name, values in self.axis_values.items()},
            "valid_reference_count": int(np.count_nonzero(self.valid_reference_mask)),
            "invalid_reference_count": int(self.valid_reference_mask.size - np.count_nonzero(self.valid_reference_mask)),
            "metadata": dict(self.metadata),
        }
        if include_arrays:
            payload["data"] = self.data
            payload["difference"] = self.difference
            payload["valid_reference_mask"] = self.valid_reference_mask
            payload["axis_values"] = dict(self.axis_values)
        return json_safe(payload)


@dataclass
class TAPrePCRecipe:
    """Lightweight TA recipe that constructs ``S(T, phases, energy)``.

    ``pump_on`` and ``pump_off`` are condition names for the same material
    system.  Pump-off dynamics depend on cycled probe phase but not delay or an
    absent pump phase, so those results are computed once and broadcast only in
    ``postprocess``.
    """

    base_params: NLevelPhysicalParams
    pump: PulseSpec
    probe: PulseSpec
    delays_fs: tuple[float, ...] | list[float] | np.ndarray
    phase_grid: PhaseGrid
    readout_plan: ReadoutPlan
    observable: str = _DETECTOR_OBSERVABLE
    number_density_m3: float = 1.0e24
    probe_center_fs: float = 0.0
    denominator_rel_threshold: float = 0.0
    denominator_abs_threshold: float = 0.0
    target_phase_vector: dict[str, int] | None = None
    normalizer: ParaNormalizer = dataclass_field(default_factory=ParaNormalizer)
    case_name: str = "ta_pre_pc"
    metadata: dict[str, Any] = dataclass_field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.base_params, NLevelPhysicalParams):
            raise TypeError("base_params must be a NLevelPhysicalParams instance.")
        if not isinstance(self.pump, PulseSpec) or not isinstance(self.probe, PulseSpec):
            raise TypeError("pump and probe must be PulseSpec instances.")
        if self.pump.name == self.probe.name:
            raise ValueError("pump.name and probe.name must be distinct.")
        if not isinstance(self.phase_grid, PhaseGrid):
            raise TypeError("phase_grid must be a PhaseGrid instance.")
        if not isinstance(self.readout_plan, ReadoutPlan):
            raise TypeError("readout_plan must be a ReadoutPlan instance.")
        delays = tuple(float(value) for value in self.delays_fs)
        if not delays or not all(np.isfinite(value) for value in delays):
            raise ValueError("delays_fs must contain at least one finite value.")
        density = float(self.number_density_m3)
        if density <= 0.0:
            raise ValueError("number_density_m3 must be > 0.")
        probe_center = float(self.probe_center_fs)
        if not np.isfinite(probe_center):
            raise ValueError("probe_center_fs must be finite.")
        rel_threshold = float(self.denominator_rel_threshold)
        abs_threshold = float(self.denominator_abs_threshold)
        if rel_threshold < 0.0 or abs_threshold < 0.0:
            raise ValueError("denominator thresholds must be >= 0.")
        observable = str(self.observable).strip()
        if observable not in _OBSERVABLES:
            raise ValueError(f"observable must be one of {sorted(_OBSERVABLES)}.")
        self._validate_readout_mode(self.readout_plan, observable=observable)
        known_tags = tuple(
            tag for tag in (self.pump.phase_tag, self.probe.phase_tag)
            if tag is not None
        )
        unknown_tags = sorted(set(self.phase_grid.tags) - set(known_tags))
        if unknown_tags:
            raise ValueError(f"phase_grid contains tags not owned by pump/probe pulses: {unknown_tags}")
        target = None
        if self.target_phase_vector is not None:
            target = normalize_target_phase_vector(
                self.target_phase_vector,
                known_tags=self.phase_grid.tags,
                fill_missing_with_zero=True,
            )
        if not isinstance(self.normalizer, ParaNormalizer):
            raise TypeError("normalizer must be a ParaNormalizer instance.")
        self.delays_fs = delays
        self.number_density_m3 = density
        self.probe_center_fs = probe_center
        self.denominator_rel_threshold = rel_threshold
        self.denominator_abs_threshold = abs_threshold
        self.observable = observable
        self.target_phase_vector = target
        self.case_name = validate_pulse_name(self.case_name)
        self.metadata = _copy_metadata(self.metadata)

    @staticmethod
    def _validate_readout_mode(readout_plan: ReadoutPlan, *, observable: str) -> None:
        if observable == _DETECTOR_OBSERVABLE and readout_plan.mode not in {"full", "weak"}:
            raise ValueError("delta_T_over_T requires ReadoutPlan mode 'full' or 'weak'.")
        if observable == _ABSORPTION_COMPATIBILITY_OBSERVABLE and readout_plan.mode != "absorption_like":
            raise ValueError("delta_absorption_like requires ReadoutPlan mode 'absorption_like'.")

    @property
    def pump_phase_tag(self) -> str | None:
        return self.pump.phase_tag

    @property
    def probe_phase_tag(self) -> str | None:
        return self.probe.phase_tag

    @property
    def phase_shape(self) -> tuple[int, ...]:
        return tuple(len(self.phase_grid.phases_by_tag[tag]) for tag in self.phase_grid.tags)

    def build_condition_sequences(self) -> dict[str, PulseSequenceSpec]:
        """Build one shared sequence definition per TA condition."""

        return {
            "pump_on": PulseSequenceSpec(
                name=f"{self.case_name}_pump_on_sequence",
                pulses=(self.pump, self.probe),
                metadata={"recipe": "TAPrePCRecipe", "condition": "pump_on"},
            ),
            "pump_off": PulseSequenceSpec(
                name=f"{self.case_name}_pump_off_sequence",
                pulses=(self.probe,),
                metadata={"recipe": "TAPrePCRecipe", "condition": "pump_off"},
            ),
        }

    def _phase_vector(self, phase_index: tuple[int, ...]) -> dict[str, float]:
        return {
            tag: float(self.phase_grid.phases_by_tag[tag][index])
            for tag, index in zip(self.phase_grid.tags, phase_index)
        }

    def _pump_off_key(self, phase_index: tuple[int, ...]) -> tuple[int, ...]:
        probe_tag = self.probe_phase_tag
        if probe_tag is None or probe_tag not in self.phase_grid.tags:
            return ()
        probe_axis = self.phase_grid.tags.index(probe_tag)
        return (int(phase_index[probe_axis]),)

    def _pump_off_phase_vector(self, off_key: tuple[int, ...]) -> dict[str, float]:
        probe_tag = self.probe_phase_tag
        if not off_key or probe_tag is None:
            return {}
        return {probe_tag: float(self.phase_grid.phases_by_tag[probe_tag][off_key[0]])}

    def _make_plan(
        self,
        *,
        sequence: PulseSequenceSpec,
        condition: str,
        centers_fs: dict[str, float],
        phase_vector: dict[str, float],
        case_name: str,
        recipe_coordinates: dict[str, Any],
    ) -> SingleRunPlan:
        metadata = {
            "recipe": "TAPrePCRecipe",
            "recipe_case_name": self.case_name,
            "condition": condition,
            "recipe_coordinates": dict(recipe_coordinates),
            "phase_vector": dict(phase_vector),
        }
        return SingleRunPlan(
            base_params=self.base_params,
            field_plan=SingleRunFieldPlan(
                sequence=sequence,
                centers_fs=centers_fs,
                phase_vector=phase_vector,
                case_name=case_name,
                metadata=metadata,
            ),
            normalizer=self.normalizer,
            case_name=case_name,
            input_metadata={"ta_recipe_first": metadata},
        )

    def build_dynamics_plans(self) -> dict[str, Any]:
        """Enumerate only physically distinct dynamics cases."""

        sequences = self.build_condition_sequences()
        phase_indices = tuple(product(*(range(size) for size in self.phase_shape)))
        pump_on: dict[tuple[int, ...], SingleRunPlan] = {}
        pump_off: dict[tuple[int, ...], SingleRunPlan] = {}
        case_metadata: dict[str, dict[tuple[int, ...], dict[str, Any]]] = {
            "pump_on": {},
            "pump_off": {},
        }
        for delay_index, delay_fs in enumerate(self.delays_fs):
            centers = TADelayCenters(delay_fs=delay_fs, probe_center_fs=self.probe_center_fs)
            for phase_index in phase_indices:
                phase_vector = self._phase_vector(phase_index)
                key = (delay_index, *phase_index)
                case_name = (
                    f"{self.case_name}_pump_on_T{delay_index:03d}_"
                    f"phase_{'_'.join(str(value) for value in phase_index)}"
                )
                recipe_coordinates = {
                    "T_fs": float(delay_fs),
                    "delay_index": int(delay_index),
                    "phase_index": tuple(int(value) for value in phase_index),
                }
                pump_on[key] = self._make_plan(
                    sequence=sequences["pump_on"],
                    condition="pump_on",
                    centers_fs={
                        self.pump.name: centers.pump_center_fs,
                        self.probe.name: centers.probe_center_fs,
                    },
                    phase_vector=phase_vector,
                    case_name=case_name,
                    recipe_coordinates=recipe_coordinates,
                )
                case_metadata["pump_on"][key] = {
                    "condition": "pump_on",
                    **recipe_coordinates,
                    "phase_vector": phase_vector,
                    "pulse_centers_fs": {
                        self.pump.name: centers.pump_center_fs,
                        self.probe.name: centers.probe_center_fs,
                    },
                }

                off_key = self._pump_off_key(phase_index)
                if off_key in pump_off:
                    continue
                off_phase_vector = self._pump_off_phase_vector(off_key)
                off_case_name = (
                    f"{self.case_name}_pump_off"
                    if not off_key
                    else f"{self.case_name}_pump_off_probe_phase_{off_key[0]:03d}"
                )
                off_coordinates = {
                    "depends_on_T": False,
                    "depends_on_pump_phase": False,
                    "probe_phase_index": None if not off_key else int(off_key[0]),
                }
                pump_off[off_key] = self._make_plan(
                    sequence=sequences["pump_off"],
                    condition="pump_off",
                    centers_fs={self.probe.name: self.probe_center_fs},
                    phase_vector=off_phase_vector,
                    case_name=off_case_name,
                    recipe_coordinates=off_coordinates,
                )
                case_metadata["pump_off"][off_key] = {
                    "condition": "pump_off",
                    **off_coordinates,
                    "phase_vector": off_phase_vector,
                    "pulse_centers_fs": {self.probe.name: self.probe_center_fs},
                }
        return {
            "pump_on": pump_on,
            "pump_off": pump_off,
            "case_metadata": case_metadata,
            "sequence_definitions": sequences,
            "reuse_policy": {
                "pump_on_key": "(delay_index, *phase_indices)",
                "pump_off_key": "(probe_phase_index,) or () when probe is not cycled",
                "pump_off_broadcast_over": ["T", "pump phase"],
            },
        }

    def execute_dynamics(
        self,
        *,
        executor: Callable[[SingleRunPlan], SingleRunResult] | None = None,
    ) -> dict[str, Any]:
        """Execute each physically distinct SimRes once.

        A caller-supplied executor may apply the existing checkpoint policy; its
        cache key should be the supplied plan's case name, which contains only
        dynamics-defining coordinates.
        """

        plans = self.build_dynamics_plans()
        run_one = (lambda plan: plan.execute()) if executor is None else executor
        dynamics: dict[str, dict[tuple[int, ...], SingleRunResult]] = {
            "pump_on": {},
            "pump_off": {},
        }
        for condition in ("pump_on", "pump_off"):
            for key, plan in plans[condition].items():
                result = run_one(plan)
                if not isinstance(result, SingleRunResult):
                    raise TypeError("TA dynamics executor must return a SingleRunResult instance.")
                dynamics[condition][key] = result
        return {
            **dynamics,
            "case_metadata": plans["case_metadata"],
            "reuse_policy": plans["reuse_policy"],
            "solver_case_counts": {
                "pump_on": len(dynamics["pump_on"]),
                "pump_off": len(dynamics["pump_off"]),
                "total": len(dynamics["pump_on"]) + len(dynamics["pump_off"]),
            },
        }

    def apply_readout(
        self,
        dynamics_cases: Mapping[str, Any],
        *,
        readout_plan: ReadoutPlan | None = None,
    ) -> dict[str, Any]:
        """Apply a cheap detector plan to reusable SimRes maps."""

        plan = self.readout_plan if readout_plan is None else readout_plan
        if not isinstance(plan, ReadoutPlan):
            raise TypeError("readout_plan must be a ReadoutPlan instance.")
        self._validate_readout_mode(plan, observable=self.observable)
        readouts: dict[str, dict[tuple[int, ...], ReadoutResult]] = {
            "pump_on": {},
            "pump_off": {},
        }
        for condition in ("pump_on", "pump_off"):
            source = dynamics_cases.get(condition)
            if not isinstance(source, Mapping):
                raise TypeError(f"dynamics_cases[{condition!r}] must be a mapping.")
            for key, result in source.items():
                if not isinstance(result, SingleRunResult):
                    raise TypeError("dynamics case values must be SingleRunResult instances.")
                polarization = compute_polarization_result(
                    result.dynamics_result,
                    number_density_m3=self.number_density_m3,
                )
                readouts[condition][key] = plan.execute(
                    polarization,
                    interaction_field=result.params.field,
                )
        return {
            **readouts,
            "case_metadata": dynamics_cases.get("case_metadata", {}),
            "reuse_policy": dynamics_cases.get("reuse_policy", {}),
            "solver_case_counts": dynamics_cases.get("solver_case_counts", {}),
            "readout_plan": plan,
        }

    def postprocess(self, readout_cases: Mapping[str, Any]) -> TAPrePCObservable:
        """Combine condition ReadoutResults into the complete pre-PC TA ``S``."""

        return build_ta_pre_pc_observable(self, readout_cases)

    def execute(
        self,
        *,
        executor: Callable[[SingleRunPlan], SingleRunResult] | None = None,
    ) -> TAPrePCObservable:
        dynamics = self.execute_dynamics(executor=executor)
        readouts = self.apply_readout(dynamics)
        return self.postprocess(readouts)

    def to_dict(self) -> dict[str, Any]:
        return json_safe({
            "class": self.__class__.__name__,
            "case_name": self.case_name,
            "delays_fs": [float(value) for value in self.delays_fs],
            "probe_center_fs": float(self.probe_center_fs),
            "delay_convention": "pump_center_fs = probe_center_fs - T; positive T means pump before probe",
            "phase_grid": self.phase_grid.to_dict(),
            "readout_plan": self.readout_plan.to_dict(),
            "observable": self.observable,
            "number_density_m3": float(self.number_density_m3),
            "denominator_policy": {
                "relative_threshold": float(self.denominator_rel_threshold),
                "absolute_threshold": float(self.denominator_abs_threshold),
                "default": "exact zero only when both thresholds are zero",
            },
            "target_phase_vector": None if self.target_phase_vector is None else dict(self.target_phase_vector),
            "metadata": dict(self.metadata),
        })


def _readout_values(
    result: ReadoutResult,
    *,
    observable: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    if not isinstance(result, ReadoutResult):
        raise TypeError("TA postprocess inputs must be ReadoutResult instances.")
    if result.spectrum is None:
        raise ValueError("TA postprocess requires spectral ReadoutResults.")
    key = "detector_intensity" if observable == _DETECTOR_OBSERVABLE else "absorption_like_response"
    if key not in result.spectrum or "energy_eV" not in result.spectrum:
        raise KeyError(f"ReadoutResult spectrum must contain {key!r} and 'energy_eV'.")
    values = np.asarray(result.spectrum[key])
    energy = np.asarray(result.spectrum["energy_eV"], dtype=float)
    omega = (
        None
        if "omega_fs_inv" not in result.spectrum
        else np.asarray(result.spectrum["omega_fs_inv"], dtype=float)
    )
    if values.ndim != 1 or values.shape != energy.shape:
        raise ValueError("TA spectral readout values must be one-dimensional and match energy_eV.")
    if omega is not None and omega.shape != energy.shape:
        raise ValueError("omega_fs_inv must match energy_eV.")
    return values, energy, omega


def build_ta_pre_pc_observable(
    recipe: TAPrePCRecipe,
    readout_cases: Mapping[str, Any],
) -> TAPrePCObservable:
    """Build ``S(T, phase dimensions, energy)`` from TA condition readouts."""

    if not isinstance(recipe, TAPrePCRecipe):
        raise TypeError("recipe must be a TAPrePCRecipe instance.")
    pump_on = readout_cases.get("pump_on")
    pump_off = readout_cases.get("pump_off")
    if not isinstance(pump_on, Mapping) or not isinstance(pump_off, Mapping):
        raise TypeError("readout_cases must contain pump_on and pump_off mappings.")
    expected_on = len(recipe.delays_fs) * int(np.prod(recipe.phase_shape))
    if len(pump_on) != expected_on:
        raise ValueError(f"pump_on readout count must be {expected_on}; got {len(pump_on)}.")

    first_result = next(iter(pump_on.values()))
    first_values, energy_eV, omega_fs_inv = _readout_values(
        first_result,
        observable=recipe.observable,
    )
    output_shape = (len(recipe.delays_fs), *recipe.phase_shape, energy_eV.size)
    pump_on_values = np.empty(output_shape, dtype=np.result_type(first_values.dtype, np.float64))
    pump_off_values = np.empty_like(pump_on_values)
    phase_indices = tuple(product(*(range(size) for size in recipe.phase_shape)))

    for delay_index, _delay in enumerate(recipe.delays_fs):
        for phase_index in phase_indices:
            on_key = (delay_index, *phase_index)
            if on_key not in pump_on:
                raise KeyError(f"Missing pump_on readout key: {on_key}")
            on_values, local_energy, local_omega = _readout_values(
                pump_on[on_key],
                observable=recipe.observable,
            )
            _validate_axis(energy_eV, local_energy, name="energy_eV")
            if (omega_fs_inv is None) != (local_omega is None):
                raise ValueError("omega_fs_inv availability differs between TA cases.")
            if omega_fs_inv is not None and local_omega is not None:
                _validate_axis(omega_fs_inv, local_omega, name="omega_fs_inv")
            off_key = recipe._pump_off_key(phase_index)
            if off_key not in pump_off:
                raise KeyError(f"Missing pump_off readout key: {off_key}")
            off_values, off_energy, off_omega = _readout_values(
                pump_off[off_key],
                observable=recipe.observable,
            )
            _validate_axis(energy_eV, off_energy, name="energy_eV")
            if (omega_fs_inv is None) != (off_omega is None):
                raise ValueError("omega_fs_inv availability differs between TA cases.")
            if omega_fs_inv is not None and off_omega is not None:
                _validate_axis(omega_fs_inv, off_omega, name="omega_fs_inv")
            target = (delay_index, *phase_index, slice(None))
            pump_on_values[target] = on_values
            pump_off_values[target] = off_values

    difference = pump_on_values - pump_off_values
    if recipe.observable == _DETECTOR_OBSERVABLE:
        reference_abs = np.abs(pump_off_values)
        reference_scale = np.max(reference_abs, axis=-1, keepdims=True)
        threshold = np.maximum(
            recipe.denominator_abs_threshold,
            recipe.denominator_rel_threshold * reference_scale,
        )
        valid = reference_abs > threshold
        data = np.full(difference.shape, np.nan, dtype=np.result_type(difference.dtype, np.float64))
        np.divide(difference, pump_off_values, out=data, where=valid)
        difference_quantity = "delta_I"
        formula = "delta_T_over_T = (I_on - I_off) / I_off"
    else:
        valid = np.ones(difference.shape, dtype=bool)
        data = difference.copy()
        difference_quantity = _ABSORPTION_COMPATIBILITY_OBSERVABLE
        formula = "delta_absorption_like = A_on - A_off; not detector-level deltaT/T"

    axis_names = (
        "T",
        *(f"phase:{tag}" for tag in recipe.phase_grid.tags),
        "energy_eV",
    )
    axis_values = {
        "T": np.asarray(recipe.delays_fs, dtype=float),
        **{
            f"phase:{tag}": np.asarray(recipe.phase_grid.phases_by_tag[tag], dtype=float)
            for tag in recipe.phase_grid.tags
        },
        "energy_eV": energy_eV,
    }
    invalid_count = int(valid.size - np.count_nonzero(valid))
    applied_readout_plan = readout_cases.get("readout_plan", recipe.readout_plan)
    if not isinstance(applied_readout_plan, ReadoutPlan):
        raise TypeError("readout_cases['readout_plan'] must be a ReadoutPlan instance.")
    recipe._validate_readout_mode(applied_readout_plan, observable=recipe.observable)
    return TAPrePCObservable(
        quantity=recipe.observable,
        data=data,
        difference=difference,
        difference_quantity=difference_quantity,
        axis_names=axis_names,
        axis_values=axis_values,
        valid_reference_mask=valid,
        metadata={
            "recipe": recipe.to_dict(),
            "condition_formula": formula,
            "condition_sequences": {
                "pump_on": "pump + probe",
                "pump_off": "probe only",
            },
            "readout_mode": applied_readout_plan.mode,
            "pump_on_readout_count": len(pump_on),
            "pump_off_readout_count": len(pump_off),
            "pump_off_reuse": "computed per physically dependent probe phase, broadcast over T and pump phase",
            "denominator_policy": {
                "relative_threshold": float(recipe.denominator_rel_threshold),
                "absolute_threshold": float(recipe.denominator_abs_threshold),
                "invalid_points_are_nan": True,
                "invalid_count": invalid_count,
                "warning": None if invalid_count == 0 else "I_off contains zero/threshold-invalid spectral points.",
            },
            "omega_fs_inv": None if omega_fs_inv is None else omega_fs_inv,
            "target_phase_vector": (
                None if recipe.target_phase_vector is None else dict(recipe.target_phase_vector)
            ),
            "phase_projection_status": "not_applied_pre_pc_observable",
            "solver_case_counts": dict(readout_cases.get("solver_case_counts", {})),
            **dict(recipe.metadata),
        },
    )


__all__ = [
    "TAPrePCObservable",
    "TAPrePCRecipe",
    "build_ta_pre_pc_observable",
]
