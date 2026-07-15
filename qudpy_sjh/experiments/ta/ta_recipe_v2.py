"""Minimal TA recipe v2 scaffold built on generic pulse-sequence layers.

本模块只表达最小 TA recipe v2 编排：

    pump/probe physical pulses
    -> pump-probe SingleRunPlan
    -> probe-only SingleRunPlan
    -> probe-channel absorption-like readout bundle
    -> optional single-delay contrast
    -> optional delay-energy scan map

TA subtraction 只由显式 `compute_ta_contrast(...)` 执行，固定 convention 为
`S_TA = S_pump_probe - S_probe_only`。delay scan 只把多个单 delay contrast
按输入 delay 顺序堆叠成 delay × energy map，不做排序、插值或重采样。
当前不实现 phase-cycling TA、TAResultIO v2、绘图、落盘或旧 demo 迁移。
readout 不是第三个激发脉冲；probe 既是 physical probe pulse，也是
`ReadoutSpec(readout_field_name=probe.name)` 的 reference field。
"""

from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
from collections.abc import Mapping
from typing import Any

import numpy as np

from qudpy_sjh.experiments.pulse_sequence import (
    AxisMetadataSpec,
    PhaseCyclingPlan,
    PhaseCyclingResult,
    PhaseGrid,
    PhaseProjectionSpec,
    ProjectedReadoutBundle,
    PulseSequenceSpec,
    PulseSpec,
    ReadoutSpec,
    SingleRunCheckpointSettings,
    SingleRunFieldPlan,
    SingleRunPlan,
    SingleRunResult,
    build_projected_readout_bundle,
)
from qudpy_sjh.experiments.pulse_sequence.pulse_sequence import validate_phase_tag, validate_pulse_name
from qudpy_sjh.utils.core import NLevelPhysicalParams, ParaNormalizer


def _energy_range_eV(energy_eV: np.ndarray) -> tuple[float, float] | None:
    energy = np.asarray(energy_eV, dtype=float)
    if energy.size == 0:
        return None
    return float(np.min(energy)), float(np.max(energy))


def _single_run_summary(result: SingleRunResult) -> dict[str, Any]:
    summary: dict[str, Any] = {"case_name": result.case_name}
    if result.dynamics_result is not None:
        summary.update(
            {
                "max_trace_error": float(result.dynamics_result.max_trace_error()),
                "max_hermiticity_error": float(result.dynamics_result.max_hermiticity_error()),
            }
        )
    return summary


def _safe_case_value(value: float) -> str:
    return f"{float(value):.6g}".replace("-", "m").replace(".", "p").replace(" ", "_")


@dataclass(frozen=True)
class TADelayCenters:
    """单个 TA delay 的 pump/probe center 约定。

    `delay_fs = probe_center_fs - pump_center_fs`，因此正 delay 表示 pump
    先于 probe 到达。
    """

    delay_fs: float
    probe_center_fs: float = 0.0

    def __post_init__(self) -> None:
        delay = float(self.delay_fs)
        probe_center = float(self.probe_center_fs)
        if not np.isfinite(delay):
            raise ValueError("delay_fs must be finite.")
        if not np.isfinite(probe_center):
            raise ValueError("probe_center_fs must be finite.")
        object.__setattr__(self, "delay_fs", delay)
        object.__setattr__(self, "probe_center_fs", probe_center)

    @property
    def pump_center_fs(self) -> float:
        return float(self.probe_center_fs - self.delay_fs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "class": self.__class__.__name__,
            "delay_fs": float(self.delay_fs),
            "probe_center_fs": float(self.probe_center_fs),
            "pump_center_fs": float(self.pump_center_fs),
            "delay_convention": "delay_fs = probe_center_fs - pump_center_fs; positive delay means pump before probe",
        }


@dataclass
class TAReadoutBundle:
    """TA recipe v2 的 absorption-like readout bundle。

    这里只打包单条 `SingleRunResult` 的 absorption 与频率轴，不做
    pump-probe minus probe-only subtraction。
    """

    case_name: str
    absorption: np.ndarray
    energy_eV: np.ndarray
    omega_fs_inv: np.ndarray | None = None
    metadata: dict[str, Any] = dataclass_field(default_factory=dict)

    def __post_init__(self) -> None:
        case_name = validate_pulse_name(self.case_name)
        absorption = np.asarray(self.absorption)
        energy = np.asarray(self.energy_eV, dtype=float)
        if absorption.shape != energy.shape:
            raise ValueError(
                f"absorption shape must match energy_eV shape. Got {absorption.shape} and {energy.shape}."
            )
        omega = None if self.omega_fs_inv is None else np.asarray(self.omega_fs_inv, dtype=float)
        if omega is not None and omega.shape != energy.shape:
            raise ValueError(
                f"omega_fs_inv shape must match energy_eV shape. Got {omega.shape} and {energy.shape}."
            )
        self.case_name = case_name
        self.absorption = absorption
        self.energy_eV = energy
        self.omega_fs_inv = omega
        self.metadata = dict(self.metadata)

    def to_dict(self, *, include_arrays: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "class": self.__class__.__name__,
            "case_name": self.case_name,
            "n_points": int(self.energy_eV.size),
            "energy_range_eV": _energy_range_eV(self.energy_eV),
            "has_omega_fs_inv": self.omega_fs_inv is not None,
            "absorption_shape": tuple(self.absorption.shape),
            "absorption_dtype": str(self.absorption.dtype),
            "metadata": dict(self.metadata),
        }
        if include_arrays:
            payload["absorption"] = self.absorption.tolist()
            payload["energy_eV"] = self.energy_eV.tolist()
            payload["omega_fs_inv"] = None if self.omega_fs_inv is None else self.omega_fs_inv.tolist()
        return payload


@dataclass(frozen=True)
class TASubtractionSpec:
    """单 delay TA subtraction 配置。

    subtraction convention 固定为：

        S_TA = S_pump_probe - S_probe_only

    当前不实现 OD、relative difference、缩放、平滑、插值、重采样或 sign flip。
    """

    signal_name: str = "delta_absorption"
    rtol: float = 1.0e-9
    atol: float = 1.0e-12
    validate_omega_axis: bool = True
    metadata: dict[str, Any] = dataclass_field(default_factory=dict)

    def __post_init__(self) -> None:
        signal_name = str(self.signal_name).strip()
        if not signal_name:
            raise ValueError("signal_name must not be empty.")
        rtol = float(self.rtol)
        atol = float(self.atol)
        if rtol < 0.0:
            raise ValueError("rtol must be >= 0.")
        if atol < 0.0:
            raise ValueError("atol must be >= 0.")
        object.__setattr__(self, "signal_name", signal_name)
        object.__setattr__(self, "rtol", rtol)
        object.__setattr__(self, "atol", atol)
        object.__setattr__(self, "validate_omega_axis", bool(self.validate_omega_axis))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "class": self.__class__.__name__,
            "signal_name": self.signal_name,
            "rtol": float(self.rtol),
            "atol": float(self.atol),
            "validate_omega_axis": bool(self.validate_omega_axis),
            "convention": "pump_probe_minus_probe_only",
            "metadata": dict(self.metadata),
        }


@dataclass
class TAContrastResult:
    """单 delay TA contrast 结果。

    `delta_absorption` 按固定 convention `pump_probe - probe_only` 得到。
    """

    case_name: str
    delay_fs: float
    signal_name: str
    delta_absorption: np.ndarray
    energy_eV: np.ndarray
    omega_fs_inv: np.ndarray | None = None
    pump_probe_bundle: TAReadoutBundle | None = None
    probe_only_bundle: TAReadoutBundle | None = None
    subtraction: TASubtractionSpec = dataclass_field(default_factory=TASubtractionSpec)
    metadata: dict[str, Any] = dataclass_field(default_factory=dict)

    def __post_init__(self) -> None:
        self.case_name = validate_pulse_name(self.case_name)
        signal_name = str(self.signal_name).strip()
        if not signal_name:
            raise ValueError("signal_name must not be empty.")
        delay = float(self.delay_fs)
        if not np.isfinite(delay):
            raise ValueError("delay_fs must be finite.")
        delta = np.asarray(self.delta_absorption)
        energy = np.asarray(self.energy_eV, dtype=float)
        if delta.shape != energy.shape:
            raise ValueError(
                f"delta_absorption shape must match energy_eV shape. Got {delta.shape} and {energy.shape}."
            )
        omega = None if self.omega_fs_inv is None else np.asarray(self.omega_fs_inv, dtype=float)
        if omega is not None and omega.shape != energy.shape:
            raise ValueError(
                f"omega_fs_inv shape must match energy_eV shape. Got {omega.shape} and {energy.shape}."
            )
        if self.pump_probe_bundle is not None and not isinstance(self.pump_probe_bundle, TAReadoutBundle):
            raise TypeError("pump_probe_bundle must be a TAReadoutBundle instance or None.")
        if self.probe_only_bundle is not None and not isinstance(self.probe_only_bundle, TAReadoutBundle):
            raise TypeError("probe_only_bundle must be a TAReadoutBundle instance or None.")
        if not isinstance(self.subtraction, TASubtractionSpec):
            raise TypeError("subtraction must be a TASubtractionSpec instance.")
        self.signal_name = signal_name
        self.delay_fs = delay
        self.delta_absorption = delta
        self.energy_eV = energy
        self.omega_fs_inv = omega
        self.metadata = dict(self.metadata)

    def to_dict(self, *, include_arrays: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "class": self.__class__.__name__,
            "case_name": self.case_name,
            "delay_fs": float(self.delay_fs),
            "signal_name": self.signal_name,
            "n_points": int(self.energy_eV.size),
            "energy_range_eV": _energy_range_eV(self.energy_eV),
            "has_omega_fs_inv": self.omega_fs_inv is not None,
            "delta_absorption_shape": tuple(self.delta_absorption.shape),
            "delta_absorption_dtype": str(self.delta_absorption.dtype),
            "subtraction": self.subtraction.to_dict(),
            "source_cases": {
                "pump_probe": None if self.pump_probe_bundle is None else self.pump_probe_bundle.case_name,
                "probe_only": None if self.probe_only_bundle is None else self.probe_only_bundle.case_name,
            },
            "metadata": dict(self.metadata),
        }
        if include_arrays:
            payload["delta_absorption"] = self.delta_absorption.tolist()
            payload["energy_eV"] = self.energy_eV.tolist()
            payload["omega_fs_inv"] = None if self.omega_fs_inv is None else self.omega_fs_inv.tolist()
        return payload


def _default_ta_phase_axis_specs() -> tuple[AxisMetadataSpec, ...]:
    return (
        AxisMetadataSpec(
            name="energy_eV",
            quantity="readout.spectrum.energy_eV",
            source="validate_all_cases",
        ),
    )


@dataclass(frozen=True)
class TAPhaseCyclingSpec:
    """TA recipe v2 的可选 pump-probe phase-cycling 配置。

    当前 scaffold 只对 pump-probe readout quantity 做 phase projection。
    `target_phase_vector` 必须由用户或上层 recipe 显式传入；这里不定义通用
    TA phase convention，也不默认使用任何固定 target vector。probe 在当前
    minimal TA recipe 中仍同时是 physical probe pulse 与 readout field
    reference；readout / LO 不是第三个激发脉冲。
    """

    phase_grid: PhaseGrid
    target_phase_vector: dict[str, int]
    projection_quantity: str = "readout.spectrum.absorption"
    signal_name: str = "phase_projected_absorption"
    axis_specs: tuple[AxisMetadataSpec, ...] = dataclass_field(default_factory=_default_ta_phase_axis_specs)
    normalize: bool = True
    sign: int = -1
    metadata: dict[str, Any] = dataclass_field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.phase_grid, PhaseGrid):
            raise TypeError("phase_grid must be a PhaseGrid instance.")
        if not self.target_phase_vector:
            raise ValueError("target_phase_vector must not be empty.")
        target: dict[str, int] = {}
        for key, value in self.target_phase_vector.items():
            tag = validate_phase_tag(key, allow_none=False)
            assert tag is not None
            coefficient = int(value)
            if float(value) != float(coefficient):
                raise ValueError(f"target_phase_vector coefficient for {tag!r} must be an integer.")
            target[tag] = coefficient
        projection_quantity = str(self.projection_quantity).strip()
        signal_name = str(self.signal_name).strip()
        if not projection_quantity:
            raise ValueError("projection_quantity must not be empty.")
        if not signal_name:
            raise ValueError("signal_name must not be empty.")
        axis_specs = tuple(self.axis_specs)
        for spec in axis_specs:
            if not isinstance(spec, AxisMetadataSpec):
                raise TypeError("axis_specs must contain only AxisMetadataSpec instances.")
        sign = int(self.sign)
        if sign not in {-1, 1}:
            raise ValueError("sign must be +1 or -1.")
        object.__setattr__(self, "target_phase_vector", target)
        object.__setattr__(self, "projection_quantity", projection_quantity)
        object.__setattr__(self, "signal_name", signal_name)
        object.__setattr__(self, "axis_specs", axis_specs)
        object.__setattr__(self, "normalize", bool(self.normalize))
        object.__setattr__(self, "sign", sign)
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "class": self.__class__.__name__,
            "phase_grid": self.phase_grid.to_dict(),
            "target_phase_vector": dict(self.target_phase_vector),
            "projection_quantity": self.projection_quantity,
            "signal_name": self.signal_name,
            "axis_specs": [spec.to_dict() for spec in self.axis_specs],
            "normalize": bool(self.normalize),
            "sign": int(self.sign),
            "metadata": dict(self.metadata),
            "ta_phase_cycling_scope": {
                "scope": "pump_probe_readout_projection_only",
                "target_phase_vector": "explicit_user_or_recipe_input",
                "no_universal_ta_phase_convention": True,
                "no_phase_cycled_ta_subtraction": True,
            },
        }


@dataclass
class TAPhaseCycledPumpProbeResult:
    """单 delay pump-probe phase projection 的结果容器。

    本结果只保存 phase-projected pump-probe readout bundle，不做
    pump-probe minus probe-only subtraction，不写 TA map，也不保存文件。
    """

    case_name: str
    delay_fs: float
    phase_cycling: TAPhaseCyclingSpec
    phase_result: PhaseCyclingResult
    bundle: ProjectedReadoutBundle
    metadata: dict[str, Any] = dataclass_field(default_factory=dict)

    def __post_init__(self) -> None:
        self.case_name = validate_pulse_name(self.case_name)
        delay = float(self.delay_fs)
        if not np.isfinite(delay):
            raise ValueError("delay_fs must be finite.")
        if not isinstance(self.phase_cycling, TAPhaseCyclingSpec):
            raise TypeError("phase_cycling must be a TAPhaseCyclingSpec instance.")
        if not isinstance(self.phase_result, PhaseCyclingResult):
            raise TypeError("phase_result must be a PhaseCyclingResult instance.")
        if not isinstance(self.bundle, ProjectedReadoutBundle):
            raise TypeError("bundle must be a ProjectedReadoutBundle instance.")
        self.delay_fs = delay
        self.metadata = dict(self.metadata)

    def to_dict(self, *, include_arrays: bool = False) -> dict[str, Any]:
        return {
            "class": self.__class__.__name__,
            "case_name": self.case_name,
            "delay_fs": float(self.delay_fs),
            "phase_cycling": self.phase_cycling.to_dict(),
            "phase_result": self.phase_result.to_dict(include_arrays=False),
            "bundle": self.bundle.to_dict(include_arrays=include_arrays),
            "metadata": dict(self.metadata),
            "ta_phase_cycling_scope": "pump_probe_projection_only; no TA subtraction",
        }


def validate_ta_readout_bundle_axes(
    pump_probe: TAReadoutBundle,
    probe_only: TAReadoutBundle,
    *,
    spec: TASubtractionSpec | None = None,
) -> dict[str, Any]:
    """验证 pump-probe 与 probe-only bundle 的 axis 可直接相减。

    当前策略只接受已对齐的 axis；不做 interpolation / resampling / 截断。
    """

    if not isinstance(pump_probe, TAReadoutBundle):
        raise TypeError("pump_probe must be a TAReadoutBundle instance.")
    if not isinstance(probe_only, TAReadoutBundle):
        raise TypeError("probe_only must be a TAReadoutBundle instance.")
    local_spec = TASubtractionSpec() if spec is None else spec
    if not isinstance(local_spec, TASubtractionSpec):
        raise TypeError("spec must be a TASubtractionSpec instance or None.")

    if pump_probe.absorption.shape != probe_only.absorption.shape:
        raise ValueError(
            "absorption shape mismatch: "
            f"pump_probe={pump_probe.absorption.shape}, probe_only={probe_only.absorption.shape}."
        )
    if pump_probe.energy_eV.shape != probe_only.energy_eV.shape:
        raise ValueError(
            "energy axis shape mismatch: "
            f"pump_probe={pump_probe.energy_eV.shape}, probe_only={probe_only.energy_eV.shape}."
        )
    if not np.allclose(pump_probe.energy_eV, probe_only.energy_eV, rtol=local_spec.rtol, atol=local_spec.atol):
        raise ValueError("energy axis mismatch: pump_probe.energy_eV and probe_only.energy_eV are not allclose.")

    has_pp_omega = pump_probe.omega_fs_inv is not None
    has_po_omega = probe_only.omega_fs_inv is not None
    if local_spec.validate_omega_axis:
        if has_pp_omega != has_po_omega:
            raise ValueError("omega axis mismatch: omega_fs_inv is present on only one bundle.")
        if has_pp_omega and has_po_omega:
            assert pump_probe.omega_fs_inv is not None
            assert probe_only.omega_fs_inv is not None
            if pump_probe.omega_fs_inv.shape != probe_only.omega_fs_inv.shape:
                raise ValueError(
                    "omega axis shape mismatch: "
                    f"pump_probe={pump_probe.omega_fs_inv.shape}, probe_only={probe_only.omega_fs_inv.shape}."
                )
            if not np.allclose(
                pump_probe.omega_fs_inv,
                probe_only.omega_fs_inv,
                rtol=local_spec.rtol,
                atol=local_spec.atol,
            ):
                raise ValueError("omega axis mismatch: pump_probe.omega_fs_inv and probe_only.omega_fs_inv are not allclose.")

    omega = pump_probe.omega_fs_inv
    summary: dict[str, Any] = {
        "n_points": int(pump_probe.energy_eV.size),
        "energy_min_eV": float(np.min(pump_probe.energy_eV)) if pump_probe.energy_eV.size else None,
        "energy_max_eV": float(np.max(pump_probe.energy_eV)) if pump_probe.energy_eV.size else None,
        "has_omega_fs_inv": omega is not None,
    }
    if omega is not None:
        summary["omega_min_fs_inv"] = float(np.min(omega)) if omega.size else None
        summary["omega_max_fs_inv"] = float(np.max(omega)) if omega.size else None
    return summary


def compute_ta_contrast(
    pump_probe: TAReadoutBundle,
    probe_only: TAReadoutBundle,
    *,
    delay_fs: float,
    case_name: str | None = None,
    subtraction: TASubtractionSpec | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> TAContrastResult:
    """计算单 delay TA contrast。

    固定 convention：`delta_absorption = pump_probe.absorption - probe_only.absorption`。
    """

    local_spec = TASubtractionSpec() if subtraction is None else subtraction
    if not isinstance(local_spec, TASubtractionSpec):
        raise TypeError("subtraction must be a TASubtractionSpec instance or None.")
    delay = float(delay_fs)
    if not np.isfinite(delay):
        raise ValueError("delay_fs must be finite.")
    axis_summary = validate_ta_readout_bundle_axes(pump_probe, probe_only, spec=local_spec)
    output_case_name = (
        f"ta_contrast_delay_{_safe_case_value(delay)}_fs"
        if case_name is None
        else validate_pulse_name(case_name)
    )
    contrast_metadata = {
        "convention": "pump_probe_minus_probe_only",
        "pump_probe_case": pump_probe.case_name,
        "probe_only_case": probe_only.case_name,
        "axis_validation": axis_summary,
    }
    contrast_metadata.update(dict(metadata or {}))
    return TAContrastResult(
        case_name=output_case_name,
        delay_fs=delay,
        signal_name=local_spec.signal_name,
        delta_absorption=pump_probe.absorption - probe_only.absorption,
        energy_eV=pump_probe.energy_eV,
        omega_fs_inv=pump_probe.omega_fs_inv,
        pump_probe_bundle=pump_probe,
        probe_only_bundle=probe_only,
        subtraction=local_spec,
        metadata=contrast_metadata,
    )


def build_ta_pump_probe_phase_cycling_plan(
    ta_plan: "TASingleDelayPlan",
    *,
    phase_cycling: TAPhaseCyclingSpec,
    case_name: str | None = None,
) -> PhaseCyclingPlan:
    """为单 delay pump-probe response 构造可选 phase-cycling plan。

    本 helper 只对 pump-probe plan 做 phase cycling；不处理 probe-only
    reference，不做 TA subtraction，也不保存文件。`target_phase_vector`
    来自 `phase_cycling`，必须由用户或上层 recipe 显式给出。
    """

    if not isinstance(ta_plan, TASingleDelayPlan):
        raise TypeError("ta_plan must be a TASingleDelayPlan instance.")
    if not isinstance(phase_cycling, TAPhaseCyclingSpec):
        raise TypeError("phase_cycling must be a TAPhaseCyclingSpec instance.")
    if ta_plan.checkpoint.enabled:
        raise ValueError("TA pump-probe phase-cycling scaffold does not support checkpoint.enabled=True.")

    base_plan = ta_plan.make_pump_probe_plan()
    output_case_name = (
        f"{ta_plan.case_name}_pump_probe_phase_cycling"
        if case_name is None
        else validate_pulse_name(case_name)
    )
    projection = PhaseProjectionSpec(
        quantity=phase_cycling.projection_quantity,
        normalize=phase_cycling.normalize,
        sign=phase_cycling.sign,
        metadata={
            "ta_context": "pump_probe_phase_cycled",
            "signal_name": phase_cycling.signal_name,
        },
    )
    return PhaseCyclingPlan(
        base_plan=base_plan,
        phase_grid=phase_cycling.phase_grid,
        target_phase_vector=phase_cycling.target_phase_vector,
        projection=projection,
        case_name_template=f"{output_case_name}_phase_{{index:04d}}",
        metadata={
            "ta_context": "pump_probe_phase_cycled",
            "ta_case_name": ta_plan.case_name,
            "delay": ta_plan.delay.to_dict(),
            "phase_cycling": phase_cycling.to_dict(),
            "scope": "pump_probe_only_no_ta_subtraction",
        },
    )


def build_ta_phase_cycled_pump_probe_bundle(
    phase_result: PhaseCyclingResult,
    *,
    phase_cycling: TAPhaseCyclingSpec,
    signal_name: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> ProjectedReadoutBundle:
    """把 pump-probe phase-cycling result 打包为 projected readout bundle。

    本 helper 不改变 projected signal 数值，不做 probe-only reference，也不做
    TA subtraction 或 delay scan。
    """

    if not isinstance(phase_result, PhaseCyclingResult):
        raise TypeError("phase_result must be a PhaseCyclingResult instance.")
    if not isinstance(phase_cycling, TAPhaseCyclingSpec):
        raise TypeError("phase_cycling must be a TAPhaseCyclingSpec instance.")
    bundle_metadata = {
        "ta_context": "pump_probe_phase_cycled",
        "target_phase_vector": dict(phase_cycling.target_phase_vector),
        "projection_quantity": phase_cycling.projection_quantity,
        "scope": "pump_probe_only_no_ta_subtraction",
    }
    bundle_metadata.update(dict(metadata or {}))
    return build_projected_readout_bundle(
        phase_result,
        signal_name=phase_cycling.signal_name if signal_name is None else signal_name,
        axis_specs=phase_cycling.axis_specs,
        metadata=bundle_metadata,
    )


@dataclass
class TADelayScanMap:
    """多个单 delay TA contrast 堆叠后的 delay-energy map。

    delay 轴严格保留输入顺序，不排序；energy / omega 轴要求每个 delay 已经
    对齐。当前不做 interpolation、resampling、smoothing 或落盘。
    """

    case_name: str
    delays_fs: np.ndarray
    energy_eV: np.ndarray
    delta_absorption: np.ndarray
    omega_fs_inv: np.ndarray | None = None
    contrast_results: tuple[TAContrastResult, ...] = ()
    metadata: dict[str, Any] = dataclass_field(default_factory=dict)

    def __post_init__(self) -> None:
        self.case_name = validate_pulse_name(self.case_name)
        delays = np.asarray(self.delays_fs, dtype=float)
        energy = np.asarray(self.energy_eV, dtype=float)
        delta = np.asarray(self.delta_absorption)
        if delays.ndim != 1:
            raise ValueError(f"delays_fs must be one-dimensional. Got shape {delays.shape}.")
        if delays.size == 0:
            raise ValueError("delays_fs must not be empty.")
        if not np.all(np.isfinite(delays)):
            raise ValueError("delays_fs must contain only finite values.")
        if energy.ndim != 1:
            raise ValueError(f"energy_eV must be one-dimensional. Got shape {energy.shape}.")
        if not np.all(np.isfinite(energy)):
            raise ValueError("energy_eV must contain only finite values.")
        expected_shape = (delays.size, energy.size)
        if delta.shape != expected_shape:
            raise ValueError(
                f"delta_absorption shape must be delay x energy {expected_shape}. Got {delta.shape}."
            )
        omega = None if self.omega_fs_inv is None else np.asarray(self.omega_fs_inv, dtype=float)
        if omega is not None:
            if omega.shape != energy.shape:
                raise ValueError(
                    f"omega_fs_inv shape must match energy_eV shape. Got {omega.shape} and {energy.shape}."
                )
            if not np.all(np.isfinite(omega)):
                raise ValueError("omega_fs_inv must contain only finite values.")
        contrasts = tuple(self.contrast_results)
        for contrast in contrasts:
            if not isinstance(contrast, TAContrastResult):
                raise TypeError("contrast_results must contain only TAContrastResult instances.")
        if contrasts and len(contrasts) != delays.size:
            raise ValueError(
                f"contrast_results length must match delays_fs length. Got {len(contrasts)} and {delays.size}."
            )
        self.delays_fs = delays
        self.energy_eV = energy
        self.delta_absorption = delta
        self.omega_fs_inv = omega
        self.contrast_results = contrasts
        self.metadata = dict(self.metadata)

    def to_dict(self, *, include_arrays: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "class": self.__class__.__name__,
            "case_name": self.case_name,
            "n_delays": int(self.delays_fs.size),
            "n_energy": int(self.energy_eV.size),
            "delay_range_fs": (
                float(np.min(self.delays_fs)),
                float(np.max(self.delays_fs)),
            ),
            "energy_range_eV": _energy_range_eV(self.energy_eV),
            "has_omega_fs_inv": self.omega_fs_inv is not None,
            "delta_absorption_shape": tuple(self.delta_absorption.shape),
            "delta_absorption_dtype": str(self.delta_absorption.dtype),
            "contrast_cases": [contrast.case_name for contrast in self.contrast_results],
            "metadata": dict(self.metadata),
            "axis_policy": {
                "delay_order": "input_order_preserved",
                "energy_axis": "all delays must match; no interpolation or resampling",
            },
        }
        if include_arrays:
            payload["delays_fs"] = self.delays_fs.tolist()
            payload["energy_eV"] = self.energy_eV.tolist()
            payload["delta_absorption"] = self.delta_absorption.tolist()
            payload["omega_fs_inv"] = None if self.omega_fs_inv is None else self.omega_fs_inv.tolist()
        return payload


def validate_ta_contrast_axes_for_scan(
    contrasts: tuple[TAContrastResult, ...] | list[TAContrastResult],
    *,
    rtol: float = 1.0e-9,
    atol: float = 1.0e-12,
    validate_omega_axis: bool = True,
) -> dict[str, Any]:
    """验证多个单 delay contrast 可直接堆叠成 delay × energy map。

    当前策略只接受完全对齐的 energy axis；不做插值、重采样或排序。
    """

    contrast_tuple = tuple(contrasts)
    if not contrast_tuple:
        raise ValueError("contrasts must not be empty.")
    rtol_value = float(rtol)
    atol_value = float(atol)
    if rtol_value < 0.0:
        raise ValueError("rtol must be >= 0.")
    if atol_value < 0.0:
        raise ValueError("atol must be >= 0.")
    for contrast in contrast_tuple:
        if not isinstance(contrast, TAContrastResult):
            raise TypeError("contrasts must contain only TAContrastResult instances.")

    reference = contrast_tuple[0]
    for index, contrast in enumerate(contrast_tuple[1:], start=1):
        if contrast.delta_absorption.shape != reference.delta_absorption.shape:
            raise ValueError(
                "delta_absorption shape mismatch at delay index "
                f"{index}: reference={reference.delta_absorption.shape}, current={contrast.delta_absorption.shape}."
            )
        if contrast.energy_eV.shape != reference.energy_eV.shape:
            raise ValueError(
                "energy axis shape mismatch at delay index "
                f"{index}: reference={reference.energy_eV.shape}, current={contrast.energy_eV.shape}."
            )
        if not np.allclose(contrast.energy_eV, reference.energy_eV, rtol=rtol_value, atol=atol_value):
            raise ValueError(f"energy axis mismatch at delay index {index}.")

        if validate_omega_axis:
            has_reference_omega = reference.omega_fs_inv is not None
            has_current_omega = contrast.omega_fs_inv is not None
            if has_reference_omega != has_current_omega:
                raise ValueError(f"omega axis mismatch at delay index {index}: omega_fs_inv is missing on one side.")
            if has_reference_omega and has_current_omega:
                assert reference.omega_fs_inv is not None
                assert contrast.omega_fs_inv is not None
                if contrast.omega_fs_inv.shape != reference.omega_fs_inv.shape:
                    raise ValueError(
                        "omega axis shape mismatch at delay index "
                        f"{index}: reference={reference.omega_fs_inv.shape}, current={contrast.omega_fs_inv.shape}."
                    )
                if not np.allclose(
                    contrast.omega_fs_inv,
                    reference.omega_fs_inv,
                    rtol=rtol_value,
                    atol=atol_value,
                ):
                    raise ValueError(f"omega axis mismatch at delay index {index}.")

    omega = reference.omega_fs_inv
    summary: dict[str, Any] = {
        "n_delays": int(len(contrast_tuple)),
        "n_energy": int(reference.energy_eV.size),
        "energy_min_eV": float(np.min(reference.energy_eV)) if reference.energy_eV.size else None,
        "energy_max_eV": float(np.max(reference.energy_eV)) if reference.energy_eV.size else None,
        "has_omega_fs_inv": omega is not None,
        "validate_omega_axis": bool(validate_omega_axis),
        "rtol": rtol_value,
        "atol": atol_value,
        "axis_policy": "input delay order preserved; energy axes must match; no interpolation or resampling",
    }
    if omega is not None:
        summary["omega_min_fs_inv"] = float(np.min(omega)) if omega.size else None
        summary["omega_max_fs_inv"] = float(np.max(omega)) if omega.size else None
    return summary


def build_ta_delay_scan_map(
    contrasts: tuple[TAContrastResult, ...] | list[TAContrastResult],
    *,
    case_name: str = "ta_delay_scan",
    rtol: float = 1.0e-9,
    atol: float = 1.0e-12,
    validate_omega_axis: bool = True,
    metadata: Mapping[str, Any] | None = None,
) -> TADelayScanMap:
    """把多个单 delay contrast 堆叠为 delay × energy map。

    delay 轴按 `contrasts` 的输入顺序生成，不按数值排序。
    """

    contrast_tuple = tuple(contrasts)
    axis_summary = validate_ta_contrast_axes_for_scan(
        contrast_tuple,
        rtol=rtol,
        atol=atol,
        validate_omega_axis=validate_omega_axis,
    )
    reference = contrast_tuple[0]
    map_metadata = {
        "axis_validation": axis_summary,
        "source_contrast_cases": [contrast.case_name for contrast in contrast_tuple],
        "delay_axis_policy": "input_order_preserved",
        "energy_axis_policy": "all delays must match; no interpolation or resampling",
        "convention": "pump_probe_minus_probe_only",
    }
    map_metadata.update(dict(metadata or {}))
    return TADelayScanMap(
        case_name=case_name,
        delays_fs=np.asarray([contrast.delay_fs for contrast in contrast_tuple], dtype=float),
        energy_eV=reference.energy_eV,
        omega_fs_inv=reference.omega_fs_inv,
        delta_absorption=np.stack([contrast.delta_absorption for contrast in contrast_tuple], axis=0),
        contrast_results=contrast_tuple,
        metadata=map_metadata,
    )


def extract_ta_absorption_bundle(
    result: SingleRunResult,
    *,
    case_name: str | None = None,
) -> TAReadoutBundle:
    """从 single-run absorption-like readout 中提取 TA recipe bundle。

    本函数不做 subtraction、不做 phase projection，也不假设 delay scan。
    """

    if not isinstance(result, SingleRunResult):
        raise TypeError("result must be a SingleRunResult instance.")
    if result.readout is None:
        raise ValueError("SingleRunResult.readout is required for TA absorption bundle.")
    spectrum = result.readout.spectrum
    if spectrum is None:
        raise ValueError("SingleRunResult.readout.spectrum is required for TA absorption bundle.")
    available = sorted(str(key) for key in spectrum)
    if "absorption" not in spectrum:
        raise KeyError(f"spectrum is missing 'absorption'. Available keys: {available}")
    if "energy_eV" not in spectrum:
        raise KeyError(f"spectrum is missing 'energy_eV'. Available keys: {available}")
    return TAReadoutBundle(
        case_name=result.case_name if case_name is None else case_name,
        absorption=np.asarray(spectrum["absorption"]),
        energy_eV=np.asarray(spectrum["energy_eV"], dtype=float),
        omega_fs_inv=None if "omega_fs_inv" not in spectrum else np.asarray(spectrum["omega_fs_inv"], dtype=float),
        metadata={
            "source_case_name": result.case_name,
            "readout_mode": result.readout.mode,
            "available_spectrum_keys": available,
        },
    )


@dataclass
class TASingleDelayPlan:
    """TA recipe v2 minimal single-delay plan。

    物理语义：

    - physical pulses: pump, probe；
    - readout: probe-channel absorption-like readout；
    - `delay_fs = probe_center_fs - pump_center_fs`；
    - 正 delay 表示 pump before probe；
    - reference cases: pump-probe response 与 probe-only response。

    `execute_pair()` 只执行并提取两条 readout bundle；TA subtraction 需要
    显式调用 `TASingleDelayPairResult.compute_contrast()`。
    """

    base_params: NLevelPhysicalParams
    pump: PulseSpec
    probe: PulseSpec
    delay: TADelayCenters
    normalizer: ParaNormalizer = dataclass_field(default_factory=ParaNormalizer)
    readout: ReadoutSpec | None = None
    checkpoint: SingleRunCheckpointSettings = dataclass_field(default_factory=SingleRunCheckpointSettings)
    case_name: str = "ta_single_delay"
    metadata: dict[str, Any] = dataclass_field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.base_params, NLevelPhysicalParams):
            raise TypeError("base_params must be a NLevelPhysicalParams instance.")
        if not isinstance(self.pump, PulseSpec):
            raise TypeError("pump must be a PulseSpec instance.")
        if not isinstance(self.probe, PulseSpec):
            raise TypeError("probe must be a PulseSpec instance.")
        if self.pump.name == self.probe.name:
            raise ValueError("pump.name and probe.name must be distinct.")
        if not isinstance(self.delay, TADelayCenters):
            raise TypeError("delay must be a TADelayCenters instance.")
        if not isinstance(self.normalizer, ParaNormalizer):
            raise TypeError("normalizer must be a ParaNormalizer instance.")
        if not isinstance(self.checkpoint, SingleRunCheckpointSettings):
            raise TypeError("checkpoint must be a SingleRunCheckpointSettings instance.")
        if self.checkpoint.enabled:
            raise ValueError("TASingleDelayPlan v2 minimal scaffold does not support checkpoint.enabled=True.")
        readout = ReadoutSpec(mode="absorption", readout_field_name=self.probe.name) if self.readout is None else self.readout
        if not isinstance(readout, ReadoutSpec):
            raise TypeError("readout must be a ReadoutSpec instance or None.")
        if readout.mode != "absorption":
            raise ValueError("TA recipe v2 minimal readout must use mode='absorption'.")
        if readout.readout_field_name != self.probe.name:
            raise ValueError("TA recipe v2 minimal readout_field_name must match probe.name.")
        self.readout = readout
        self.case_name = validate_pulse_name(self.case_name)
        self.metadata = dict(self.metadata)

    def build_pump_probe_sequence(self) -> PulseSequenceSpec:
        return PulseSequenceSpec(
            name=f"{self.case_name}_pump_probe_sequence",
            pulses=(self.pump, self.probe),
            metadata={
                "recipe": "ta_recipe_v2_minimal",
                "case_role": "pump_probe",
                "delay": self.delay.to_dict(),
            },
        )

    def build_probe_only_sequence(self) -> PulseSequenceSpec:
        return PulseSequenceSpec(
            name=f"{self.case_name}_probe_only_sequence",
            pulses=(self.probe,),
            metadata={
                "recipe": "ta_recipe_v2_minimal",
                "case_role": "probe_only",
                "delay": self.delay.to_dict(),
            },
        )

    def _make_plan(
        self,
        *,
        sequence: PulseSequenceSpec,
        centers_fs: dict[str, float],
        case_role: str,
    ) -> SingleRunPlan:
        case_name = f"{self.case_name}_{case_role}"
        field_plan = SingleRunFieldPlan(
            sequence=sequence,
            centers_fs=centers_fs,
            phase_vector={},
            case_name=case_name,
            metadata={
                "recipe": "ta_recipe_v2_minimal",
                "case_role": case_role,
                "delay": self.delay.to_dict(),
            },
        )
        return SingleRunPlan(
            base_params=self.base_params,
            field_plan=field_plan,
            normalizer=self.normalizer,
            readout=self.readout,
            checkpoint=self.checkpoint,
            case_name=case_name,
            input_metadata={
                "ta_recipe_v2": {
                    "case_name": self.case_name,
                    "case_role": case_role,
                    "delay": self.delay.to_dict(),
                    "readout_semantics": "probe-channel absorption-like readout; no TA subtraction",
                    **dict(self.metadata),
                }
            },
        )

    def make_pump_probe_plan(self) -> SingleRunPlan:
        return self._make_plan(
            sequence=self.build_pump_probe_sequence(),
            centers_fs={
                self.pump.name: self.delay.pump_center_fs,
                self.probe.name: self.delay.probe_center_fs,
            },
            case_role="pump_probe",
        )

    def make_probe_only_plan(self) -> SingleRunPlan:
        return self._make_plan(
            sequence=self.build_probe_only_sequence(),
            centers_fs={self.probe.name: self.delay.probe_center_fs},
            case_role="probe_only",
        )

    def make_pump_probe_phase_cycling_plan(
        self,
        *,
        phase_cycling: TAPhaseCyclingSpec,
        case_name: str | None = None,
    ) -> PhaseCyclingPlan:
        """构造 pump-probe response 的可选 phase-cycling plan。

        本方法只构造 plan，不执行 solver，不处理 probe-only reference，也不做
        TA subtraction。
        """

        return build_ta_pump_probe_phase_cycling_plan(
            self,
            phase_cycling=phase_cycling,
            case_name=case_name,
        )

    def execute_pump_probe(self) -> SingleRunResult:
        return self.make_pump_probe_plan().execute()

    def execute_probe_only(self) -> SingleRunResult:
        return self.make_probe_only_plan().execute()

    def execute_pair(self) -> "TASingleDelayPairResult":
        pump_probe = self.execute_pump_probe()
        probe_only = self.execute_probe_only()
        return TASingleDelayPairResult(
            case_name=self.case_name,
            delay_fs=self.delay.delay_fs,
            pump_probe=pump_probe,
            probe_only=probe_only,
            pump_probe_bundle=extract_ta_absorption_bundle(pump_probe),
            probe_only_bundle=extract_ta_absorption_bundle(probe_only),
            metadata={"ta_recipe_v2": self.to_dict()},
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "class": self.__class__.__name__,
            "case_name": self.case_name,
            "delay": self.delay.to_dict(),
            "pump": self.pump.to_dict(),
            "probe": self.probe.to_dict(),
            "readout": self.readout.to_dict(),
            "checkpoint": self.checkpoint.to_dict(),
            "metadata": dict(self.metadata),
            "ta_semantics": {
                "physical_pulses": [self.pump.name, self.probe.name],
                "readout_field_name": self.readout.readout_field_name,
                "future_signal": "S_TA(omega, delay) = S_pump_probe(omega, delay) - S_probe_only(omega)",
                "current_scope": "single delay plans and absorption-like bundles only; no subtraction",
            },
        }


@dataclass
class TASingleDelayPairResult:
    """单个 delay 的 pump-probe / probe-only 执行结果容器。"""

    case_name: str
    delay_fs: float
    pump_probe: SingleRunResult
    probe_only: SingleRunResult
    pump_probe_bundle: TAReadoutBundle | None = None
    probe_only_bundle: TAReadoutBundle | None = None
    metadata: dict[str, Any] = dataclass_field(default_factory=dict)

    def __post_init__(self) -> None:
        self.case_name = validate_pulse_name(self.case_name)
        self.delay_fs = float(self.delay_fs)
        if not np.isfinite(self.delay_fs):
            raise ValueError("delay_fs must be finite.")
        if not isinstance(self.pump_probe, SingleRunResult):
            raise TypeError("pump_probe must be a SingleRunResult instance.")
        if not isinstance(self.probe_only, SingleRunResult):
            raise TypeError("probe_only must be a SingleRunResult instance.")
        if self.pump_probe_bundle is not None and not isinstance(self.pump_probe_bundle, TAReadoutBundle):
            raise TypeError("pump_probe_bundle must be a TAReadoutBundle instance or None.")
        if self.probe_only_bundle is not None and not isinstance(self.probe_only_bundle, TAReadoutBundle):
            raise TypeError("probe_only_bundle must be a TAReadoutBundle instance or None.")
        self.metadata = dict(self.metadata)

    def compute_contrast(
        self,
        *,
        subtraction: TASubtractionSpec | None = None,
        case_name: str | None = None,
    ) -> TAContrastResult:
        """基于已有 bundles 计算单 delay TA contrast。

        本方法不重新计算 bundles、不执行 solver、不保存输出文件。
        """

        if self.pump_probe_bundle is None:
            raise ValueError("compute_contrast requires pump_probe_bundle.")
        if self.probe_only_bundle is None:
            raise ValueError("compute_contrast requires probe_only_bundle.")
        return compute_ta_contrast(
            self.pump_probe_bundle,
            self.probe_only_bundle,
            delay_fs=self.delay_fs,
            case_name=case_name,
            subtraction=subtraction,
            metadata={"pair_case_name": self.case_name},
        )

    def to_dict(self, *, include_arrays: bool = False) -> dict[str, Any]:
        return {
            "class": self.__class__.__name__,
            "case_name": self.case_name,
            "delay_fs": float(self.delay_fs),
            "pump_probe": _single_run_summary(self.pump_probe),
            "probe_only": _single_run_summary(self.probe_only),
            "pump_probe_bundle": None
            if self.pump_probe_bundle is None
            else self.pump_probe_bundle.to_dict(include_arrays=include_arrays),
            "probe_only_bundle": None
            if self.probe_only_bundle is None
            else self.probe_only_bundle.to_dict(include_arrays=include_arrays),
            "metadata": dict(self.metadata),
        }


@dataclass
class TADelayScanPlan:
    """最小 TA delay scan plan。

    本类只把多个 `TASingleDelayPlan` 串起来，并把每个 delay 的 contrast
    堆叠成 delay × energy map。delay 输入顺序会被保留；当前不做排序、
    插值、重采样、绘图、保存或 phase cycling。
    """

    base_params: NLevelPhysicalParams
    pump: PulseSpec
    probe: PulseSpec
    delays_fs: tuple[float, ...] | list[float] | np.ndarray
    probe_center_fs: float = 0.0
    normalizer: ParaNormalizer = dataclass_field(default_factory=ParaNormalizer)
    readout: ReadoutSpec | None = None
    subtraction: TASubtractionSpec = dataclass_field(default_factory=TASubtractionSpec)
    checkpoint: SingleRunCheckpointSettings = dataclass_field(default_factory=SingleRunCheckpointSettings)
    case_name: str = "ta_delay_scan"
    metadata: dict[str, Any] = dataclass_field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.base_params, NLevelPhysicalParams):
            raise TypeError("base_params must be a NLevelPhysicalParams instance.")
        if not isinstance(self.pump, PulseSpec):
            raise TypeError("pump must be a PulseSpec instance.")
        if not isinstance(self.probe, PulseSpec):
            raise TypeError("probe must be a PulseSpec instance.")
        if self.pump.name == self.probe.name:
            raise ValueError("pump.name and probe.name must be distinct.")
        delays = tuple(float(delay) for delay in self.delays_fs)
        if not delays:
            raise ValueError("delays_fs must not be empty.")
        if not all(np.isfinite(delay) for delay in delays):
            raise ValueError("delays_fs must contain only finite values.")
        probe_center = float(self.probe_center_fs)
        if not np.isfinite(probe_center):
            raise ValueError("probe_center_fs must be finite.")
        if not isinstance(self.normalizer, ParaNormalizer):
            raise TypeError("normalizer must be a ParaNormalizer instance.")
        if self.readout is not None and not isinstance(self.readout, ReadoutSpec):
            raise TypeError("readout must be a ReadoutSpec instance or None.")
        if not isinstance(self.subtraction, TASubtractionSpec):
            raise TypeError("subtraction must be a TASubtractionSpec instance.")
        if not isinstance(self.checkpoint, SingleRunCheckpointSettings):
            raise TypeError("checkpoint must be a SingleRunCheckpointSettings instance.")
        if self.checkpoint.enabled:
            raise ValueError("TADelayScanPlan v2 minimal scaffold does not support checkpoint.enabled=True.")
        self.delays_fs = delays
        self.probe_center_fs = probe_center
        self.case_name = validate_pulse_name(self.case_name)
        self.metadata = dict(self.metadata)

    def make_single_delay_plan(self, delay_fs: float, *, index: int) -> TASingleDelayPlan:
        """为 delay scan 中的一个 delay 生成单 delay plan。"""

        delay = float(delay_fs)
        if not np.isfinite(delay):
            raise ValueError("delay_fs must be finite.")
        delay_index = int(index)
        if delay_index < 0:
            raise ValueError("index must be >= 0.")
        single_case_name = f"{self.case_name}_i{delay_index:03d}_delay_{_safe_case_value(delay)}_fs"
        return TASingleDelayPlan(
            base_params=self.base_params,
            pump=self.pump,
            probe=self.probe,
            delay=TADelayCenters(delay_fs=delay, probe_center_fs=self.probe_center_fs),
            normalizer=self.normalizer,
            readout=self.readout,
            checkpoint=self.checkpoint,
            case_name=single_case_name,
            metadata={
                "scan_case_name": self.case_name,
                "scan_index": delay_index,
                "scan_delay_order": "input_order_preserved",
                **dict(self.metadata),
            },
        )

    def make_single_delay_plans(self) -> tuple[TASingleDelayPlan, ...]:
        return tuple(
            self.make_single_delay_plan(delay_fs, index=index)
            for index, delay_fs in enumerate(self.delays_fs)
        )

    def execute(self, *, executor: Any | None = None) -> "TADelayScanResult":
        """执行 delay scan。

        `executor` 是测试和上层编排入口，输入 `TASingleDelayPlan`，必须返回
        `TASingleDelayPairResult`。当 executor 为 None 时才调用真实
        `TASingleDelayPlan.execute_pair()`。
        """

        single_delay_plans = self.make_single_delay_plans()
        pair_results: list[TASingleDelayPairResult] = []
        contrast_results: list[TAContrastResult] = []
        for index, single_plan in enumerate(single_delay_plans):
            pair_result = single_plan.execute_pair() if executor is None else executor(single_plan)
            if not isinstance(pair_result, TASingleDelayPairResult):
                raise TypeError("executor must return a TASingleDelayPairResult instance.")
            contrast = pair_result.compute_contrast(
                subtraction=self.subtraction,
                case_name=f"{single_plan.case_name}_contrast",
            )
            contrast.metadata.update(
                {
                    "scan_case_name": self.case_name,
                    "scan_index": index,
                    "scan_delay_order": "input_order_preserved",
                }
            )
            pair_results.append(pair_result)
            contrast_results.append(contrast)

        scan_map = build_ta_delay_scan_map(
            tuple(contrast_results),
            case_name=f"{self.case_name}_map",
            rtol=self.subtraction.rtol,
            atol=self.subtraction.atol,
            validate_omega_axis=self.subtraction.validate_omega_axis,
            metadata={
                "scan_case_name": self.case_name,
                "scan_plan": self.to_dict(),
            },
        )
        return TADelayScanResult(
            case_name=self.case_name,
            scan_plan=self,
            single_delay_plans=single_delay_plans,
            pair_results=tuple(pair_results),
            contrast_results=tuple(contrast_results),
            scan_map=scan_map,
            metadata={"delay_order": "input_order_preserved"},
        )

    def to_dict(self) -> dict[str, Any]:
        readout = ReadoutSpec(mode="absorption", readout_field_name=self.probe.name) if self.readout is None else self.readout
        return {
            "class": self.__class__.__name__,
            "case_name": self.case_name,
            "n_delays": int(len(self.delays_fs)),
            "delays_fs": [float(delay) for delay in self.delays_fs],
            "probe_center_fs": float(self.probe_center_fs),
            "pump": self.pump.to_dict(),
            "probe": self.probe.to_dict(),
            "readout": readout.to_dict(),
            "subtraction": self.subtraction.to_dict(),
            "checkpoint": self.checkpoint.to_dict(),
            "metadata": dict(self.metadata),
            "ta_semantics": {
                "delay_order": "input_order_preserved",
                "scan_signal": "S_TA(omega, delay) = S_pump_probe(omega, delay) - S_probe_only(omega)",
                "axis_policy": "all delay energy axes must match; no interpolation or resampling",
            },
        }


@dataclass
class TADelayScanResult:
    """最小 TA delay scan 执行结果容器。"""

    case_name: str
    scan_plan: TADelayScanPlan
    single_delay_plans: tuple[TASingleDelayPlan, ...]
    pair_results: tuple[TASingleDelayPairResult, ...]
    contrast_results: tuple[TAContrastResult, ...]
    scan_map: TADelayScanMap
    metadata: dict[str, Any] = dataclass_field(default_factory=dict)

    def __post_init__(self) -> None:
        self.case_name = validate_pulse_name(self.case_name)
        if not isinstance(self.scan_plan, TADelayScanPlan):
            raise TypeError("scan_plan must be a TADelayScanPlan instance.")
        single_plans = tuple(self.single_delay_plans)
        pair_results = tuple(self.pair_results)
        contrast_results = tuple(self.contrast_results)
        if not isinstance(self.scan_map, TADelayScanMap):
            raise TypeError("scan_map must be a TADelayScanMap instance.")
        n_delays = len(single_plans)
        if n_delays == 0:
            raise ValueError("single_delay_plans must not be empty.")
        if len(pair_results) != n_delays:
            raise ValueError("pair_results length must match single_delay_plans length.")
        if len(contrast_results) != n_delays:
            raise ValueError("contrast_results length must match single_delay_plans length.")
        if self.scan_map.delays_fs.shape != (n_delays,):
            raise ValueError("scan_map delay axis length must match single_delay_plans length.")
        for item in single_plans:
            if not isinstance(item, TASingleDelayPlan):
                raise TypeError("single_delay_plans must contain only TASingleDelayPlan instances.")
        for item in pair_results:
            if not isinstance(item, TASingleDelayPairResult):
                raise TypeError("pair_results must contain only TASingleDelayPairResult instances.")
        for index, item in enumerate(contrast_results):
            if not isinstance(item, TAContrastResult):
                raise TypeError("contrast_results must contain only TAContrastResult instances.")
            if not np.isclose(item.delay_fs, self.scan_map.delays_fs[index], rtol=0.0, atol=0.0):
                raise ValueError(f"contrast_results delay mismatch at index {index}.")
        self.single_delay_plans = single_plans
        self.pair_results = pair_results
        self.contrast_results = contrast_results
        self.metadata = dict(self.metadata)

    def to_dict(self, *, include_arrays: bool = False) -> dict[str, Any]:
        return {
            "class": self.__class__.__name__,
            "case_name": self.case_name,
            "scan_plan": self.scan_plan.to_dict(),
            "scan_map": self.scan_map.to_dict(include_arrays=include_arrays),
            "single_delay_cases": [plan.case_name for plan in self.single_delay_plans],
            "pair_cases": [pair.case_name for pair in self.pair_results],
            "contrast_cases": [contrast.case_name for contrast in self.contrast_results],
            "metadata": dict(self.metadata),
        }


__all__ = [
    "TADelayCenters",
    "TAReadoutBundle",
    "TASubtractionSpec",
    "TAContrastResult",
    "TAPhaseCyclingSpec",
    "TAPhaseCycledPumpProbeResult",
    "TADelayScanMap",
    "TASingleDelayPlan",
    "TASingleDelayPairResult",
    "TADelayScanPlan",
    "TADelayScanResult",
    "extract_ta_absorption_bundle",
    "validate_ta_readout_bundle_axes",
    "compute_ta_contrast",
    "build_ta_pump_probe_phase_cycling_plan",
    "build_ta_phase_cycled_pump_probe_bundle",
    "validate_ta_contrast_axes_for_scan",
    "build_ta_delay_scan_map",
]
