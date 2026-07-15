"""Minimal TA recipe v2 scaffold built on generic pulse-sequence layers.

本模块只表达单个 delay 的最小 TA 编排：

    pump/probe physical pulses
    -> pump-probe SingleRunPlan
    -> probe-only SingleRunPlan
    -> probe-channel absorption-like readout bundle

TA subtraction 只由显式 `compute_ta_contrast(...)` 执行，固定 convention 为
`S_TA = S_pump_probe - S_probe_only`。当前不实现 delay scan、phase-cycling
TA、TAResultIO v2 或旧 demo 迁移。readout 不是第三个激发脉冲；probe
既是 physical probe pulse，也是 `ReadoutSpec(readout_field_name=probe.name)`
的 reference field。
"""

from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
from collections.abc import Mapping
from typing import Any

import numpy as np

from qudpy_sjh.experiments.pulse_sequence import (
    PulseSequenceSpec,
    PulseSpec,
    ReadoutSpec,
    SingleRunCheckpointSettings,
    SingleRunFieldPlan,
    SingleRunPlan,
    SingleRunResult,
)
from qudpy_sjh.experiments.pulse_sequence.pulse_sequence import validate_pulse_name
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


__all__ = [
    "TADelayCenters",
    "TAReadoutBundle",
    "TASubtractionSpec",
    "TAContrastResult",
    "TASingleDelayPlan",
    "TASingleDelayPairResult",
    "extract_ta_absorption_bundle",
    "validate_ta_readout_bundle_axes",
    "compute_ta_contrast",
]
