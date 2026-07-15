"""TA workflow settings.

这是 experimental TA recipe v1 prototype，包含 pump/probe、probe-only
reference 和 TA map 等 TA-specific 语义。它不是 generic pulse-sequence
simulation framework，当前 phase-cycling validation demo 也没有使用它。
本 prototype 不默认执行 phase cycling；phase cycling 应由上层 wrapper /
generic cycler 负责。未来 TA recipe v2 可以调用 generic pulse-sequence /
phase-cycling 基础层，但本轮不迁移。

Legacy / deprecation policy:

- 这是 legacy TA recipe v1 prototype，不是当前 TA recipe v2 主线；
- 当前保留为 historical reference、IO/export behavior reference、
  migration comparison 和 regression reference；
- 新开发应优先使用 ``qudpy_sjh.experiments.pulse_sequence`` 以及
  ``qudpy_sjh.experiments.ta.ta_recipe_v2``；
- 本文件不在运行时发 ``DeprecationWarning``，避免污染测试输出；
- 本轮不删除、不移动、不重构旧文件。

``TASettings`` is the pure calculation definition for an intrinsic transient
absorption workflow.  It intentionally does not contain output directories,
checkpoint paths, preview plotting flags, or core result export policies.

Expected high-level flow:

    TASettings -> TAPlan.execute() -> TAResult -> TAResultIO / analysis

``base_params`` is the canonical physical template.  Its ``field`` must expose
``field["pump"]`` and ``field["probe"]``; those are treated as pump/probe
templates.  Their reference centers are declared by ``TATemplateSettings`` and
are not inferred from the waveform.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
from typing import Any, Literal, Sequence

import numpy as np

from qudpy_sjh.utils.core import NLevelPhysicalParams
from qudpy_sjh.utils.fields import FieldPhyRoot


TA_EXPERIMENT_NAME = "ta_intrinsic_response"


def _float_tuple_1d(name: str, values: Sequence[float]) -> tuple[float, ...]:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a 1D sequence or 1D numpy.ndarray.")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty.")
    return tuple(float(item) for item in array)


@dataclass(frozen=True)
class TATemplateSettings:
    """Declared reference centers of pump/probe field templates."""

    pump_template_center_fs: float = 0.0
    probe_template_center_fs: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "pump_template_center_fs", float(self.pump_template_center_fs))
        object.__setattr__(self, "probe_template_center_fs", float(self.probe_template_center_fs))

    def to_dict(self) -> dict[str, float]:
        return {
            "pump_template_center_fs": float(self.pump_template_center_fs),
            "probe_template_center_fs": float(self.probe_template_center_fs),
        }


@dataclass(frozen=True)
class TAAbsorptionSettings:
    """Settings for ``rho(t) -> P(t) -> absorption response``."""

    number_density_m3: float = 1.0e24
    window: str | None = "hann"
    subtract_mean: bool = True
    rel_threshold: float = 1.0e-6
    zero_padding_factor: int = 4
    return_intermediates: bool = True

    def __post_init__(self) -> None:
        if self.number_density_m3 <= 0:
            raise ValueError("number_density_m3 must be positive.")
        if self.window not in (None, "none", "hann"):
            raise ValueError("window must be None, 'none', or 'hann'.")
        if self.rel_threshold <= 0:
            raise ValueError("rel_threshold must be positive.")
        if self.zero_padding_factor < 1:
            raise ValueError("zero_padding_factor must be >= 1.")
        object.__setattr__(self, "number_density_m3", float(self.number_density_m3))
        object.__setattr__(self, "subtract_mean", bool(self.subtract_mean))
        object.__setattr__(self, "rel_threshold", float(self.rel_threshold))
        object.__setattr__(self, "zero_padding_factor", int(self.zero_padding_factor))
        object.__setattr__(self, "return_intermediates", bool(self.return_intermediates))

    def to_dict(self) -> dict[str, Any]:
        return {
            "number_density_m3": float(self.number_density_m3),
            "window": self.window,
            "subtract_mean": bool(self.subtract_mean),
            "rel_threshold": float(self.rel_threshold),
            "zero_padding_factor": int(self.zero_padding_factor),
            "return_intermediates": bool(self.return_intermediates),
        }


@dataclass(frozen=True)
class TAStandardizeSettings:
    """Settings for standardizing spectra into one delay-energy TA map."""

    allow_energy_axis_interpolation: bool = True
    common_axis_policy: Literal["overlap", "first"] = "overlap"
    min_common_energy_points: int = 2
    kinetic_energy_eV: float | None = None

    def __post_init__(self) -> None:
        if self.common_axis_policy not in ("overlap", "first"):
            raise ValueError("common_axis_policy must be 'overlap' or 'first'.")
        if self.min_common_energy_points < 1:
            raise ValueError("min_common_energy_points must be >= 1.")
        object.__setattr__(self, "allow_energy_axis_interpolation", bool(self.allow_energy_axis_interpolation))
        object.__setattr__(self, "min_common_energy_points", int(self.min_common_energy_points))
        if self.kinetic_energy_eV is not None:
            object.__setattr__(self, "kinetic_energy_eV", float(self.kinetic_energy_eV))

    @property
    def response_definition(self) -> str:
        return (
            "S_TA(omega, delay) = omega*Im[P_pump_probe(omega, delay)/E_probe(omega)] "
            "- omega*Im[P_probe_only(omega)/E_probe(omega)]"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "allow_energy_axis_interpolation": bool(self.allow_energy_axis_interpolation),
            "common_axis_policy": self.common_axis_policy,
            "min_common_energy_points": int(self.min_common_energy_points),
            "kinetic_energy_eV": self.kinetic_energy_eV,
            "response_definition": self.response_definition,
        }


@dataclass(frozen=True)
class TASettings:
    """Calculation settings for a full-window intrinsic TA delay scan."""

    base_params: NLevelPhysicalParams
    probe_delays_fs: Sequence[float]
    probe_center_fs: float = 0.0
    experiment_name: str = TA_EXPERIMENT_NAME

    template: TATemplateSettings = dataclass_field(default_factory=TATemplateSettings)
    absorption: TAAbsorptionSettings = dataclass_field(default_factory=TAAbsorptionSettings)
    standardize: TAStandardizeSettings = dataclass_field(default_factory=TAStandardizeSettings)

    metadata: dict[str, Any] = dataclass_field(default_factory=dict)
    max_time_points: int | None = 80000

    def __post_init__(self) -> None:
        if not isinstance(self.base_params, NLevelPhysicalParams):
            raise TypeError("base_params must be a NLevelPhysicalParams instance.")
        if not self.experiment_name:
            raise ValueError("experiment_name must be non-empty.")

        object.__setattr__(self, "probe_delays_fs", _float_tuple_1d("probe_delays_fs", self.probe_delays_fs))
        object.__setattr__(self, "probe_center_fs", float(self.probe_center_fs))
        object.__setattr__(self, "metadata", dict(self.metadata))

        if not isinstance(self.template, TATemplateSettings):
            raise TypeError("template must be a TATemplateSettings instance.")
        if not isinstance(self.absorption, TAAbsorptionSettings):
            raise TypeError("absorption must be a TAAbsorptionSettings instance.")
        if not isinstance(self.standardize, TAStandardizeSettings):
            raise TypeError("standardize must be a TAStandardizeSettings instance.")

        field_template = getattr(self.base_params, "field", None)
        if field_template is None:
            raise ValueError("base_params.field is required and must contain pump/probe templates.")

        try:
            pump = field_template["pump"]
            probe = field_template["probe"]
        except Exception as exc:
            raise TypeError(
                "base_params.field must support field['pump'] and field['probe']; "
                "use FieldPhySeries or TAField with sub_field_names=('pump', 'probe')."
            ) from exc

        if not isinstance(pump, FieldPhyRoot):
            raise TypeError("base_params.field['pump'] must be a FieldPhyRoot instance.")
        if not isinstance(probe, FieldPhyRoot):
            raise TypeError("base_params.field['probe'] must be a FieldPhyRoot instance.")

        if getattr(self.base_params, "solver_mode", "lab_exact") != "lab_exact":
            raise ValueError("TA workflow currently requires base_params.solver_mode == 'lab_exact'.")

        dt_fs = float(self.base_params.dt_fs)
        t_start_fs = float(self.base_params.t_start_fs)
        t_end_fs = float(self.base_params.t_end_fs)
        if dt_fs <= 0:
            raise ValueError("base_params.dt_fs must be positive.")
        if t_end_fs <= t_start_fs:
            raise ValueError("base_params.t_end_fs must be larger than t_start_fs.")

        if self.max_time_points is not None:
            max_points = int(self.max_time_points)
            if max_points < 2:
                raise ValueError("max_time_points must be >= 2 or None.")
            n_points = int(np.floor((t_end_fs - t_start_fs) / dt_fs)) + 1
            if n_points > max_points:
                raise ValueError(
                    "base_params time grid is too large for this TASettings guard: "
                    f"n_points={n_points}, max_time_points={max_points}."
                )
            object.__setattr__(self, "max_time_points", max_points)

    @property
    def delays_fs(self) -> tuple[float, ...]:
        return tuple(self.probe_delays_fs)

    @property
    def field_template(self):
        return self.base_params.field

    @property
    def pump_template(self) -> FieldPhyRoot:
        return self.base_params.field["pump"]

    @property
    def probe_template(self) -> FieldPhyRoot:
        return self.base_params.field["probe"]

    def pump_center_fs_for_delay(self, delay_fs: float) -> float:
        return float(self.probe_center_fs) - float(delay_fs)

    def probe_shift_fs(self) -> float:
        return float(self.probe_center_fs) - float(self.template.probe_template_center_fs)

    def pump_shift_fs_for_delay(self, delay_fs: float) -> float:
        return self.pump_center_fs_for_delay(delay_fs) - float(self.template.pump_template_center_fs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_name": self.experiment_name,
            "probe_delays_fs": [float(item) for item in self.probe_delays_fs],
            "probe_center_fs": float(self.probe_center_fs),
            "template": self.template.to_dict(),
            "absorption": self.absorption.to_dict(),
            "standardize": self.standardize.to_dict(),
            "max_time_points": self.max_time_points,
            "metadata": dict(self.metadata),
            "delay_convention": {
                "probe_center_rule": "probe center is fixed at probe_center_fs",
                "pump_center_rule": "pump_center_fs = probe_center_fs - delay_fs",
                "positive_delay": "delay_fs > 0 means pump arrives before probe",
            },
        }


__all__ = [
    "TA_EXPERIMENT_NAME",
    "TATemplateSettings",
    "TAAbsorptionSettings",
    "TAStandardizeSettings",
    "TASettings",
]
