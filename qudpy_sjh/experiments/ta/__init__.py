"""Transient absorption workflow helpers.

package 顶层导出保持 legacy TA v1 prototype 的裸名稳定；TA recipe v2 的
scan 类通过显式 ``*V2`` alias 暴露，避免同名 API 静默切换语义。
"""

from .ta_settings import (
    TA_EXPERIMENT_NAME,
    TATemplateSettings,
    TAAbsorptionSettings,
    TAStandardizeSettings,
    TASettings,
)
from .ta_case_plan import (
    TACheckpointSettings,
    TAPlanIOSettings,
    TAExecutionPolicy,
    TAPulseCenters,
    TADelayCasePlan,
    TADelayScanPlan,
    TAPlan,
    compute_pulse_centers,
    make_delay_case_name,
    make_delay_case_plan,
)
from .ta_result import (
    TASpectrum,
    TADelayResult,
    TAResult,
    TAResultIO,
)
from .ta_recipe_v2 import (
    TAContrastResult,
    TADelayCenters,
    TADelayScanMap as TADelayScanMapV2,
    TADelayScanPlan as TADelayScanPlanV2,
    TADelayScanResult as TADelayScanResultV2,
    TAPhaseCycledPumpProbeResult,
    TAPhaseCyclingSpec,
    TAReadoutBundle,
    TASubtractionSpec,
    TASingleDelayPairResult,
    TASingleDelayPlan,
    build_ta_phase_cycled_pump_probe_bundle,
    build_ta_pump_probe_phase_cycling_plan,
    build_ta_delay_scan_map,
    compute_ta_contrast,
    extract_ta_absorption_bundle,
    validate_ta_contrast_axes_for_scan,
    validate_ta_readout_bundle_axes,
)

LegacyTASettings = TASettings
LegacyTAPlan = TAPlan
LegacyTADelayScanPlan = TADelayScanPlan
LegacyTAResult = TAResult
LegacyTAResultIO = TAResultIO

__all__ = [
    "TA_EXPERIMENT_NAME",
    "TATemplateSettings",
    "TAAbsorptionSettings",
    "TAStandardizeSettings",
    "TASettings",
    "TACheckpointSettings",
    "TAPlanIOSettings",
    "TAExecutionPolicy",
    "TAPulseCenters",
    "TADelayCasePlan",
    "TADelayScanPlan",
    "TAPlan",
    "compute_pulse_centers",
    "make_delay_case_name",
    "make_delay_case_plan",
    "TASpectrum",
    "TADelayResult",
    "TAResult",
    "TAResultIO",
    "LegacyTASettings",
    "LegacyTAPlan",
    "LegacyTADelayScanPlan",
    "LegacyTAResult",
    "LegacyTAResultIO",
    "TAContrastResult",
    "TADelayCenters",
    "TADelayScanMapV2",
    "TADelayScanPlanV2",
    "TADelayScanResultV2",
    "TAPhaseCycledPumpProbeResult",
    "TAPhaseCyclingSpec",
    "TAReadoutBundle",
    "TASubtractionSpec",
    "TASingleDelayPairResult",
    "TASingleDelayPlan",
    "build_ta_phase_cycled_pump_probe_bundle",
    "build_ta_pump_probe_phase_cycling_plan",
    "build_ta_delay_scan_map",
    "compute_ta_contrast",
    "extract_ta_absorption_bundle",
    "validate_ta_contrast_axes_for_scan",
    "validate_ta_readout_bundle_axes",
]
