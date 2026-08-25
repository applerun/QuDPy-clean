"""Transient absorption workflow helpers.

Canonical recipe-first exports are listed first. Legacy TA v1/v2 names remain
importable for compatibility and historical validation.
"""

from .ta_recipe_first import (
    TAPrePCObservable,
    TAPrePCRecipe,
    build_ta_pre_pc_observable,
)

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
    "TAPrePCObservable",
    "TAPrePCRecipe",
    "build_ta_pre_pc_observable",
    # Legacy TA v1/v2 compatibility exports.
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
