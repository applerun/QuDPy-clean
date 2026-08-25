"""Generic pulse-sequence scaffolds for one concrete field configuration."""

from qudpy_sjh.experiments.readout import (
    PolarizationResult,
    ReadoutPlan,
    ReadoutResult,
    coherent_detector_terms,
    compute_polarization_result,
    resolve_readout_field,
)

from .phase_projection import (
    PHASE_PROJECTION_CONVENTION,
    PHASE_PROJECTION_CONVENTION_VERSION,
    TARGET_PHASE_VECTOR_SEMANTICS,
    PhaseGrid,
    build_uniform_phase_grid,
    project_phase_orders,
)
from .pulse_sequence import (
    FieldGroupSpec,
    PulseSequenceSpec,
    PulseSpec,
    SingleRunFieldPlan,
    is_supported_phase_backend,
    normalize_phase_vector,
    supports_phase_override,
    validate_phase_tag,
    validate_pulse_name,
)
from .single_run import (
    SingleRunCheckpointSettings,
    SingleRunPlan,
    SingleRunResult,
)

__all__ = [
    "PHASE_PROJECTION_CONVENTION",
    "PHASE_PROJECTION_CONVENTION_VERSION",
    "TARGET_PHASE_VECTOR_SEMANTICS",
    "FieldGroupSpec",
    "PhaseGrid",
    "PolarizationResult",
    "PulseSequenceSpec",
    "PulseSpec",
    "ReadoutPlan",
    "ReadoutResult",
    "SingleRunCheckpointSettings",
    "SingleRunFieldPlan",
    "SingleRunPlan",
    "SingleRunResult",
    "build_uniform_phase_grid",
    "coherent_detector_terms",
    "compute_polarization_result",
    "is_supported_phase_backend",
    "normalize_phase_vector",
    "project_phase_orders",
    "resolve_readout_field",
    "supports_phase_override",
    "validate_phase_tag",
    "validate_pulse_name",
]
