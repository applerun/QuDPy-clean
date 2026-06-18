"""core 层公开入口。

core 只负责参数、归一化、Hamiltonian / collapse operator 构造、求解入口和
`DynamicsResult` 容器；不要从这里依赖 analysis observable。
"""

from .model import build_c_ops, build_lab_hamiltonian, compute_detuning, compute_energy_gap
from .config import FORCE_RWA, RWA_DISABLED_MESSAGE, ensure_rwa_enabled
from .normalization import ParaNormalizer
from .parameters import (
    NLevelPhysicalParams,
    NLevelSolverParams,
    PureDephasingChannel,
    RelaxationChannel,
    SolverParams,
)
from .results import DynamicsResult
from .solvers import (
    make_rotating_view,
    run_case,
    run_cases,
)

__all__ = [
    "DynamicsResult",
    "FORCE_RWA",
    "RWA_DISABLED_MESSAGE",
    "ensure_rwa_enabled",
    "NLevelPhysicalParams",
    "NLevelSolverParams",
    "RelaxationChannel",
    "PureDephasingChannel",
    "SolverParams",
    "ParaNormalizer",
    "run_case",
    "run_cases",
    "make_rotating_view",
]
