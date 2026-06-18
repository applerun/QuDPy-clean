"""物理常数和单位换算常数。

优先使用 `scipy.constants`，这样常数来源清楚且便于审计；如果运行环境
没有 scipy，退回到 SI 定义值或本项目已有固定换算值。
"""

from __future__ import annotations

try:
    from scipy import constants

    C_LIGHT = float(constants.c)
    H_PLANCK = float(constants.h)
    HBAR_J_S = float(constants.hbar)
    E_CHARGE_C = float(constants.eV)
    EPSILON0_F_PER_M = float(constants.epsilon_0)
    FS_TO_S = float(constants.femto)
    MV_PER_CM_TO_V_PER_M = float(constants.mega / constants.centi)
    DEBYE_TO_C_M = float(constants.physical_constants["debye"][0])
except (ImportError, AttributeError, KeyError):
    C_LIGHT = 299792458.0
    H_PLANCK = 6.62607015e-34
    HBAR_J_S = 1.054571817e-34
    E_CHARGE_C = 1.602176634e-19
    EPSILON0_F_PER_M = 8.8541878128e-12
    FS_TO_S = 1e-15
    MV_PER_CM_TO_V_PER_M = 1e8
    DEBYE_TO_C_M = 3.33564e-30

E_CHARGE = E_CHARGE_C
EV_TO_FS_INV = (E_CHARGE_C / HBAR_J_S) * FS_TO_S
DIPOLE_FIELD_TO_RABI_FS_INV = (
    DEBYE_TO_C_M * MV_PER_CM_TO_V_PER_M / HBAR_J_S
) * FS_TO_S

__all__ = [
    "C_LIGHT",
    "H_PLANCK",
    "HBAR_J_S",
    "E_CHARGE",
    "E_CHARGE_C",
    "EPSILON0_F_PER_M",
    "FS_TO_S",
    "DEBYE_TO_C_M",
    "MV_PER_CM_TO_V_PER_M",
    "EV_TO_FS_INV",
    "DIPOLE_FIELD_TO_RABI_FS_INV",
]
