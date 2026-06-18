#!/usr/bin/env python3
"""检查 N=2 physical mainline 的封装层与显式归一化路径是否一致。"""

from __future__ import annotations

from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sjh_learn.utils.core import (
    NLevelPhysicalParams,
    ParaNormalizer,
    PureDephasingChannel,
    RelaxationChannel,
)
from sjh_learn.utils.checks import n2_mainline_equivalence_check
from sjh_learn.utils.fields import make_default_carrier_field


def make_physical_params() -> NLevelPhysicalParams:
    """把旧 N=2 code-unit 检查映射成普通 `NLevelPhysicalParams`。

    `time_scale_fs=1.0` 时，1 fs^-1 对应 1 solver code unit。这里把旧检查中
    的目标耦合强度反推出 `dipole_matrix_D`，从而继续保持数值尺度接近旧版。
    """

    field_MV_per_cm = 0.7
    old_code_dipole_01 = 0.08
    dipole_01_D = old_code_dipole_01 / ParaNormalizer.DIPOLE_FIELD_TO_RABI_FS_INV
    return NLevelPhysicalParams(
        basis=("g", "e"),
        energies_eV=(0.0, float(ParaNormalizer.fs_inv_to_energy_eV(1.25))),
        dipole_matrix_D=((0.0, dipole_01_D), (dipole_01_D, 0.0)),
        t_start_fs=0.0,
        t_end_fs=20.0,
        dt_fs=0.01,
        field=make_default_carrier_field(
            E0_MV_per_cm=field_MV_per_cm,
            laser_energy_eV=float(ParaNormalizer.fs_inv_to_energy_eV(1.0)),
        ),
        relaxation_channels=(
            RelaxationChannel(name="relaxation_1_to_0", from_level=1, to_level=0, rate_fs_inv=0.03),
        ),
        pure_dephasing_channels=(
            PureDephasingChannel(name="pure_dephasing_level_0", level=0, rate_fs_inv=0.02),
            PureDephasingChannel(name="pure_dephasing_level_1", level=1, rate_fs_inv=0.02),
        ),
    )


def main() -> None:
    normalizer = ParaNormalizer(time_scale_fs=1.0, auto_scale=False)
    differences = n2_mainline_equivalence_check(make_physical_params(), normalizer=normalizer)
    print("N=2 physical mainline equivalence check")
    for key, value in differences.items():
        print(f"{key:<24}: {value:.6e}")


if __name__ == "__main__":
    main()
