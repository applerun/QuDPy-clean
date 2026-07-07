#!/usr/bin/env python3
"""最小 N=3 lab-frame 示例。

这个脚本走正式主线：
`NLevelPhysicalParams -> ParaNormalizer -> run_case`。
旧的 solver-ready multilevel 路径不再作为普通 multilevel 入口。
"""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qudpy_sjh.utils.core import (
    NLevelPhysicalParams,
    ParaNormalizer,
    RelaxationChannel,
    run_case,
)
from qudpy_sjh.utils.fields.carrier_envelope import make_constant_carrier_envelope_field
from qudpy_sjh.utils.io import (
    QuantumResultIO,
    save_figure,
)
from qudpy_sjh.utils.plotting import plot_multilevel_components


OUTPUT_PATH = Path(__file__).resolve().parent / "optical_bloch_plots" / "multilevel_demo.png"
RESULT_IO = QuantumResultIO(str(Path(__file__).resolve().parent / "optical_bloch_plots" / "quantum_results_single"))


def make_demo_params() -> NLevelPhysicalParams:
    """构造一个 N=3 物理参数对象，并尽量保持旧 demo 的数值尺度。

    这里使用 `time_scale_fs=1.0` 的手动 normalizer，因此 1 fs^-1 会对应
    1 code unit。为了保留旧 demo 中的光场耦合强度，先设定目标
    coupling_matrix_code，再反推出 `dipole_matrix_D`。普通用户示例仍然只
    接触 `NLevelPhysicalParams` 的物理单位字段。
    """

    field_MV_per_cm = 0.5
    old_code_energies = np.array([0.0, 1.0, 1.5], dtype=float)
    old_code_dipole = np.array(
        [
            [0.0, 0.05, 0.0],
            [0.05, 0.0, 0.03],
            [0.0, 0.03, 0.0],
        ],
        dtype=float,
    )
    target_coupling_code = field_MV_per_cm * old_code_dipole
    dipole_matrix_D = target_coupling_code / (
        field_MV_per_cm * ParaNormalizer.DIPOLE_FIELD_TO_RABI_FS_INV
    )

    return NLevelPhysicalParams(
        basis=("g", "e1", "e2"),
        energies_eV=tuple(float(ParaNormalizer.fs_inv_to_energy_eV(value)) for value in old_code_energies),
        dipole_matrix_D=tuple(tuple(float(item) for item in row) for row in dipole_matrix_D),
        t_start_fs=0.0,
        t_end_fs=120.0,
        dt_fs=0.05,
        field=make_constant_carrier_envelope_field(
            E0_MV_per_cm=field_MV_per_cm,
            laser_energy_eV=float(ParaNormalizer.fs_inv_to_energy_eV(1.0)),
            name="multilevel_demo_constant_field",
        ),
        relaxation_channels=(
            RelaxationChannel(name="relaxation_2_to_1", from_level=2, to_level=1, rate_fs_inv=0.02),
            RelaxationChannel(name="relaxation_1_to_0", from_level=1, to_level=0, rate_fs_inv=0.01),
        ),
    )


def main() -> None:
    physical = make_demo_params()
    normalizer = ParaNormalizer(time_scale_fs=1.0, auto_scale=False)
    result = run_case(physical, normalizer=normalizer)

    fig, _axes = plot_multilevel_components(result, populations=None, coherences=[(0, 1), (1, 2)])
    save_figure(fig, OUTPUT_PATH, dpi=160)
    plt.close(fig)

    case_name = "multilevel_N3_physical_mainline"
    saved_case = RESULT_IO.save_case(
        result,
        output_data=True,
        output_preview=False,
        save_npz=True,
        save_csv=True,
        save_json=True,
        example_name="multilevel_demo",
        condition_name="N3_lab_exact",
        case_name=case_name,
        selected_elements={"rho_01": (0, 1), "rho_12": (1, 2)},
    )

    print("N=3 lab-frame demo using NLevelPhysicalParams")
    print(f"dimension           : {result.dimension()}")
    print(f"final populations   : {[float(value.real) for value in result.populations()[-1]]}")
    print(f"trace error         : {result.max_trace_error():.3e}")
    print(f"Hermiticity error   : {result.max_hermiticity_error():.3e}")
    print(f"output plot         : {OUTPUT_PATH}")
    print(f"result case dir     : {saved_case['case_dir']}")
    print(f"sanity checks       : {result.sanity_checks}")


if __name__ == "__main__":
    main()
