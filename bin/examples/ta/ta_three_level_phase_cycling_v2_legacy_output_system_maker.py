#!/usr/bin/env python3
"""System-maker + TA v2 phase-cycling legacy-output validation demo.

这是一个桥接验证脚本，不替代旧的
`ta_three_level_intrinsic_response_phase_cycling_demo.py`，也不修改旧 demo
的 checkpoint / output 行为。

本脚本使用 `qudpy_sjh.systems` 构造三能级 single-exciton ladder system，
再通过 systems adapter 生成 `NLevelPhysicalParams`，随后复用
`bin/dev/ta_three_level_phase_cycling_v2_legacy_output.py` 的全套 v2 计算与
legacy-output 写出逻辑。因此输出应与 v2 legacy-output 脚本对齐，包括：

    checkpoints/carrier_envelope_v2/*.ckp
    data/map_stats.csv
    data/map_stats.json
    data/ta_all_delay_spectra.csv
    data/ta_phase_cycling_comparison.npz
    meta.json
    figures/plot/*
    figures/legacy/*
    figures/preview/*

未来它可以作为旧 demo 迁移到 TA recipe v2 + systems maker 的桥接脚本。
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

LEGACY_OUTPUT_RUNNER_PATH = REPO_ROOT / "bin" / "dev" / "ta_three_level_phase_cycling_v2_legacy_output.py"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "bin" / "optical_bloch_plots" / "ta_three_level_phase_cycling_v2_system_maker"

from qudpy_sjh.systems import make_base_physical_params_from_system, make_single_exciton_ladder_system  # noqa: E402
from qudpy_sjh.utils.core import PureDephasingChannel  # noqa: E402


def _load_runner_module():
    if not LEGACY_OUTPUT_RUNNER_PATH.exists():
        raise FileNotFoundError(LEGACY_OUTPUT_RUNNER_PATH)
    spec = importlib.util.spec_from_file_location(
        "ta_three_level_phase_cycling_v2_legacy_output_runner",
        LEGACY_OUTPUT_RUNNER_PATH,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load runner module: {LEGACY_OUTPUT_RUNNER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _make_system_from_legacy_config(config):
    """用 systems maker 复现旧 demo 的三能级 matter-side 参数。"""

    energy_0x = float(config.energies_eV[1] - config.energies_eV[0])
    energy_x_xx = float(config.energies_eV[2] - config.energies_eV[1])
    mu_0x = float(config.dipole_matrix_D[0][1])
    mu_x_xx = float(config.dipole_matrix_D[1][2])
    gamma_0x = 1.0 / float(config.Tphi_1_fs)
    gamma_x_xx = 1.0 / float(config.Tphi_2_fs)

    eis_eV = energy_x_xx - energy_0x
    pb = mu_x_xx / (math.sqrt(2.0) * mu_0x)
    eid = gamma_x_xx / gamma_0x

    system = make_single_exciton_ladder_system(
        n_quantum=2,
        energy_1q_eV=energy_0x,
        mu_1q_D=mu_0x,
        gamma_1q_fs_inv=gamma_0x,
        eis_eV=eis_eV,
        pb=pb,
        eid=eid,
        initial_state="ground",
        name="ta_three_level_system_maker",
        metadata={
            "legacy_demo_reference": "ta_three_level_intrinsic_response_phase_cycling_demo.py",
            "legacy_basis": tuple(config.basis),
            "legacy_energies_eV": tuple(float(x) for x in config.energies_eV),
            "legacy_dipole_matrix_D": tuple(tuple(float(v) for v in row) for row in config.dipole_matrix_D),
            "legacy_Tphi_1_fs": float(config.Tphi_1_fs),
            "legacy_Tphi_2_fs": float(config.Tphi_2_fs),
            "system_maker_bridge": True,
        },
    )
    return system, {
        "eis_eV": float(eis_eV),
        "pb": float(pb),
        "eid": float(eid),
        "energy_0x_eV": energy_0x,
        "energy_x_xx_eV": energy_x_xx,
        "mu_0x_D": mu_0x,
        "mu_x_xx_D": mu_x_xx,
        "gamma_0x_fs_inv": gamma_0x,
        "gamma_x_xx_fs_inv": gamma_x_xx,
    }


def _build_system_maker_base_params(legacy, smoke_v2, config, probe):
    system, corrections = _make_system_from_legacy_config(config)
    base_params = make_base_physical_params_from_system(
        system,
        field=probe.field_template,
        t_start_fs=float(config.t_start_fs),
        t_end_fs=float(config.t_end_fs),
        dt_fs=float(config.dt_fs),
        solver_mode="lab_exact",
        pure_dephasing_channels=(
            PureDephasingChannel(
                name="pure_dephasing_level_1",
                level=1,
                Tphi_fs=float(config.Tphi_1_fs),
            ),
            PureDephasingChannel(
                name="pure_dephasing_level_2",
                level=2,
                Tphi_fs=float(config.Tphi_2_fs),
            ),
        ),
        input_description="TA v2 legacy-output system-maker bridge base params.",
        input_metadata={
            "system_maker_script": str(Path(__file__).resolve()),
            "system_maker_corrections": corrections,
            "transition_dephasing_note": (
                "NLevelSystem transition_dephasing_fs_inv is metadata-only for "
                "NLevelPhysicalParams; old demo level pure dephasing channels are "
                "passed explicitly to preserve numerical behavior."
            ),
            "source_smoke_builder_used_for_pulses_readout": True,
        },
    )
    return base_params, {
        "system": system.to_dict(include_arrays=True),
        "system_maker_corrections": corrections,
        "adapter": base_params.input_metadata.get("system_adapter"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--force-run", action="store_true", help="Ignore existing checkpoints and rerun all simulations.")
    parser.add_argument("--force", action="store_true", help="Alias for --force-run.")
    parser.add_argument("--no-checkpoints", action="store_true", help="Run without checkpoint load/save.")
    parser.add_argument("--quick", action="store_true", help="Use the v2 smoke quick grid for fast validation.")
    parser.add_argument("--wavelength", action="store_true", help="Plot wavelength instead of photon energy on the x-axis.")
    parser.add_argument("--max-delays", type=int, default=None, help="Diagnostic option: run only the first N selected delays.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.force_run = bool(args.force_run or args.force)
    runner = _load_runner_module()
    result = runner.run_v2_legacy_output(
        args,
        base_params_builder=_build_system_maker_base_params,
        example_name="ta_three_level_phase_cycling_v2_legacy_output_system_maker",
        workflow_extra={
            "system_maker_bridge": True,
            "system_maker_api": "make_single_exciton_ladder_system",
            "adapter_api": "make_base_physical_params_from_system",
        },
    )
    metadata = result.get("base_params_builder", {}) if isinstance(result, dict) else {}
    corrections = metadata.get("system_maker_corrections", {})
    if corrections:
        print(f"system-maker EIS/PB/EID: {corrections}")


if __name__ == "__main__":
    main()
