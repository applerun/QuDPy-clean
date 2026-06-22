#!/usr/bin/env python3
"""Three-level intrinsic TA response using the refactored TA workflow."""

from __future__ import annotations

from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    # Path:
    #   QuDPy-clean/bin/examples/ta/ta_three_level_intrinsic_response_plan_demo.py
    # parents[3] is the repository root.
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from qudpy_sjh.experiments.ta import (
    TACheckpointSettings,
    TAAbsorptionSettings,
    TAPlan,
    TAPlanIOSettings,
    TASettings,
    TAStandardizeSettings,
    TATemplateSettings,
)
from qudpy_sjh.utils.core import (
    NLevelPhysicalParams,
    PureDephasingChannel,
    RelaxationChannel,
)
from qudpy_sjh.utils.fields import FieldPhySeries, make_default_gaussian_carrier_field


EXAMPLE_NAME = "ta_three_level_intrinsic_response_plan_demo"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / EXAMPLE_NAME


def make_base_params() -> NLevelPhysicalParams:
    pump_template = make_default_gaussian_carrier_field(
        E0_MV_per_cm=0.18,
        laser_energy_eV=1.55,
        pulse_center_fs=0.0,
        pulse_sigma_fs=12.0,
        phase_rad=0.0,
        name="pump_template",
    )
    probe_template = make_default_gaussian_carrier_field(
        E0_MV_per_cm=0.012,
        laser_energy_eV=1.65,
        pulse_center_fs=0.0,
        pulse_sigma_fs=7.0,
        phase_rad=0.0,
        name="probe_template",
    )
    field_template = FieldPhySeries(
        fields=(pump_template, probe_template),
        sub_field_names=("pump", "probe"),
        name="zero_centered_ta_templates",
        metadata={"role": "ta_zero_centered_templates"},
    )
    return NLevelPhysicalParams(
        energies_eV=(0.0, 1.55, 3.30),
        dipole_matrix_D=(
            (0.0, 5.0, 0.0),
            (5.0, 0.0, 4.0),
            (0.0, 4.0, 0.0),
        ),
        # This grid must cover the full shifted pump/probe sequence.
        t_start_fs=-1700.0,
        t_end_fs=500.0,
        dt_fs=0.2,
        field=field_template,
        basis=("g", "e", "f"),
        relaxation_channels=(
            RelaxationChannel(name="relaxation_2_to_1", from_level=2, to_level=1, T1_fs=150.0),
            RelaxationChannel(name="relaxation_1_to_0", from_level=1, to_level=0, T1_fs=350.0),
        ),
        pure_dephasing_channels=(
            PureDephasingChannel(name="pure_dephasing_level_1", level=1, Tphi_fs=90.0),
            PureDephasingChannel(name="pure_dephasing_level_2", level=2, Tphi_fs=80.0),
        ),
        solver_mode="lab_exact",
        input_description="Three-level ladder template for intrinsic TA response.",
        input_metadata={
            "example_name": EXAMPLE_NAME,
            "model_note": "Demonstration parameters; not fitted to a specific material.",
        },
    )


def make_plan() -> TAPlan:
    settings = TASettings(
        base_params=make_base_params(),
        probe_delays_fs=(
            -300.0, -220.0, -160.0, -110.0, -80.0, -60.0, -45.0,
            -30.0, -20.0, -10.0, 0.0, 10.0, 20.0, 30.0, 45.0,
            60.0, 80.0, 110.0, 150.0, 220.0, 320.0, 460.0,
            650.0, 900.0, 1200.0, 1500.0,
        ),
        probe_center_fs=0.0,
        experiment_name=EXAMPLE_NAME,
        template=TATemplateSettings(
            pump_template_center_fs=0.0,
            probe_template_center_fs=0.0,
        ),
        absorption=TAAbsorptionSettings(
            number_density_m3=1.0e24,
            window="hann",
            subtract_mean=True,
            rel_threshold=1.0e-6,
            zero_padding_factor=4,
            return_intermediates=True,
        ),
        standardize=TAStandardizeSettings(
            allow_energy_axis_interpolation=True,
            common_axis_policy="overlap",
            kinetic_energy_eV=1.55,
        ),
        metadata={"purpose": "refactored TA workflow demo"},
        max_time_points=20000,
    )
    return TAPlan(
        settings=settings,
        checkpoint=TACheckpointSettings(
            enabled=True,
            force_run=True,
            checkpoint_dir_name="checkpoints",
        ),
        io=TAPlanIOSettings(
            output_dir=DEFAULT_OUTPUT_DIR,
            save_default_outputs_after_execute=True,
            save_ta_preview_figures=True,
            ta_preview_dir_name="figures",
            ta_preview_energy_range_eV=(1.25, 1.95),
            ta_preview_cmap="plasma",
            selected_lineout_delays_fs=(-220.0, -60.0, 0.0, 45.0, 220.0, 650.0, 1200.0),
            preview_case_dir_name="res_per_delay",
            save_case_previews=True,
            save_probe_only_preview=True,
            preview_delay_cases_fs=(-60.0, 0.0, 45.0, 220.0, 650.0, 1200.0),
            preview_all_delay_cases=False,
        ),
    )


def main() -> None:
    plan = make_plan()

    # Stage 1: compute dynamics, save checkpoints, and save standardized TA outputs.
    result = plan.execute()

    # Stage 2: optional raw DynamicsResult preview/export from existing checkpoints.
    # This does not rerun the solver.
    preview_paths = plan.save_preview_from_checkpoints()

    print("TA workflow finished.")
    print(f"n delays        : {len(result.delays_fs)}")
    print(f"energy points   : {result.common_energy_eV.size}")
    print(f"output directory: {plan.output_dir}")
    print(f"preview cases   : {list(preview_paths.keys())}")


if __name__ == "__main__":
    main()
