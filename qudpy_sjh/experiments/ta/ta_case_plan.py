"""Full-window TA delay-scan plans.

这是 experimental TA recipe v1 prototype 的执行层，包含 pump/probe delay
scan、probe-only reference、TA subtraction 和 TA output policy 等
TA-specific 语义。它不是 generic pulse-sequence simulation framework，
当前 phase-cycling validation demo 没有使用它，也不默认执行 phase
cycling。phase cycling 应由上层 wrapper / generic cycler 负责；未来 TA
recipe v2 可以逐步调用 generic pulse-sequence 基础层。

This is the top-level TA orchestration layer.  It owns execution policy,
checkpoint policy, and default standardized TA output policy.  It does not
implement piecewise propagation, dark propagation, active/readout windows, or
materialization.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field, replace
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np

from qudpy_sjh.utils.core import DynamicsResult, ParaNormalizer, run_case
from qudpy_sjh.utils.fields import FieldPhyRoot, FieldPhySeries
from qudpy_sjh.utils.spectroscopy import lab_frame_absorption_response, polarization_C_per_m2

try:
    from qudpy_sjh.utils.io import save_result_case
except Exception:  # pragma: no cover
    save_result_case = None

from .ta_result import (
    TADelayResult,
    TAResult,
    TAResultIO,
    TASpectrum,
    build_common_ta_map,
    nearest_value,
    safe_delay_label,
)
from .ta_settings import TASettings


@dataclass(frozen=True)
class TACheckpointSettings:
    enabled: bool = True
    force_run: bool = False
    checkpoint_dir_name: str = "checkpoints"
    require_existing_for_load_only: bool = True
    require_existing_for_preview: bool = True

    def __post_init__(self) -> None:
        if not self.checkpoint_dir_name:
            raise ValueError("checkpoint_dir_name must be non-empty.")
        object.__setattr__(self, "enabled", bool(self.enabled))
        object.__setattr__(self, "force_run", bool(self.force_run))
        object.__setattr__(self, "require_existing_for_load_only", bool(self.require_existing_for_load_only))
        object.__setattr__(self, "require_existing_for_preview", bool(self.require_existing_for_preview))


@dataclass(frozen=True)
class TAPlanIOSettings:
    """Plan-level output policy.

    ``TASettings`` does not contain IO.  The plan owns output/checkpoint paths.
    """

    output_dir: Path | str = Path("outputs") / "ta_intrinsic_response"
    save_default_outputs_after_execute: bool = True

    # Standard TA result figures.  These are TA-level previews, not raw
    # DynamicsResult previews from core IO.
    save_ta_preview_figures: bool = True
    ta_preview_dir_name: str = "figures"
    ta_preview_dpi: int = 180
    ta_preview_energy_range_eV: tuple[float, float] | None = None
    ta_preview_cmap: str = "plasma"
    selected_lineout_delays_fs: Sequence[float] = ()

    # Raw DynamicsResult preview/export from checkpoints.
    preview_case_dir_name: str = "res_per_delay"
    save_case_previews: bool = False
    save_probe_only_preview: bool = True
    preview_delay_cases_fs: Sequence[float] = ()
    preview_all_delay_cases: bool = False
    append_preview_results_csv: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        object.__setattr__(self, "save_default_outputs_after_execute", bool(self.save_default_outputs_after_execute))
        object.__setattr__(self, "save_ta_preview_figures", bool(self.save_ta_preview_figures))
        if not self.ta_preview_dir_name:
            raise ValueError("ta_preview_dir_name must be non-empty.")
        object.__setattr__(self, "ta_preview_dpi", int(self.ta_preview_dpi))
        if self.ta_preview_dpi <= 0:
            raise ValueError("ta_preview_dpi must be positive.")
        if self.ta_preview_energy_range_eV is not None:
            lo, hi = self.ta_preview_energy_range_eV
            lo = float(lo)
            hi = float(hi)
            if hi <= lo:
                raise ValueError("ta_preview_energy_range_eV must be (min, max) with max > min.")
            object.__setattr__(self, "ta_preview_energy_range_eV", (lo, hi))
        object.__setattr__(
            self,
            "selected_lineout_delays_fs",
            tuple(float(item) for item in np.asarray(self.selected_lineout_delays_fs, dtype=float).reshape(-1)),
        )

        if not self.preview_case_dir_name:
            raise ValueError("preview_case_dir_name must be non-empty.")
        object.__setattr__(self, "save_case_previews", bool(self.save_case_previews))
        object.__setattr__(self, "save_probe_only_preview", bool(self.save_probe_only_preview))
        object.__setattr__(
            self,
            "preview_delay_cases_fs",
            tuple(float(item) for item in np.asarray(self.preview_delay_cases_fs, dtype=float).reshape(-1)),
        )
        object.__setattr__(self, "preview_all_delay_cases", bool(self.preview_all_delay_cases))
        object.__setattr__(self, "append_preview_results_csv", bool(self.append_preview_results_csv))


@dataclass(frozen=True)
class TAExecutionPolicy:
    print_progress: bool = True
    keep_dynamics_results: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "print_progress", bool(self.print_progress))
        object.__setattr__(self, "keep_dynamics_results", bool(self.keep_dynamics_results))


@dataclass(frozen=True)
class TAPulseCenters:
    delay_fs: float
    pump_center_fs: float
    probe_center_fs: float

    def to_dict(self) -> dict[str, float]:
        return {
            "delay_fs": float(self.delay_fs),
            "pump_center_fs": float(self.pump_center_fs),
            "probe_center_fs": float(self.probe_center_fs),
        }


@dataclass(frozen=True)
class TADelayCasePlan:
    """One full-window pump+probe delay case."""

    delay_fs: float
    case_name: str
    pump_center_fs: float
    probe_center_fs: float
    pump_shift_fs: float
    probe_shift_fs: float
    field: FieldPhySeries
    probe_field: FieldPhyRoot
    metadata: dict[str, Any] = dataclass_field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.field, FieldPhySeries):
            raise TypeError("TADelayCasePlan.field must be a FieldPhySeries.")
        if not isinstance(self.probe_field, FieldPhyRoot):
            raise TypeError("TADelayCasePlan.probe_field must be a FieldPhyRoot.")
        object.__setattr__(self, "delay_fs", float(self.delay_fs))
        object.__setattr__(self, "pump_center_fs", float(self.pump_center_fs))
        object.__setattr__(self, "probe_center_fs", float(self.probe_center_fs))
        object.__setattr__(self, "pump_shift_fs", float(self.pump_shift_fs))
        object.__setattr__(self, "probe_shift_fs", float(self.probe_shift_fs))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def execute(self, plan: "TAPlan", *, probe_only_spectrum: TASpectrum) -> TADelayResult:
        return plan.execute_delay_case(self, probe_only_spectrum=probe_only_spectrum)

    def to_dict(self) -> dict[str, Any]:
        return {
            "delay_fs": float(self.delay_fs),
            "case_name": self.case_name,
            "pump_center_fs": float(self.pump_center_fs),
            "probe_center_fs": float(self.probe_center_fs),
            "pump_shift_fs": float(self.pump_shift_fs),
            "probe_shift_fs": float(self.probe_shift_fs),
            "field": self.field.to_dict(),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class TADelayScanPlan:
    settings: TASettings

    def iter_delay_cases(self) -> Iterator[TADelayCasePlan]:
        for delay_fs in self.settings.delays_fs:
            yield make_delay_case_plan(self.settings, delay_fs=delay_fs)

    def execute(self, plan: "TAPlan") -> TAResult:
        return plan.execute()


def compute_pulse_centers(*, delay_fs: float, probe_center_fs: float = 0.0) -> TAPulseCenters:
    probe_center = float(probe_center_fs)
    pump_center = probe_center - float(delay_fs)
    return TAPulseCenters(
        delay_fs=float(delay_fs),
        pump_center_fs=float(pump_center),
        probe_center_fs=float(probe_center),
    )


def make_delay_case_name(*, delay_fs: float) -> str:
    return f"delay_{safe_delay_label(delay_fs)}_fs_pump_probe"


def make_delay_case_plan(settings: TASettings, *, delay_fs: float) -> TADelayCasePlan:
    centers = compute_pulse_centers(delay_fs=delay_fs, probe_center_fs=settings.probe_center_fs)
    pump_shift = settings.pump_shift_fs_for_delay(delay_fs)
    probe_shift = settings.probe_shift_fs()

    pump_field = settings.pump_template.time_shifted(
        pump_shift,
        name="pump",
        metadata={
            "role": "ta_pump",
            "delay_fs": float(delay_fs),
            "desired_center_fs": float(centers.pump_center_fs),
            "template_center_fs": float(settings.template.pump_template_center_fs),
        },
    )
    probe_field = settings.probe_template.time_shifted(
        probe_shift,
        name="probe",
        metadata={
            "role": "ta_probe",
            "delay_fs": float(delay_fs),
            "desired_center_fs": float(centers.probe_center_fs),
            "template_center_fs": float(settings.template.probe_template_center_fs),
        },
    )

    case_name = make_delay_case_name(delay_fs=delay_fs)
    field = FieldPhySeries(
        fields=(pump_field, probe_field),
        sub_field_names=("pump", "probe"),
        name=case_name,
        metadata={
            "role": "ta_pump_probe_field",
            "delay_fs": float(delay_fs),
            "probe_center_fs": float(centers.probe_center_fs),
            "pump_center_fs": float(centers.pump_center_fs),
            "delay_convention": "pump_center_fs = probe_center_fs - delay_fs",
        },
    )

    return TADelayCasePlan(
        delay_fs=float(delay_fs),
        case_name=case_name,
        pump_center_fs=float(centers.pump_center_fs),
        probe_center_fs=float(centers.probe_center_fs),
        pump_shift_fs=float(pump_shift),
        probe_shift_fs=float(probe_shift),
        field=field,
        probe_field=probe_field,
        metadata={
            "pulse_centers": centers.to_dict(),
            "delay_convention": "delay_fs > 0 means pump arrives before probe",
        },
    )


@dataclass
class TAPlan:
    """Top-level TA execution plan."""

    settings: TASettings
    normalizer: ParaNormalizer = dataclass_field(default_factory=ParaNormalizer)
    checkpoint: TACheckpointSettings = dataclass_field(default_factory=TACheckpointSettings)
    io: TAPlanIOSettings = dataclass_field(default_factory=TAPlanIOSettings)
    execution: TAExecutionPolicy = dataclass_field(default_factory=TAExecutionPolicy)

    def __post_init__(self) -> None:
        if not isinstance(self.settings, TASettings):
            raise TypeError("settings must be a TASettings instance.")
        if not isinstance(self.checkpoint, TACheckpointSettings):
            raise TypeError("checkpoint must be a TACheckpointSettings instance.")
        if not isinstance(self.io, TAPlanIOSettings):
            raise TypeError("io must be a TAPlanIOSettings instance.")
        if not isinstance(self.execution, TAExecutionPolicy):
            raise TypeError("execution must be a TAExecutionPolicy instance.")

    @property
    def output_dir(self) -> Path:
        return Path(self.io.output_dir)

    @property
    def delay_scan_plan(self) -> TADelayScanPlan:
        return TADelayScanPlan(self.settings)

    def checkpoint_path(self, case_key: str) -> Path:
        return self.output_dir / self.checkpoint.checkpoint_dir_name / f"{case_key}.ckp"

    def probe_only_case_key(self) -> str:
        return "probe_only"

    def delay_case_key(self, delay_fs: float) -> str:
        return make_delay_case_name(delay_fs=delay_fs)

    def make_probe_reference_field(self) -> FieldPhyRoot:
        return self.settings.probe_template.time_shifted(
            self.settings.probe_shift_fs(),
            name="probe",
            metadata={
                "role": "ta_probe_reference",
                "desired_center_fs": float(self.settings.probe_center_fs),
                "template_center_fs": float(self.settings.template.probe_template_center_fs),
            },
        )

    def make_probe_only_params(self, probe_field: FieldPhyRoot):
        return self._replace_base_params(
            field=probe_field,
            case_name=self.probe_only_case_key(),
            input_description="TA probe-only reference shared by all pump-probe delays.",
            extra_metadata={
                "case_type": "probe_only_reference",
                "probe_center_fs": float(self.settings.probe_center_fs),
            },
        )

    def make_pump_probe_params(self, case: TADelayCasePlan):
        return self._replace_base_params(
            field=case.field,
            case_name=case.case_name,
            input_description=f"TA pump+probe delay case, delay_fs={case.delay_fs:g}.",
            extra_metadata={
                "case_type": "pump_probe",
                "delay_fs": float(case.delay_fs),
                "pump_center_fs": float(case.pump_center_fs),
                "probe_center_fs": float(case.probe_center_fs),
                "pump_shift_fs": float(case.pump_shift_fs),
                "probe_shift_fs": float(case.probe_shift_fs),
            },
        )

    def _replace_base_params(
        self,
        *,
        field: FieldPhyRoot,
        case_name: str,
        input_description: str,
        extra_metadata: dict[str, Any],
    ):
        base_meta = dict(getattr(self.settings.base_params, "input_metadata", {}) or {})
        merged_metadata = {
            **base_meta,
            "ta_workflow": {
                "experiment_name": self.settings.experiment_name,
                "case_name": case_name,
                "response_definition": self.settings.standardize.response_definition,
                **extra_metadata,
            },
        }
        return replace(
            self.settings.base_params,
            field=field,
            input_description=input_description,
            input_metadata=merged_metadata,
        )

    def run_case_with_checkpoint(self, params, *, case_key: str) -> DynamicsResult:
        if not self.checkpoint.enabled:
            return run_case(params, normalizer=self.normalizer)
        ckp = self.checkpoint_path(case_key)
        ckp.parent.mkdir(parents=True, exist_ok=True)
        return run_case(
            params,
            normalizer=self.normalizer,
            load_ckp=ckp,
            save_ckp=ckp,
            force_run=bool(self.checkpoint.force_run),
        )

    def load_case_from_checkpoint(self, case_key: str, *, require_existing: bool = True) -> DynamicsResult:
        path = self.checkpoint_path(case_key)
        if require_existing and not path.exists():
            raise FileNotFoundError(path)
        return DynamicsResult.from_ckp(path)

    def spectrum_from_result(self, result: DynamicsResult, probe_field: FieldPhyRoot, *, case_name: str) -> TASpectrum:
        physical = result.physical_params
        if physical is None:
            raise ValueError("DynamicsResult.physical_params is required for TA spectroscopy.")

        time_fs = np.asarray(result.times_fs, dtype=float)
        E_probe = np.asarray(probe_field(time_fs), dtype=float)
        P_t = polarization_C_per_m2(
            result.density_array(),
            physical.dipole_matrix_D,
            float(self.settings.absorption.number_density_m3),
        )

        response = lab_frame_absorption_response(
            time_fs=time_fs,
            polarization_C_per_m2=P_t,
            field=E_probe,
            window=self.settings.absorption.window,
            subtract_mean=bool(self.settings.absorption.subtract_mean),
            rel_threshold=float(self.settings.absorption.rel_threshold),
            zero_padding_factor=int(self.settings.absorption.zero_padding_factor),
            return_intermediates=bool(self.settings.absorption.return_intermediates),
        )
        return TASpectrum.from_response(
            response,
            metadata={
                "case_name": case_name,
                "number_density_m3": float(self.settings.absorption.number_density_m3),
                "field_used_for_denominator": "probe field only",
            },
        )

    def execute_delay_case(self, case: TADelayCasePlan, *, probe_only_spectrum: TASpectrum) -> TADelayResult:
        params = self.make_pump_probe_params(case)
        dyn = self.run_case_with_checkpoint(params, case_key=case.case_name)
        pump_probe_spectrum = self.spectrum_from_result(dyn, case.probe_field, case_name=case.case_name)
        probe_on_axis = probe_only_spectrum.on_axis(
            pump_probe_spectrum.energy_eV,
            allow_interpolation=bool(self.settings.standardize.allow_energy_axis_interpolation),
        )
        ta_absorption = pump_probe_spectrum.absorption - probe_on_axis.absorption
        ta_spectrum = TASpectrum(
            energy_eV=pump_probe_spectrum.energy_eV,
            omega_fs_inv=pump_probe_spectrum.omega_fs_inv,
            absorption=ta_absorption,
            omega_im_P_over_E=ta_absorption,
            metadata={
                "definition": self.settings.standardize.response_definition,
                "delay_fs": float(case.delay_fs),
            },
        )
        return TADelayResult(
            delay_fs=float(case.delay_fs),
            case_name=case.case_name,
            pump_center_fs=float(case.pump_center_fs),
            probe_center_fs=float(case.probe_center_fs),
            field_metadata=case.field.to_dict(),
            pump_probe_spectrum=pump_probe_spectrum,
            probe_only_spectrum_on_axis=probe_on_axis,
            ta_spectrum=ta_spectrum,
            pump_probe_result=dyn if self.execution.keep_dynamics_results else None,
            metadata=case.metadata,
        )

    def execute(self) -> TAResult:
        """Run dynamics, save checkpoints, build TAResult, and save standard outputs."""

        probe_field = self.make_probe_reference_field()
        probe_params = self.make_probe_only_params(probe_field)
        probe_only_result = self.run_case_with_checkpoint(probe_params, case_key=self.probe_only_case_key())
        probe_only_spectrum = self.spectrum_from_result(probe_only_result, probe_field, case_name=self.probe_only_case_key())

        delay_results: list[TADelayResult] = []
        for case in self.delay_scan_plan.iter_delay_cases():
            if self.execution.print_progress:
                print(f"Running TA delay {case.delay_fs:g} fs...")
            delay_results.append(case.execute(self, probe_only_spectrum=probe_only_spectrum))

        result = self._assemble_ta_result(
            probe_field=probe_field,
            probe_only_result=probe_only_result,
            probe_only_spectrum=probe_only_spectrum,
            delay_results=delay_results,
            source="execute",
        )

        if self.io.save_default_outputs_after_execute:
            TAResultIO(output_dir=self.output_dir, io_settings=self.io).save(result)
        return result

    def load_result_from_checkpoints(self) -> TAResult:
        """Rebuild TAResult from existing checkpoints without rerunning solver."""

        if not self.checkpoint.enabled:
            raise RuntimeError("load_result_from_checkpoints requires checkpoint.enabled=True.")

        probe_field = self.make_probe_reference_field()
        probe_only_result = self.load_case_from_checkpoint(
            self.probe_only_case_key(),
            require_existing=bool(self.checkpoint.require_existing_for_load_only),
        )
        probe_only_spectrum = self.spectrum_from_result(probe_only_result, probe_field, case_name=self.probe_only_case_key())

        delay_results = []
        for case in self.delay_scan_plan.iter_delay_cases():
            dyn = self.load_case_from_checkpoint(
                case.case_name,
                require_existing=bool(self.checkpoint.require_existing_for_load_only),
            )
            pump_probe_spectrum = self.spectrum_from_result(dyn, case.probe_field, case_name=case.case_name)
            probe_on_axis = probe_only_spectrum.on_axis(
                pump_probe_spectrum.energy_eV,
                allow_interpolation=bool(self.settings.standardize.allow_energy_axis_interpolation),
            )
            ta_absorption = pump_probe_spectrum.absorption - probe_on_axis.absorption
            delay_results.append(
                TADelayResult(
                    delay_fs=float(case.delay_fs),
                    case_name=case.case_name,
                    pump_center_fs=float(case.pump_center_fs),
                    probe_center_fs=float(case.probe_center_fs),
                    field_metadata=case.field.to_dict(),
                    pump_probe_spectrum=pump_probe_spectrum,
                    probe_only_spectrum_on_axis=probe_on_axis,
                    ta_spectrum=TASpectrum(
                        energy_eV=pump_probe_spectrum.energy_eV,
                        omega_fs_inv=pump_probe_spectrum.omega_fs_inv,
                        absorption=ta_absorption,
                        omega_im_P_over_E=ta_absorption,
                        metadata={
                            "definition": self.settings.standardize.response_definition,
                            "delay_fs": float(case.delay_fs),
                            "rebuilt_from_checkpoint": True,
                        },
                    ),
                    pump_probe_result=dyn if self.execution.keep_dynamics_results else None,
                    metadata={**case.metadata, "rebuilt_from_checkpoint": True},
                )
            )

        return self._assemble_ta_result(
            probe_field=probe_field,
            probe_only_result=probe_only_result,
            probe_only_spectrum=probe_only_spectrum,
            delay_results=delay_results,
            source="load_result_from_checkpoints",
        )

    def _assemble_ta_result(
        self,
        *,
        probe_field: FieldPhyRoot,
        probe_only_result: DynamicsResult | None,
        probe_only_spectrum: TASpectrum,
        delay_results: list[TADelayResult],
        source: str,
    ) -> TAResult:
        common_energy, common_omega, ta_map, pump_probe_map, probe_only_map = build_common_ta_map(
            delay_results,
            common_axis_policy=self.settings.standardize.common_axis_policy,
            allow_energy_axis_interpolation=self.settings.standardize.allow_energy_axis_interpolation,
            min_common_energy_points=self.settings.standardize.min_common_energy_points,
        )
        delays = np.asarray([item.delay_fs for item in delay_results], dtype=float)
        probe_common = probe_only_map[0] if probe_only_map.size else np.asarray([], dtype=float)

        dynamic_items = [item.pump_probe_result for item in delay_results if item.pump_probe_result is not None]
        metadata = {
            "experiment_name": self.settings.experiment_name,
            "source": source,
            "response_definition": self.settings.standardize.response_definition,
            "delay_convention": {
                "probe_center_fs": float(self.settings.probe_center_fs),
                "pump_center_rule": "pump_center_fs = probe_center_fs - delay_fs",
            },
            "n_delays": int(delays.size),
            "checkpoint": {
                "enabled": bool(self.checkpoint.enabled),
                "checkpoint_dir": str(self.output_dir / self.checkpoint.checkpoint_dir_name),
                "force_run": bool(self.checkpoint.force_run),
            },
            "time_grid": {
                "t_start_fs": float(self.settings.base_params.t_start_fs),
                "t_end_fs": float(self.settings.base_params.t_end_fs),
                "dt_fs": float(self.settings.base_params.dt_fs),
            },
            "sanity": {
                "probe_only": None if probe_only_result is None else {
                    "max_trace_error": float(probe_only_result.max_trace_error()),
                    "max_hermiticity_error": float(probe_only_result.max_hermiticity_error()),
                },
                "pump_probe_max": {
                    "max_trace_error": None if not dynamic_items else float(max(item.max_trace_error() for item in dynamic_items)),
                    "max_hermiticity_error": None if not dynamic_items else float(max(item.max_hermiticity_error() for item in dynamic_items)),
                },
            },
        }

        return TAResult(
            settings=self.settings,
            probe_field_metadata=probe_field.to_dict(),
            probe_only_result=probe_only_result if self.execution.keep_dynamics_results else None,
            probe_only_spectrum=probe_only_spectrum,
            delay_results=delay_results,
            common_energy_eV=common_energy,
            common_omega_fs_inv=common_omega,
            delays_fs=delays,
            ta_map=ta_map,
            pump_probe_map=pump_probe_map,
            probe_only_spectrum_on_common_axis=probe_common,
            metadata=metadata,
        )

    def preview_case_keys(self, *, delays_fs: Sequence[float] | None = None, include_probe_only: bool | None = None) -> list[str]:
        include_probe = self.io.save_probe_only_preview if include_probe_only is None else bool(include_probe_only)
        keys: list[str] = []
        if include_probe:
            keys.append(self.probe_only_case_key())

        if self.io.preview_all_delay_cases:
            selected_delays = list(self.settings.delays_fs)
        elif delays_fs is not None:
            selected_delays = [nearest_value(self.settings.delays_fs, item) for item in delays_fs]
        else:
            selected_delays = [nearest_value(self.settings.delays_fs, item) for item in self.io.preview_delay_cases_fs]

        seen = set(keys)
        for delay in selected_delays:
            key = self.delay_case_key(float(delay))
            if key not in seen:
                keys.append(key)
                seen.add(key)
        return keys

    def save_preview_from_checkpoints(
        self,
        *,
        delays_fs: Sequence[float] | None = None,
        include_probe_only: bool | None = None,
        output_dir: str | Path | None = None,
        output_preview: bool | None = None,
    ) -> dict[str, str]:
        """Export selected raw DynamicsResult cases from checkpoints using core IO."""

        if save_result_case is None:
            raise RuntimeError("qudpy_sjh.utils.io.save_result_case is unavailable.")
        if not self.checkpoint.enabled:
            raise RuntimeError("save_preview_from_checkpoints requires checkpoint.enabled=True.")

        preview_root = Path(output_dir) if output_dir is not None else self.output_dir / self.io.preview_case_dir_name
        preview = self.io.save_case_previews if output_preview is None else bool(output_preview)

        paths: dict[str, str] = {}
        for key in self.preview_case_keys(delays_fs=delays_fs, include_probe_only=include_probe_only):
            dyn = self.load_case_from_checkpoint(
                key,
                require_existing=bool(self.checkpoint.require_existing_for_preview),
            )
            written = save_result_case(
                dyn,
                preview_root,
                output_data=True,
                output_preview=preview,
                case_name=key,
                example_name=self.settings.experiment_name,
                condition_name="ta_delay_scan",
                append_results_csv=bool(self.io.append_preview_results_csv),
            )
            paths[key] = str(written.get("case_dir", preview_root / key))
        return paths


__all__ = [
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
]
