"""Delay-scan runner for full-window transient absorption cases."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from qudpy_sjh.experiments.transient_absorption.ta_case_plan import RunCaseFn, TAFullWindowCasePlan, make_case_plan
from qudpy_sjh.experiments.transient_absorption.ta_settings import TASettings
from qudpy_sjh.utils.core.normalization import ParaNormalizer
from qudpy_sjh.utils.core.results import DynamicsResult
from qudpy_sjh.utils.io import save_result_case


@dataclass(frozen=True)
class TADelayScanCase:
    plan: TAFullWindowCasePlan
    result: DynamicsResult
    saved_files: dict | None = None


def run_delay_scan(
    settings: TASettings,
    *,
    normalizer: ParaNormalizer | None = None,
    run_case_fn: RunCaseFn | None = None,
    save_output_dir: str | Path | None = None,
    save_case_fn: Callable[..., dict] = save_result_case,
) -> list[TADelayScanCase]:
    cases: list[TADelayScanCase] = []
    for delay_fs in settings.delays_fs:
        plan = make_case_plan(settings, delay_fs)
        execute_kwargs = {"normalizer": normalizer}
        if run_case_fn is not None:
            execute_kwargs["run_case_fn"] = run_case_fn
        result = plan.execute(**execute_kwargs)
        written = None
        if save_output_dir is not None:
            written = save_case_fn(
                result,
                save_output_dir,
                case_name=plan.case_name,
                condition_name=f"delay_{delay_fs:g}_fs",
            )
        cases.append(TADelayScanCase(plan=plan, result=result, saved_files=written))
    return cases


__all__ = ["TADelayScanCase", "run_delay_scan"]

