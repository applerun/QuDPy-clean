"""基于 QuTiP mesolve 的单轨迹求解流程。"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
from qutip import Qobj, mesolve

from sjh_learn.utils.checks import evaluate_sanity_checks
from sjh_learn.utils.core.config import ensure_rwa_enabled
from sjh_learn.utils.fields import FieldPhyRoot
from sjh_learn.utils.core.model import build_c_ops, build_lab_hamiltonian, build_rwa_hamiltonian, initial_density_matrix, parameter_field
from sjh_learn.utils.core.normalization import ParaNormalizer
from sjh_learn.utils.core.parameters import NLevelPhysicalParams, NLevelSolverParams, SolverParams
from sjh_learn.utils.core.results import DynamicsResult


def _default_tlist(parameters: NLevelSolverParams) -> np.ndarray:
    if parameters.tlist is not None:
        return np.asarray(parameters.tlist, dtype=float)
    t_end = parameters.t_final if parameters.t_end is None else parameters.t_end
    return np.arange(parameters.t_start, t_end + 0.5 * parameters.dt, parameters.dt)


def _mesolve_options(parameters: NLevelSolverParams) -> dict[str, float]:
    return {"max_step": float(parameters.dt)}


def _rho0(parameters: NLevelSolverParams, rho0: Qobj | None) -> Qobj:
    return initial_density_matrix(len(parameters.energies)) if rho0 is None else rho0


def _basic_sanity_checks(result: DynamicsResult) -> dict[str, object]:
    return {
        "trace_error_small": {
            "value": result.max_trace_error(),
            "threshold": 1e-8,
            "passed": bool(result.max_trace_error() < 1e-8),
        },
        "hermiticity_error_small": {
            "value": result.max_hermiticity_error(),
            "threshold": 1e-8,
            "passed": bool(result.max_hermiticity_error() < 1e-8),
        },
    }


def _run_lab_case(
    parameters: NLevelSolverParams,
    rho0: Qobj | None = None,
    *,
    physical_params: NLevelPhysicalParams | None = None,
    solver_params: SolverParams | None = None,
) -> DynamicsResult:
    times = _default_tlist(parameters)
    field = parameter_field(parameters)
    solver_result = mesolve(
        H=build_lab_hamiltonian(parameters),
        rho0=_rho0(parameters, rho0),
        tlist=times,
        c_ops=build_c_ops(parameters),
        e_ops=[],
        args={"field": field},
        options=_mesolve_options(parameters),
    )
    states = list(solver_result.states)
    drive = field
    result = DynamicsResult(
        mode="lab_exact",
        times=times,
        times_fs=parameters.times_fs,
        states=states,
        parameters=parameters,
        physical_params=physical_params,
        solver_params=solver_params,
        metadata={"energies_code": parameters.energies},
        drive=drive,
        drive_dict=drive.to_dict() if drive is not None and hasattr(drive, "to_dict") else None,
        drive_expr=drive.to_expr() if drive is not None and hasattr(drive, "to_expr") else None,
        drive_name=getattr(drive, "name", None),
    )
    result.sanity_checks = evaluate_sanity_checks(result)
    return result


def _run_rwa_case(
    parameters: NLevelSolverParams,
    rho0: Qobj | None = None,
    drive: object | None = None,
) -> DynamicsResult:
    ensure_rwa_enabled()
    raise RuntimeError(
        "Legacy RWA solver-unit drive classes have been removed. "
        "RWA is disabled by default; use lab_exact with physical FieldPhyRoot input."
    )


def _rotating_frame_unitary(time: float, omega_drive: float) -> Qobj:
    return Qobj(np.diag([1.0, np.exp(-1j * omega_drive * time)]).astype(np.complex128))


def _rotate_density_trajectory(times: np.ndarray, states: list[Qobj], omega_drive: float) -> list[Qobj]:
    rotated_states: list[Qobj] = []
    for time, rho_lab in zip(times, states):
        unitary = _rotating_frame_unitary(time, omega_drive)
        rotated_states.append(unitary.dag() * rho_lab * unitary)
    return rotated_states


def _bound_physical_field(
    physical: NLevelPhysicalParams,
    normalizer: ParaNormalizer,
    solver: SolverParams,
):
    if not isinstance(physical.field, FieldPhyRoot):
        raise TypeError("NLevelPhysicalParams.field must be a FieldPhyRoot instance.")
    return normalizer.make_code_field(physical.field, solver)


def make_rotating_view(lab_result: DynamicsResult) -> DynamicsResult:
    if lab_result.mode != "lab_exact":
        raise ValueError("make_rotating_view expects a lab_exact DynamicsResult.")
    if lab_result.dimension() != 2:
        raise ValueError("rotating_view 当前只用于 N=2 lab_exact 后处理。")
    states = _rotate_density_trajectory(
        np.asarray(lab_result.times, dtype=float),
        lab_result.states,
        lab_result.parameters.omega_drive,
    )
    result = DynamicsResult(
        mode="rotating_view",
        times=lab_result.times,
        times_fs=lab_result.times_fs,
        states=states,
        parameters=lab_result.parameters,
        physical_params=lab_result.physical_params,
        solver_params=lab_result.solver_params,
        metadata=dict(lab_result.metadata),
        source_mode=lab_result.mode,
    )
    result.sanity_checks = _basic_sanity_checks(result)
    return result


def _optical_codeparams_from_solverparams(
    solver: SolverParams,
    physical: NLevelPhysicalParams | None = None,
    normalizer: ParaNormalizer | None = None,
) -> NLevelSolverParams:
    if normalizer is not None:
        times_fs = normalizer.denormalize_time_array(solver.tlist, solver)
    elif physical is not None:
        times_fs = np.linspace(physical.t_start_fs, physical.t_end_fs, len(solver.tlist))
    else:
        times_fs = None

    field = None
    if physical is not None and normalizer is not None:
        field = _bound_physical_field(physical, normalizer, solver)

    return NLevelSolverParams(
        t_start=solver.t_start,
        t_end=solver.t_end,
        dt=solver.dt,
        t_final=solver.t_end,
        hbar=1.0,
        energies=tuple(float(value) for value in solver.energies_code),
        dipole_matrix=tuple(tuple(complex(item) for item in row) for row in solver.coupling_matrix_code),
        coupling_matrix=tuple(tuple(complex(item) for item in row) for row in solver.coupling_matrix_code),
        omega_drive=0.0 if solver.omega_L is None else solver.omega_L,
        relaxation_channels=solver.relaxation_channels_code,
        pure_dephasing_channels=solver.pure_dephasing_channels_code,
        detuning=0.0 if solver.detuning is None else solver.detuning,
        pulse_center=solver.pulse_center,
        pulse_sigma=solver.pulse_sigma,
        field=field,
        tlist=solver.tlist,
        times_fs=times_fs,
        basis=None if physical is None else physical.basis,
    )



def run_case(
    physical_params: NLevelPhysicalParams,
    normalizer: ParaNormalizer | None = None,
    rho0: Qobj | None = None,
    *,
    load_ckp: str | Path | None = None,
    save_ckp: str | Path | None = None,
    force_run: bool = False,
) -> DynamicsResult:
    load_path = None if load_ckp is None else Path(load_ckp)
    if load_path is not None and load_path.exists() and not force_run:
        print(f"Loading checkpoint: {load_path}")
        return DynamicsResult.from_ckp(load_path)
    if load_path is not None and force_run:
        print(f"force_run=True, running simulation and ignoring checkpoint: {load_path}")
    elif load_path is not None:
        print(f"Checkpoint not found, running simulation: {load_path}")

    if physical_params.solver_mode == "rwa":
        ensure_rwa_enabled()
        raise ValueError(
            "RWA mode is legacy and has not been migrated to field-only NLevelPhysicalParams input."
        )
    local_normalizer = ParaNormalizer() if normalizer is None else normalizer
    solver = local_normalizer.normalize(physical_params)
    parameters = _optical_codeparams_from_solverparams(solver=solver, physical=physical_params, normalizer=local_normalizer)
    if physical_params.solver_mode == "lab_exact":
        result = _run_lab_case(parameters, rho0=rho0, physical_params=physical_params, solver_params=solver)
    if physical_params.solver_mode == "rwa":
        ensure_rwa_enabled()
        result = _run_rwa_case(parameters, rho0=rho0)
        result.physical_params = physical_params
        result.solver_params = solver
    if physical_params.solver_mode not in {"lab_exact", "rwa"}:
        raise ValueError(f"Unsupported solver_mode: {physical_params.solver_mode!r}. Expected 'lab_exact' or 'rwa'.")

    checkpoint_save_path = save_ckp if save_ckp is not None else load_ckp
    if checkpoint_save_path is not None:
        save_path = Path(checkpoint_save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Saving checkpoint: {save_path}")
        result.save_ckp(save_path)
    return result


def run_cases(
    physical_params_list: Iterable[NLevelPhysicalParams],
    normalizer: ParaNormalizer | None = None,
) -> list[DynamicsResult]:
    return [run_case(physical_params, normalizer=normalizer) for physical_params in physical_params_list]


__all__ = [
    "run_case",
    "run_cases",
    "make_rotating_view",
]
