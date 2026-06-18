"""N-level Hamiltonian 和 Lindblad collapse operator 构造。"""

from __future__ import annotations

import numpy as np
from qutip import Qobj, basis

from typing import Any

from sjh_learn.utils.core.parameters import NLevelSolverParams, as_complex_matrix


def electric_field(times: np.ndarray, amplitude: float, omega_drive: float) -> np.ndarray:
    t = np.asarray(times, dtype=float)
    return 2.0 * float(amplitude) * np.cos(float(omega_drive) * t)


def compute_detuning(epsilon_1: float, epsilon_2: float, omega_drive: float, hbar: float) -> float:
    return (epsilon_2 - epsilon_1) - hbar * omega_drive


def compute_energy_gap(detuning: float, omega_drive: float, hbar: float) -> float:
    return hbar * omega_drive + detuning


def dimension(parameters: NLevelSolverParams) -> int:
    return len(parameters.energies)


def _basis_operators() -> tuple[Qobj, Qobj, Qobj]:
    ket_0 = basis(2, 0)
    ket_1 = basis(2, 1)
    projector_0 = ket_0 * ket_0.dag()
    projector_1 = ket_1 * ket_1.dag()
    dipole_operator = ket_0 * ket_1.dag() + ket_1 * ket_0.dag()
    return projector_0, projector_1, dipole_operator


def sigma_minus_operator() -> Qobj:
    return basis(2, 0) * basis(2, 1).dag()


def sigma_z_operator() -> Qobj:
    projector_0, projector_1, _dipole = _basis_operators()
    return projector_0 - projector_1


def initial_density_matrix(n_levels: int = 2, occupied_level: int = 0) -> Qobj:
    ket = basis(n_levels, occupied_level)
    return ket * ket.dag()


def excited_density_matrix() -> Qobj:
    return basis(2, 1) * basis(2, 1).dag()


def coherent_superposition_density_matrix() -> Qobj:
    psi = (basis(2, 0) + basis(2, 1)).unit()
    return psi * psi.dag()


def parameter_field(parameters: NLevelSolverParams):
    if parameters.field is None:
        raise ValueError(
            "NLevelSolverParams.field is required. Construct physical FieldPhyRoot input "
            "and convert it with ParaNormalizer.make_code_field()."
        )
    return parameters.field


def pulse_envelope(time: float, pulse_center: float | None, pulse_sigma: float | None) -> float:
    if pulse_sigma is None:
        return 1.0
    center = 0.0 if pulse_center is None else float(pulse_center)
    sigma = float(pulse_sigma)
    if sigma <= 0:
        raise ValueError("pulse_sigma must be positive.")
    return float(np.exp(-((float(time) - center) ** 2) / (2.0 * sigma**2)))


def build_static_hamiltonian(parameters: NLevelSolverParams) -> Qobj:
    h0 = np.diag(np.asarray(parameters.energies, dtype=np.complex128))
    _require_hermitian_matrix(h0, "field-free Hamiltonian")
    return Qobj(h0)


def _require_hermitian_matrix(matrix: np.ndarray, name: str) -> None:
    array = np.asarray(matrix, dtype=np.complex128)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError(f"{name} must be square.")
    if not np.allclose(array, array.conj().T, rtol=1e-10, atol=1e-12):
        raise ValueError(f"{name} must be Hermitian.")


def _field_value_from_args(time: float, args: dict[str, Any]) -> float:
    field = args.get("field")
    if field is None:
        raise ValueError(
            'lab_exact Hamiltonian requires args["field"]. Use physical FieldPhyRoot input '
            "and ParaNormalizer.make_code_field()."
        )
    return float(field(float(time)))


def build_lab_hamiltonian(parameters: NLevelSolverParams) -> list[Qobj | list[object]]:
    h0 = build_static_hamiltonian(parameters)
    dipole_matrix = as_complex_matrix(parameters.dipole_matrix)
    _require_hermitian_matrix(dipole_matrix, "lab-frame dipole interaction matrix")
    dipole_operator = Qobj(dipole_matrix)
    # H_int(t) = -E_code(t) * mu_code_matrix。这里所有量已经是 solver code unit。
    return [
        h0,
        [
            -dipole_operator,
            lambda t, args: _field_value_from_args(float(t), args),
        ],
    ]


def build_rwa_hamiltonian(parameters: NLevelSolverParams) -> list[Qobj | list[object]]:
    energies = np.asarray(parameters.energies, dtype=float)
    n_levels = len(energies)
    shifted = energies - energies[0]
    if n_levels >= 2:
        shifted[1] = shifted[1] - parameters.omega_drive
    h_static_matrix = np.diag(shifted.astype(np.complex128))
    _require_hermitian_matrix(h_static_matrix, "RWA static Hamiltonian")
    h_static = Qobj(h_static_matrix)
    coupling = as_complex_matrix(parameters.coupling_matrix or parameters.dipole_matrix)
    # RWA path 使用慢变量 coupling matrix；光学 carrier 已经移除。
    _require_hermitian_matrix(coupling, "RWA coupling matrix")
    diagonal = np.diag(np.diag(coupling))
    lower = np.tril(coupling, k=-1)
    h_diagonal = Qobj(-diagonal)
    h_lower = Qobj(-lower)
    h_upper = h_lower.dag()
    return [
        h_static,
        [h_diagonal, lambda t, args: float(np.real(args["drive"](t)))],
        [h_lower, lambda t, args: complex(args["drive"](t))],
        [h_upper, lambda t, args: complex(np.conjugate(args["drive"](t)))],
    ]


def _collapse_projector(n_levels: int, level: int) -> Qobj:
    ket = basis(n_levels, level)
    return ket * ket.dag()


def build_c_ops(parameters: NLevelSolverParams) -> list[Qobj]:
    n_levels = dimension(parameters)
    c_ops: list[Qobj] = []
    for channel in parameters.relaxation_channels:
        rate = float(channel.get("rate_code", channel.get("rate", 0.0)))
        if rate <= 0:
            continue
        from_level = int(channel["from_level"])
        to_level = int(channel["to_level"])
        c_ops.append(np.sqrt(rate) * (basis(n_levels, to_level) * basis(n_levels, from_level).dag()))
    for channel in parameters.pure_dephasing_channels:
        rate = float(channel.get("rate_code", channel.get("rate", 0.0)))
        if rate <= 0:
            continue
        level = int(channel["level"])
        c_ops.append(np.sqrt(rate) * _collapse_projector(n_levels, level))
    return c_ops


__all__ = [
    "electric_field",
    "compute_detuning",
    "compute_energy_gap",
    "_basis_operators",
    "sigma_minus_operator",
    "sigma_z_operator",
    "initial_density_matrix",
    "excited_density_matrix",
    "coherent_superposition_density_matrix",
    "parameter_field",
    "pulse_envelope",
    "build_lab_hamiltonian",
    "build_rwa_hamiltonian",
    "build_c_ops",
]
