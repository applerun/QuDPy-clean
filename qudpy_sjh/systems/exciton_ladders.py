"""Single-exciton ladder matter-system makers."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from .core import NLevelSystem
from .nlevel import _finite_float, _nonnegative_finite_float, _resolve_initial_state


def _is_sequence(value: Any) -> bool:
    if isinstance(value, (str, bytes)):
        return False
    if np.isscalar(value):
        return False
    return isinstance(value, Sequence) or isinstance(value, np.ndarray)


def _is_default_higher_order_value(value: float | Sequence[float], *, default: float) -> bool:
    if _is_sequence(value):
        arr = np.asarray(value, dtype=float).ravel()
        return bool(arr.size == 0 or np.allclose(arr, float(default), rtol=0.0, atol=1.0e-15))
    return bool(np.isclose(float(value), float(default), rtol=0.0, atol=1.0e-15))


def _normalize_higher_order_values(
    name: str,
    value: float | Sequence[float],
    *,
    n_quantum: int,
    default: float,
) -> np.ndarray:
    expected_len = int(n_quantum) - 1
    if expected_len < 0:
        raise ValueError(f"n_quantum must be >= 1. Got {n_quantum}.")

    if expected_len == 0:
        if not _is_default_higher_order_value(value, default=default):
            raise ValueError(
                f"{name} has no higher-order meaning for n_quantum=1; "
                f"expected default-equivalent value {default!r}."
            )
        return np.asarray([], dtype=float)

    if n_quantum == 2 and not _is_sequence(value):
        arr = np.asarray([float(value)], dtype=float)
    else:
        if not _is_sequence(value):
            raise ValueError(
                f"{name} must be a sequence for n_quantum={n_quantum}; "
                f"expected length {expected_len}, got scalar."
            )
        arr = np.asarray(value, dtype=float).ravel()

    if arr.size != expected_len:
        raise ValueError(
            f"{name} length mismatch for n_quantum={n_quantum}; "
            f"expected length {expected_len}, got {arr.size}."
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values.")
    return arr


def _default_exciton_basis_labels(n_quantum: int) -> tuple[str, ...]:
    labels = ["0"]
    for quantum in range(1, int(n_quantum) + 1):
        if quantum == 1:
            labels.append("X")
        elif quantum == 2:
            labels.append("XX")
        elif quantum == 3:
            labels.append("XXX")
        else:
            labels.append(f"{quantum}X")
    return tuple(labels)


def _transition_records(
    *,
    basis: tuple[str, ...],
    values: Sequence[float],
    value_key: str,
) -> list[dict[str, Any]]:
    records = []
    for index, value in enumerate(values, start=1):
        records.append(
            {
                "from": basis[index - 1],
                "to": basis[index],
                value_key: float(value),
            }
        )
    return records


def make_single_exciton_ladder_system(
    *,
    n_quantum: int = 2,
    energy_1q_eV: float,
    mu_1q_D: float,
    gamma_1q_fs_inv: float | None = None,
    eis_eV: float | Sequence[float] = 0.0,
    pb: float | Sequence[float] = 1.0,
    eid: float | Sequence[float] = 1.0,
    initial_state: str | np.ndarray | None = "ground",
    name: str = "single_exciton_ladder",
    metadata: Mapping[str, Any] | None = None,
) -> NLevelSystem:
    """生成单激子 ladder matter system。

    EIS 修正相邻 transition energy；PB 是相对于 harmonic oscillator
    dipole 的无量纲因子；EID 只缩放 transition dephasing，不表示
    population relaxation。
    """

    n = int(n_quantum)
    if n < 1:
        raise ValueError(f"n_quantum must be >= 1. Got {n_quantum}.")

    energy_1q = _finite_float("energy_1q_eV", energy_1q_eV)
    mu_1q = _finite_float("mu_1q_D", mu_1q_D)
    gamma_1q = None if gamma_1q_fs_inv is None else _nonnegative_finite_float("gamma_1q_fs_inv", gamma_1q_fs_inv)
    normalized_eis = _normalize_higher_order_values("eis_eV", eis_eV, n_quantum=n, default=0.0)
    normalized_pb = _normalize_higher_order_values("pb", pb, n_quantum=n, default=1.0)
    normalized_eid = _normalize_higher_order_values("eid", eid, n_quantum=n, default=1.0)

    basis = _default_exciton_basis_labels(n)
    transition_energies = [energy_1q]
    transition_dipoles = [mu_1q]
    for quantum in range(2, n + 1):
        higher_index = quantum - 2
        transition_energies.append(energy_1q + float(normalized_eis[higher_index]))
        transition_dipoles.append(float(normalized_pb[higher_index]) * np.sqrt(float(quantum)) * mu_1q)

    energies = np.zeros(n + 1, dtype=float)
    for quantum, transition_energy in enumerate(transition_energies, start=1):
        energies[quantum] = energies[quantum - 1] + float(transition_energy)

    dipoles = np.zeros((n + 1, n + 1), dtype=float)
    for quantum, transition_dipole in enumerate(transition_dipoles, start=1):
        dipoles[quantum - 1, quantum] = float(transition_dipole)
        dipoles[quantum, quantum - 1] = float(transition_dipole)

    transition_dephasing: dict[tuple[str, str], float] = {}
    transition_dephasing_values: list[float | None] = []
    if gamma_1q is not None:
        transition_dephasing_values.append(gamma_1q)
        transition_dephasing[(basis[0], basis[1])] = gamma_1q
        for quantum in range(2, n + 1):
            higher_index = quantum - 2
            gamma = float(normalized_eid[higher_index]) * gamma_1q
            transition_dephasing_values.append(gamma)
            transition_dephasing[(basis[quantum - 1], basis[quantum])] = gamma
    else:
        transition_dephasing_values = [None for _ in transition_energies]

    generated_metadata = {
        "generator": "make_single_exciton_ladder_system",
        "n_quantum": n,
        "basis_label_rule": "0, X, XX, XXX, then {q}X for q > 3",
        "energy_1q_eV": energy_1q,
        "mu_1q_D": mu_1q,
        "gamma_1q_fs_inv": gamma_1q,
        "normalized_eis_eV": normalized_eis.tolist(),
        "normalized_pb": normalized_pb.tolist(),
        "normalized_eid": normalized_eid.tolist(),
        "transition_energies_eV": _transition_records(
            basis=basis,
            values=transition_energies,
            value_key="energy_eV",
        ),
        "transition_dipoles_D": _transition_records(
            basis=basis,
            values=transition_dipoles,
            value_key="mu_D",
        ),
        "transition_dephasing_fs_inv": [
            {
                "from": basis[index - 1],
                "to": basis[index],
                "gamma_fs_inv": None if gamma is None else float(gamma),
            }
            for index, gamma in enumerate(transition_dephasing_values, start=1)
        ],
        "eis_interpretation": "higher transition energy shift: DeltaE_q = energy_1q_eV + eis_eV[q-2]",
        "pb_interpretation": "dimensionless factor relative to harmonic oscillator dipole sqrt(q) * mu_1q_D",
        "eid_interpretation": "relative_to_1q_transition_dephasing",
        "no_population_relaxation_generated": True,
    }
    merged_metadata = dict(metadata or {})
    merged_metadata.update(generated_metadata)

    return NLevelSystem(
        name=name,
        basis=basis,
        energies_eV=energies,
        dipole_matrix_D=dipoles,
        initial_state=_resolve_initial_state(initial_state, dimension=n + 1),
        transition_dephasing_fs_inv=transition_dephasing,
        metadata=merged_metadata,
    )


__all__ = [
    "make_single_exciton_ladder_system",
]
