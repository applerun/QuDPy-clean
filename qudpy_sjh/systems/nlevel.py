"""Basic N-level matter-system makers."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .core import NLevelSystem


def _finite_float(name: str, value: float) -> float:
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"{name} must be finite. Got {value!r}.")
    return number


def _nonnegative_finite_float(name: str, value: float) -> float:
    number = _finite_float(name, value)
    if number < 0.0:
        raise ValueError(f"{name} must be >= 0. Got {value!r}.")
    return number


def _ground_density_matrix(dimension: int) -> np.ndarray:
    density = np.zeros((dimension, dimension), dtype=float)
    density[0, 0] = 1.0
    return density


def _resolve_initial_state(initial_state: str | np.ndarray | None, *, dimension: int) -> np.ndarray | None:
    if initial_state is None:
        return None
    if isinstance(initial_state, str):
        label = initial_state.strip().lower()
        if label != "ground":
            raise ValueError(f"initial_state string must be 'ground'. Got {initial_state!r}.")
        return _ground_density_matrix(dimension)
    return np.asarray(initial_state)


def _merged_metadata(metadata: Mapping[str, Any] | None, **generated: Any) -> dict[str, Any]:
    payload = dict(metadata or {})
    payload.update(generated)
    return payload


def make_two_level_system(
    *,
    energy_eV: float,
    mu_D: float,
    gamma_fs_inv: float | None = None,
    ground_label: str = "0",
    excited_label: str = "X",
    initial_state: str | np.ndarray | None = "ground",
    name: str = "two_level",
    metadata: Mapping[str, Any] | None = None,
) -> NLevelSystem:
    """生成 two-level matter system；不自动生成 population relaxation。"""

    energy = _finite_float("energy_eV", energy_eV)
    mu = _finite_float("mu_D", mu_D)
    gamma = None if gamma_fs_inv is None else _nonnegative_finite_float("gamma_fs_inv", gamma_fs_inv)
    basis = (str(ground_label), str(excited_label))
    dipoles = np.zeros((2, 2), dtype=float)
    dipoles[0, 1] = mu
    dipoles[1, 0] = mu
    transition_dephasing = {}
    if gamma is not None:
        transition_dephasing[(basis[0], basis[1])] = gamma

    return NLevelSystem(
        name=name,
        basis=basis,
        energies_eV=np.asarray([0.0, energy], dtype=float),
        dipole_matrix_D=dipoles,
        initial_state=_resolve_initial_state(initial_state, dimension=2),
        transition_dephasing_fs_inv=transition_dephasing,
        metadata=_merged_metadata(
            metadata,
            generator="make_two_level_system",
            energy_eV=energy,
            mu_D=mu,
            gamma_fs_inv=gamma,
            dephasing_interpretation="transition_dephasing_fs_inv",
            no_population_relaxation_generated=True,
        ),
    )


def make_three_level_ladder_system(
    *,
    energy_01_eV: float,
    mu01_D: float,
    energy_12_eV: float | None = None,
    mu12_D: float | None = None,
    gamma01_fs_inv: float | None = None,
    gamma12_fs_inv: float | None = None,
    labels: tuple[str, str, str] = ("0", "X", "XX"),
    initial_state: str | np.ndarray | None = "ground",
    name: str = "three_level_ladder",
    metadata: Mapping[str, Any] | None = None,
) -> NLevelSystem:
    """生成三能级 ladder；只包含相邻跃迁偶极和 transition dephasing。"""

    if len(labels) != 3:
        raise ValueError(f"labels must contain exactly 3 items. Got {labels!r}.")
    energy01 = _finite_float("energy_01_eV", energy_01_eV)
    mu01 = _finite_float("mu01_D", mu01_D)
    energy12 = energy01 if energy_12_eV is None else _finite_float("energy_12_eV", energy_12_eV)
    mu12 = np.sqrt(2.0) * mu01 if mu12_D is None else _finite_float("mu12_D", mu12_D)
    gamma01 = None if gamma01_fs_inv is None else _nonnegative_finite_float("gamma01_fs_inv", gamma01_fs_inv)
    gamma12 = None if gamma12_fs_inv is None else _nonnegative_finite_float("gamma12_fs_inv", gamma12_fs_inv)

    basis = tuple(str(item) for item in labels)
    dipoles = np.zeros((3, 3), dtype=float)
    dipoles[0, 1] = mu01
    dipoles[1, 0] = mu01
    dipoles[1, 2] = mu12
    dipoles[2, 1] = mu12
    transition_dephasing = {}
    if gamma01 is not None:
        transition_dephasing[(basis[0], basis[1])] = gamma01
    if gamma12 is not None:
        transition_dephasing[(basis[1], basis[2])] = gamma12

    return NLevelSystem(
        name=name,
        basis=basis,
        energies_eV=np.asarray([0.0, energy01, energy01 + energy12], dtype=float),
        dipole_matrix_D=dipoles,
        initial_state=_resolve_initial_state(initial_state, dimension=3),
        transition_dephasing_fs_inv=transition_dephasing,
        metadata=_merged_metadata(
            metadata,
            generator="make_three_level_ladder_system",
            energy_01_eV=energy01,
            energy_12_eV=energy12,
            mu01_D=mu01,
            mu12_D=mu12,
            gamma01_fs_inv=gamma01,
            gamma12_fs_inv=gamma12,
            dephasing_interpretation="transition_dephasing_fs_inv",
            no_population_relaxation_generated=True,
        ),
    )


__all__ = [
    "make_two_level_system",
    "make_three_level_ladder_system",
]
