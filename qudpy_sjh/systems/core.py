"""Matter-side N-level system definitions.

本模块只描述物质系统本身：basis、能级、跃迁偶极、初态、
transition dephasing 以及用户手动追加的 dissipation channels。
它不生成 field、pulse sequence、readout、checkpoint 或 experiment params。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field as dataclass_field, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


def _json_safe(value: Any) -> Any:
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _json_safe(value.to_dict())
    if hasattr(value, "__dataclass_fields__"):
        return _json_safe(asdict(value))
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, complex):
        return {"real": float(value.real), "imag": float(value.imag)}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return {
        "type": f"{value.__class__.__module__}.{value.__class__.__name__}",
    }


def _normalize_basis(basis: Sequence[str]) -> tuple[str, ...]:
    if not basis:
        raise ValueError("basis must be non-empty.")
    labels = tuple(str(item).strip() for item in basis)
    blank = [index for index, label in enumerate(labels) if not label]
    if blank:
        raise ValueError(f"basis contains blank label at indices: {blank}.")
    if len(set(labels)) != len(labels):
        raise ValueError(f"basis labels must be unique. Got {labels}.")
    return labels


def _normalize_transition_dephasing(
    values: Mapping[tuple[str, str], float] | None,
    *,
    basis: tuple[str, ...],
) -> dict[tuple[str, str], float]:
    if values is None:
        return {}

    known = set(basis)
    normalized: dict[tuple[str, str], float] = {}
    for key, rate in values.items():
        if not isinstance(key, tuple) or len(key) != 2:
            raise TypeError("transition_dephasing_fs_inv keys must be (label_i, label_j) tuples.")
        left = str(key[0]).strip()
        right = str(key[1]).strip()
        if left not in known or right not in known:
            raise ValueError(
                "transition_dephasing_fs_inv labels must exist in basis. "
                f"Got ({left!r}, {right!r}) for basis={basis}."
            )
        if left == right:
            raise ValueError("transition_dephasing_fs_inv requires distinct labels; got i == j.")
        gamma = float(rate)
        if not np.isfinite(gamma) or gamma < 0.0:
            raise ValueError(f"transition_dephasing_fs_inv rate must be finite and >= 0. Got {rate!r}.")
        normalized[(left, right)] = gamma
    return normalized


def _normalize_dissipation(dissipation: Any | Sequence[Any] | None) -> tuple[Any, ...]:
    if dissipation is None:
        return ()
    if isinstance(dissipation, tuple):
        return dissipation
    if isinstance(dissipation, list):
        return tuple(dissipation)
    return (dissipation,)


@dataclass(frozen=True)
class NLevelSystem:
    """N-level matter system definition.

    这里的 dephasing 是 transition-level 描述；population relaxation 等
    dissipation channels 需要用户通过 with_dissipation/append_dissipation
    显式追加。
    """

    name: str
    basis: tuple[str, ...]
    energies_eV: np.ndarray
    dipole_matrix_D: np.ndarray
    initial_state: np.ndarray | None = None
    transition_dephasing_fs_inv: dict[tuple[str, str], float] = dataclass_field(default_factory=dict)
    dissipation: tuple[Any, ...] = dataclass_field(default_factory=tuple)
    metadata: dict[str, Any] = dataclass_field(default_factory=dict)

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        if not name:
            raise ValueError("name must be non-empty.")

        basis = _normalize_basis(self.basis)
        dimension = len(basis)

        energies = np.asarray(self.energies_eV, dtype=float)
        if energies.ndim != 1:
            raise ValueError(f"energies_eV must be a 1D array. Got shape {energies.shape}.")
        if energies.shape != (dimension,):
            raise ValueError(
                "energies_eV length must match dimension. "
                f"Got length={energies.size}, dimension={dimension}."
            )
        if not np.all(np.isfinite(energies)):
            raise ValueError("energies_eV must contain only finite values.")

        dipoles = np.asarray(self.dipole_matrix_D, dtype=float)
        if dipoles.ndim != 2:
            raise ValueError(f"dipole_matrix_D must be a 2D array. Got shape {dipoles.shape}.")
        if dipoles.shape != (dimension, dimension):
            raise ValueError(
                "dipole_matrix_D shape must be (dimension, dimension). "
                f"Got shape={dipoles.shape}, dimension={dimension}."
            )
        if not np.all(np.isfinite(dipoles)):
            raise ValueError("dipole_matrix_D must contain only finite values.")
        if not np.allclose(dipoles, dipoles.T, rtol=1.0e-12, atol=1.0e-12):
            raise ValueError(
                "dipole_matrix_D must be symmetric in this first systems implementation. "
                "Pass an explicitly symmetrized matrix to avoid hidden direction-convention mistakes."
            )

        initial = None
        if self.initial_state is not None:
            initial = np.asarray(self.initial_state)
            if initial.shape not in ((dimension,), (dimension, dimension)):
                raise ValueError(
                    "initial_state must be None, a state vector with shape (dimension,), "
                    f"or a density matrix with shape (dimension, dimension). Got {initial.shape}."
                )
            if not np.all(np.isfinite(initial)):
                raise ValueError("initial_state must contain only finite values.")

        transition_dephasing = _normalize_transition_dephasing(
            self.transition_dephasing_fs_inv,
            basis=basis,
        )
        dissipation = _normalize_dissipation(self.dissipation)
        metadata = dict(self.metadata)

        object.__setattr__(self, "name", name)
        object.__setattr__(self, "basis", basis)
        object.__setattr__(self, "energies_eV", energies)
        object.__setattr__(self, "dipole_matrix_D", dipoles)
        object.__setattr__(self, "initial_state", initial)
        object.__setattr__(self, "transition_dephasing_fs_inv", transition_dephasing)
        object.__setattr__(self, "dissipation", dissipation)
        object.__setattr__(self, "metadata", metadata)

    @property
    def dimension(self) -> int:
        return len(self.basis)

    def with_dissipation(self, dissipation: Any | Sequence[Any] | None) -> "NLevelSystem":
        return replace(self, dissipation=_normalize_dissipation(dissipation))

    def append_dissipation(self, *channels: Any) -> "NLevelSystem":
        return replace(self, dissipation=tuple(self.dissipation) + tuple(channels))

    def with_transition_dephasing(
        self,
        transition_dephasing_fs_inv: Mapping[tuple[str, str], float] | None,
    ) -> "NLevelSystem":
        return replace(self, transition_dephasing_fs_inv=dict(transition_dephasing_fs_inv or {}))

    def to_dict(self, *, include_arrays: bool = True) -> dict[str, Any]:
        energy_min = float(np.min(self.energies_eV))
        energy_max = float(np.max(self.energies_eV))
        nonzero_dipoles = int(np.count_nonzero(np.abs(self.dipole_matrix_D) > 0.0))
        summary: dict[str, Any] = {
            "class": self.__class__.__name__,
            "name": self.name,
            "dimension": self.dimension,
            "basis": list(self.basis),
            "energy_range_eV": [energy_min, energy_max],
            "n_nonzero_dipoles": nonzero_dipoles,
            "n_transition_dephasing": len(self.transition_dephasing_fs_inv),
            "n_dissipation_channels": len(self.dissipation),
            "metadata": _json_safe(self.metadata),
        }
        if not include_arrays:
            return summary

        payload = dict(summary)
        payload.update(
            {
                "energies_eV": self.energies_eV.tolist(),
                "dipole_matrix_D": self.dipole_matrix_D.tolist(),
                "initial_state": None if self.initial_state is None else _json_safe(self.initial_state),
                "transition_dephasing_fs_inv": [
                    {
                        "from": left,
                        "to": right,
                        "gamma_fs_inv": gamma,
                    }
                    for (left, right), gamma in self.transition_dephasing_fs_inv.items()
                ],
                "dissipation": [_json_safe(item) for item in self.dissipation],
            }
        )
        return payload
