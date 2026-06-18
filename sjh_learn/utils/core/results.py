"""Single-trajectory quantum dynamics result container."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields as dataclass_fields, is_dataclass
from pathlib import Path
import pickle
from typing import Any

import numpy as np
from qutip import Qobj

from sjh_learn.utils.core.parameters import NLevelPhysicalParams, SolverParams


def _require_pandas():
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("pandas is required for dataframe export. Install pandas in this environment.") from exc
    return pd


def _complex_matrix_to_json(value: Any) -> list:
    array = np.asarray(value, dtype=np.complex128)
    return [[{"real": float(item.real), "imag": float(item.imag)} for item in row] for row in array]


def _json_safe(value: Any) -> Any:
    if type(value).__name__ == "ParaNormalizer":
        return {"class": "ParaNormalizer", "note": "runtime object omitted from JSON metadata"}
    if type(value).__name__ == "NLevelPhysicalParams":
        payload = {
            item.name: getattr(value, item.name)
            for item in dataclass_fields(value)
            if item.name != "field"
        }
        field_value = getattr(value, "field", None)
        payload["field"] = None if field_value is None else _json_safe(field_value)
        return _json_safe(payload)
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _json_safe(value.to_dict())
    if is_dataclass(value):
        return _json_safe({item.name: getattr(value, item.name) for item in dataclass_fields(value)})
    if isinstance(value, Qobj):
        return {"qobj_shape": list(value.shape), "data": _complex_matrix_to_json(value.full())}
    if isinstance(value, complex):
        return {"real": float(value.real), "imag": float(value.imag)}
    if isinstance(value, np.ndarray):
        if np.iscomplexobj(value):
            return _json_safe(value.tolist())
        return value.tolist()
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if callable(value):
        return {"callable_serialized": False, "repr": repr(value)}
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _time_axis_fs(times: np.ndarray, times_fs: np.ndarray | None) -> np.ndarray:
    if times_fs is not None:
        return np.asarray(times_fs, dtype=float)
    return np.asarray(times, dtype=float)


def _rho_label(i: int, j: int) -> str:
    return f"rho_{i}{j}"


def _upper_triangular_pairs(dimension: int) -> list[tuple[int, int]]:
    return [(i, j) for i in range(dimension) for j in range(i + 1, dimension)]


def _phase_with_mask(values: np.ndarray, *, threshold: float = 1e-8) -> tuple[np.ndarray, np.ndarray]:
    abs_values = np.abs(values)
    phase = np.angle(values).astype(float)
    phase_unwrapped = np.unwrap(np.angle(values)).astype(float)
    mask = abs_values < threshold
    phase[mask] = np.nan
    phase_unwrapped[mask] = np.nan
    return phase, phase_unwrapped


def _solver_params_fs_inv_dict(solver: SolverParams) -> dict[str, Any]:
    return {
        "time_scale_fs": solver.time_scale_fs,
        "energies_fs_inv": solver.energies_fs_inv,
        "coupling_matrix_fs_inv": solver.coupling_matrix_fs_inv,
        "relaxation_channels_fs_inv": solver.relaxation_channels_fs_inv,
        "pure_dephasing_channels_fs_inv": solver.pure_dephasing_channels_fs_inv,
        "omega_eg_fs_inv": solver.omega_eg_fs_inv,
        "omega_L_fs_inv": solver.omega_L_fs_inv,
        "detuning_fs_inv": solver.detuning_fs_inv,
        "rabi_fs_inv": solver.rabi_fs_inv,
        "gamma1_fs_inv": solver.gamma1_fs_inv,
        "gamma_phi_fs_inv": solver.gamma_phi_fs_inv,
        "gamma2_fs_inv": solver.gamma2_fs_inv,
    }


def _solver_params_code_dict(solver: SolverParams) -> dict[str, Any]:
    return {
        "energies_code": solver.energies_code,
        "coupling_matrix_code": solver.coupling_matrix_code,
        "relaxation_channels_code": solver.relaxation_channels_code,
        "pure_dephasing_channels_code": solver.pure_dephasing_channels_code,
        "omega_eg_code": solver.omega_eg,
        "omega_L_code": solver.omega_L,
        "detuning_code": solver.detuning,
        "rabi_code": solver.rabi,
        "gamma1_code": solver.gamma1,
        "gamma_phi_code": solver.gamma_phi,
        "gamma2_code": solver.gamma2,
        "t_start_code": solver.t_start,
        "t_end_code": solver.t_end,
        "dt_code": solver.dt,
        "tlist_code": solver.tlist,
        "pulse_center_code": solver.pulse_center,
        "pulse_sigma_code": solver.pulse_sigma,
    }


@dataclass
class DynamicsResult:
    """One result object for one trajectory from one simulation mode or derived view."""

    mode: str
    times: np.ndarray
    times_fs: np.ndarray | None
    states: list[Qobj]
    parameters: Any
    physical_params: NLevelPhysicalParams | None = None
    solver_params: SolverParams | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    source_mode: str | None = None
    drive: Any | None = None
    drive_dict: dict[str, Any] | None = None
    drive_expr: str | None = None
    drive_name: str | None = None
    sanity_checks: dict[str, Any] = field(default_factory=dict)

    def density_array(self) -> np.ndarray:
        return np.stack([state.full() for state in self.states], axis=0)

    def save_ckp(self, file: str | Path) -> Path:
        """保存当前 `DynamicsResult` checkpoint。

        `.ckp` 文件用于分析层或后处理重新载入完整轨迹，包含 density matrix
        trajectory、时间轴、物理参数、solver 参数和 metadata。它是内部
        checkpoint / 后处理缓存，不保证跨版本长期稳定；不要把它当作替代
        `density.npz`、`components.csv`、`meta.json`、`debug_meta.json` 的归档
        格式。
        """

        path = Path(file)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return path

    @classmethod
    def from_ckp(cls, file: str | Path) -> "DynamicsResult":
        """从 `.ckp` checkpoint 文件读取 `DynamicsResult`。

        该方法只负责反序列化已经完成的模拟结果，不调用 solver，也不改变结果
        主架构。`.ckp` 使用 pickle，因此不要加载不可信来源的文件。若文件
        不存在、对象类型不对或必要字段缺失，会直接抛出清晰错误。
        """

        path = Path(file)
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open("rb") as handle:
            payload = pickle.load(handle)
        if not isinstance(payload, cls):
            raise TypeError("checkpoint does not contain a DynamicsResult.")
        required = ("mode", "times", "states", "parameters")
        for name in required:
            if not hasattr(payload, name):
                raise AttributeError(f"checkpoint DynamicsResult is missing required field: {name}")
        if payload.times is None or len(payload.times) == 0:
            raise ValueError("checkpoint DynamicsResult has an empty time axis.")
        if payload.states is None or len(payload.states) == 0:
            raise ValueError("checkpoint DynamicsResult has no states.")
        return payload

    def dimension(self) -> int:
        return int(self.density_array().shape[1])

    def populations(self) -> np.ndarray:
        density = self.density_array()
        return np.stack([density[:, index, index] for index in range(density.shape[1])], axis=1)

    def matrix_element(self, i: int, j: int) -> np.ndarray:
        return self.density_array()[:, i, j]

    def matrix_elements(self, pairs: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> dict[str, np.ndarray]:
        return {_rho_label(i, j): self.matrix_element(i, j) for i, j in pairs}

    def selected_elements(self, elements: dict[str, tuple[int, int]]) -> dict[str, np.ndarray]:
        return {label: self.matrix_element(i, j) for label, (i, j) in elements.items()}

    def drive_code_values(self, times: np.ndarray | None = None) -> np.ndarray | None:
        if self.drive is None:
            return None
        sample_times = self.times if times is None else np.asarray(times, dtype=float)
        values = np.asarray(self.drive(sample_times), dtype=float)
        if self.mode == "rwa" and self.solver_params is not None:
            return values * float(self.solver_params.rabi)
        return values

    def drive_fs_inv_values(self, times: np.ndarray | None = None) -> np.ndarray | None:
        drive_code = self.drive_code_values(times)
        if drive_code is None or self.solver_params is None or self.mode != "rwa":
            return None
        return np.asarray(drive_code, dtype=float) / float(self.solver_params.time_scale_fs)

    def field_MV_per_cm_values(
        self,
        times: np.ndarray | None = None,
        *,
        times_fs: np.ndarray | None = None,
    ) -> np.ndarray | None:
        if self.mode != "lab_exact" or self.physical_params is None or self.drive is None:
            return None
        if times is None:
            if times_fs is None:
                sample_times_code = np.asarray(self.times, dtype=float)
            else:
                if self.solver_params is None:
                    raise ValueError("solver_params is required to convert times_fs to solver code time.")
                sample_times_code = np.asarray(times_fs, dtype=float) / float(self.solver_params.time_scale_fs)
        else:
            sample_times_code = np.asarray(times, dtype=float)
        if times_fs is None:
            if times is None:
                if self.times_fs is None:
                    sample_times_fs = sample_times_code
                else:
                    sample_times_fs = np.asarray(self.times_fs, dtype=float)
                    if sample_times_fs.shape != sample_times_code.shape:
                        raise ValueError("times and times_fs shapes do not match.")
            elif self.solver_params is not None:
                sample_times_fs = sample_times_code * float(self.solver_params.time_scale_fs)
            else:
                sample_times_fs = sample_times_code
        else:
            sample_times_fs = np.asarray(times_fs, dtype=float)
            if sample_times_fs.shape != sample_times_code.shape:
                raise ValueError("times and times_fs shapes do not match.")
        if np.asarray(sample_times_fs).shape != np.asarray(sample_times_code).shape:
            raise ValueError("times and times_fs shapes do not match.")
        # 输出层必须复用 solver 实际使用的 lab-frame field callable，避免展示/分析
        # 重新拼写物理场函数后与 Hamiltonian 输入脱节。
        if hasattr(self.drive, "physical"):
            return np.asarray(self.drive.physical(sample_times_fs), dtype=float)
        reference = None
        if isinstance(self.drive_dict, dict):
            reference = self.drive_dict.get("reference_field_MV_per_cm")
        if reference is None:
            raise ValueError("lab_exact drive metadata must contain reference_field_MV_per_cm.")
        return np.asarray(self.drive(sample_times_code), dtype=float) * float(reference)

    def drive_values(self, times: np.ndarray | None = None) -> np.ndarray | None:
        return self.drive_code_values(times)

    def max_trace_error(self) -> float:
        density = self.density_array()
        traces = np.trace(density, axis1=1, axis2=2)
        return float(np.max(np.abs(traces - 1.0)))

    def max_hermiticity_error(self) -> float:
        density = self.density_array()
        return float(np.max(np.abs(density - np.conjugate(np.swapaxes(density, 1, 2)))))

    def summary_dict(self) -> dict[str, Any]:
        populations = self.populations()
        final_population_map = {
            _rho_label(index, index): float(value.real) for index, value in enumerate(populations[-1])
        }
        summary: dict[str, Any] = {
            "mode": self.mode,
            "dimension": self.dimension(),
            "final_populations": final_population_map,
            "max trace error": f"{self.max_trace_error():.3e}",
            "max Hermitian error": f"{self.max_hermiticity_error():.3e}",
            "time_range_fs": f"{_time_axis_fs(self.times, self.times_fs)[0]:.6f} -> "
            f"{_time_axis_fs(self.times, self.times_fs)[-1]:.6f}",
        }
        return summary

    def metadata_dict(self) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "result_type": "DynamicsResult",
            "mode": self.mode,
            "source_mode": self.source_mode,
            "summary": self.summary_dict(),
            "sanity_checks": self.sanity_checks,
            "result_metadata": self.metadata,
            "drive_dict": self.drive_dict,
            "drive_expr": self.drive_expr,
            "drive_name": self.drive_name,
            "density_matrix_unit": "dimensionless",
            "population_unit": "dimensionless",
            "coherence_unit": "dimensionless",
            "time_axis_saved_as": "time_fs",
            "time_axis_unit": "fs",
            "time_fs_is_physical": self.times_fs is not None,
            "n_time_points": int(len(self.times)),
            "dimension": self.dimension(),
            "component_indexing": "zero_based",
            "saved_populations": "all diagonal elements",
            "saved_coherences": "upper triangular off-diagonal elements only",
            "coherence_components": ["real", "imag", "abs", "phase", "phase_unwrapped"],
            "parameters_code": self.parameters,
        }
        if self.physical_params is not None:
            metadata["physical_params"] = self.physical_params
        if self.solver_params is not None:
            metadata["solver_params_fs_inv"] = _solver_params_fs_inv_dict(self.solver_params)
            metadata["solver_params_code"] = _solver_params_code_dict(self.solver_params)
        return _json_safe(metadata)

    def parameter_summary_dict(self) -> dict[str, Any]:
        return self.metadata_dict()

    def to_npz_dict(self) -> dict[str, np.ndarray]:
        payload = {
            "time_fs": _time_axis_fs(self.times, self.times_fs),
            "time_code": np.asarray(self.times, dtype=float),
            "density": self.density_array(),
        }
        drive_code = self.drive_code_values()
        drive_fs_inv = self.drive_fs_inv_values()
        field_MV_per_cm = self.field_MV_per_cm_values()
        if drive_code is not None:
            payload["drive_code"] = drive_code
        if drive_fs_inv is not None:
            payload["drive_fs_inv"] = drive_fs_inv
        if field_MV_per_cm is not None:
            payload["field_MV_per_cm"] = field_MV_per_cm
        return payload

    def _add_drive_columns(self, data: dict[str, Any]) -> dict[str, Any]:
        drive_code = self.drive_code_values()
        drive_fs_inv = self.drive_fs_inv_values()
        field_MV_per_cm = self.field_MV_per_cm_values()
        if drive_code is not None:
            data["drive_code"] = drive_code
        if drive_fs_inv is not None:
            data["drive_fs_inv"] = drive_fs_inv
        if field_MV_per_cm is not None:
            data["field_MV_per_cm"] = field_MV_per_cm
        return data

    def components_dataframe(self):
        pd = _require_pandas()
        density = self.density_array()
        dimension = self.dimension()
        data: dict[str, Any] = {"time_fs": _time_axis_fs(self.times, self.times_fs)}
        for index in range(dimension):
            data[_rho_label(index, index)] = density[:, index, index].real
        for i, j in _upper_triangular_pairs(dimension):
            label = _rho_label(i, j)
            values = density[:, i, j]
            phase, phase_unwrapped = _phase_with_mask(values)
            data[f"Re_{label}"] = values.real
            data[f"Im_{label}"] = values.imag
            data[f"abs_{label}"] = np.abs(values)
            data[f"phase_{label}"] = phase
            data[f"phase_{label}_unwrapped"] = phase_unwrapped
        self._add_drive_columns(data)
        return pd.DataFrame(data)

    def populations_dataframe(self):
        pd = _require_pandas()
        populations = self.populations()
        data: dict[str, Any] = {"time_fs": _time_axis_fs(self.times, self.times_fs)}
        for index in range(populations.shape[1]):
            data[_rho_label(index, index)] = populations[:, index].real
        self._add_drive_columns(data)
        return pd.DataFrame(data)

    def selected_elements_dataframe(self, elements: dict[str, tuple[int, int]]):
        pd = _require_pandas()
        data: dict[str, Any] = {"time_fs": _time_axis_fs(self.times, self.times_fs)}
        for label, values in self.selected_elements(elements).items():
            phase, phase_unwrapped = _phase_with_mask(values)
            data[f"Re_{label}"] = values.real
            data[f"Im_{label}"] = values.imag
            data[f"abs_{label}"] = np.abs(values)
            data[f"phase_{label}"] = phase
            data[f"phase_{label}_unwrapped"] = phase_unwrapped
        self._add_drive_columns(data)
        return pd.DataFrame(data)

    def plot_times_and_label(self) -> tuple[np.ndarray, str]:
        if self.times_fs is not None:
            return self.times_fs, "Time (fs)"
        return self.times, "Time"


__all__ = ["DynamicsResult"]
