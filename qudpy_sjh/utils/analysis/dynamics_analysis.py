"""基于 `DynamicsResult` checkpoint 的动力学后处理分析。

本模块属于 analysis 层：只读取已经完成的 `DynamicsResult`，计算偶极矩、
宏观 polarization、FFT 和 response-like 频域量；不调用 solver，也不改变
solver/result 主架构。
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields as dataclass_fields, is_dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

from qudpy_sjh.utils.core.normalization import ParaNormalizer
from qudpy_sjh.utils.core.results import DynamicsResult
from qudpy_sjh.utils.spectroscopy.observables import dipole_expectation_D, polarization_C_per_m2


def _require_pandas():
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("pandas is required for DynamicsAnalysis CSV output.") from exc
    return pd


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
    if isinstance(value, np.ndarray):
        if np.iscomplexobj(value):
            return [{"real": float(item.real), "imag": float(item.imag)} for item in value.ravel()]
        return value.tolist()
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, complex):
        return {"real": float(value.real), "imag": float(value.imag)}
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _safe_name(value: str | None, fallback: str) -> str:
    if value is None or str(value).strip() == "":
        return fallback
    cleaned = [char if (char.isalnum() or char in ("-", "_")) else "_" for char in str(value)]
    return "".join(cleaned).strip("_") or fallback


def _complex_columns(prefix: str, values: np.ndarray) -> dict[str, Any]:
    """Return CSV-friendly real-valued columns for a complex spectrum."""

    return {
        f"Re_{prefix}"   : values.real,
        f"Im_{prefix}"   : values.imag,
        f"abs_{prefix}"  : np.abs(values),
        f"phase_{prefix}": np.angle(values),
    }


def _require_number_density(number_density_m3: float | None) -> float:
    if number_density_m3 is None:
        raise ValueError("number_density_m3 must be provided explicitly for polarization analysis.")
    value = float(number_density_m3)
    if value < 0:
        raise ValueError("number_density_m3 must be non-negative.")
    return value


def _divide_with_mask(
        numerator: np.ndarray,
        denominator: np.ndarray,
        response_rel_epsilon: float,
) -> np.ndarray:
    """Divide spectra with a relative denominator mask.

    Frequencies where ``abs(denominator)`` is too small relative to its maximum
    are assigned complex NaN. A relative threshold is less unit-dependent than
    an absolute FFT-amplitude cutoff.
    """

    if response_rel_epsilon <= 0:
        raise ValueError("response_rel_epsilon must be positive.")
    max_denominator = float(np.max(np.abs(denominator)))
    if max_denominator == 0.0:
        raise ValueError("input FFT is identically zero; response division is undefined.")
    threshold = float(response_rel_epsilon) * max_denominator
    response = np.full_like(numerator, np.nan + 1j * np.nan, dtype = np.complex128)
    valid = np.abs(denominator) > threshold
    response[valid] = numerator[valid] / denominator[valid]
    return response


def _require_real_physical_signal(values: np.ndarray, *, name: str, tolerance: float) -> np.ndarray:
    if tolerance < 0:
        raise ValueError("imag_tolerance must be non-negative.")
    array = np.asarray(values, dtype = np.complex128)
    max_imag = float(np.max(np.abs(array.imag))) if array.size else 0.0
    if max_imag > tolerance:
        raise ValueError(
            f"{name} has imaginary component {max_imag:.6g}, which exceeds imag_tolerance={tolerance:.6g}."
        )
    return array.real


@dataclass
class DynamicsAnalysis:
    """对单条 `DynamicsResult` 轨迹做后处理分析。

    一个 `DynamicsAnalysis` 对象对应一个已经完成的 `DynamicsResult`。它可以
    从内存中的 result 创建，也可以从 `.ckp` checkpoint 读取。未来可继续扩展
    多能级谱学、参数扫描汇总等分析。
    """

    result: DynamicsResult
    example_name: str | None = None
    case_name: str | None = None
    metadata: dict[str, Any] = field(default_factory = dict)

    @classmethod
    def from_result(
            cls,
            result: DynamicsResult,
            *,
            example_name: str | None = None,
            case_name: str | None = None,
            metadata: dict[str, Any] | None = None,
    ) -> "DynamicsAnalysis":
        """从内存中的 `DynamicsResult` 创建分析对象。"""

        if not isinstance(result, DynamicsResult):
            raise TypeError("result must be a DynamicsResult.")
        return cls(
            result = result,
            example_name = example_name,
            case_name = case_name,
            metadata = {} if metadata is None else dict(metadata),
        )

    @classmethod
    def from_checkpoint(
            cls,
            file: str | Path,
            *,
            example_name: str | None = None,
            case_name: str | None = None,
            metadata: dict[str, Any] | None = None,
    ) -> "DynamicsAnalysis":
        """从 `.ckp` checkpoint 文件创建分析对象。

        `.ckp` 必须由 `DynamicsResult.save_ckp()` 生成。该方法只读取 checkpoint，
        不调用 solver；不要加载不可信来源的 `.ckp` 文件。
        """

        result = DynamicsResult.from_ckp(file)
        local_metadata = {} if metadata is None else dict(metadata)
        local_metadata["checkpoint_file"] = str(Path(file))
        return cls.from_result(
            result,
            example_name = example_name,
            case_name = case_name,
            metadata = local_metadata,
        )

    @property
    def resolved_example_name(self) -> str:
        """返回用于输出目录的 example 名称。"""

        return _safe_name(self.example_name, "analysis")

    @property
    def resolved_case_name(self) -> str:
        """返回用于输出目录的 case 名称。"""

        return _safe_name(self.case_name, f"{self.result.mode}_N{self.result.dimension()}")

    def time_fs(self) -> np.ndarray:
        """返回物理时间轴，单位 fs。

        analysis 层要求物理时间轴存在。若 result 没有 `times_fs`，直接报错，
        不猜测 code time 是否等价于 fs。
        """

        if self.result.times_fs is None:
            raise ValueError("DynamicsAnalysis requires result.times_fs in physical fs units.")
        return np.asarray(self.result.times_fs, dtype = float)

    def coherence(self, pair: tuple[int, int] = (0, 1)) -> np.ndarray:
        """返回指定 coherence 轨迹，默认使用 zero-based `rho_01`。

        对 N>2，可通过 `pair=(i, j)` 显式选择矩阵元。索引越界会由
        `DynamicsResult.matrix_element()` 直接抛错。
        """

        i, j = pair
        return self.result.matrix_element(i, j)

    def dipole_expectation_D(self, *, imag_tolerance: float = 1e-10) -> np.ndarray:
        """计算单体系偶极矩期望值 `p_D(t)=Tr[rho(t) mu_D]`。

        `rho(t)` 来自完整 density matrix trajectory，`mu_D` 来自
        `physical_params.dipole_matrix_D`，单位 Debye。若缺少物理偶极矩或维度
        不匹配，底层 observable 会直接抛出清晰错误。
        """

        physical = self.result.physical_params
        if physical is None:
            raise ValueError("physical_params is required to compute dipole expectation.")
        if not hasattr(physical, "dipole_matrix_D"):
            raise ValueError("physical_params.dipole_matrix_D is required.")
        dipole = dipole_expectation_D(self.result.density_array(), physical.dipole_matrix_D)
        return _require_real_physical_signal(dipole, name = "dipole_expectation_D", tolerance = imag_tolerance)

    def full_polarization_C_per_m2(
            self,
            *,
            number_density_m3: float | None = None,
            imag_tolerance: float = 1e-10,
    ) -> np.ndarray:
        """计算通用 N-level 宏观 polarization，单位 C/m^2。

        通用公式为 `P(t)=n Tr[rho(t) mu_D] * DEBYE_TO_C_M`。这里使用完整
        density matrix 和完整 `dipole_matrix_D`，因此适用于 N=2 和 N>2。
        `number_density_m3` 必须显式提供，单位是 `m^-3`。
        """

        physical = self.result.physical_params
        if physical is None:
            raise ValueError("physical_params is required to compute polarization.")
        if not hasattr(physical, "dipole_matrix_D"):
            raise ValueError("physical_params.dipole_matrix_D is required.")
        polarization = polarization_C_per_m2(
            self.result.density_array(),
            physical.dipole_matrix_D,
            _require_number_density(number_density_m3),
        )
        return _require_real_physical_signal(
            polarization,
            name = "full_polarization_C_per_m2",
            tolerance = imag_tolerance,
        )

    def _resolve_input_kind(self, kind: str = "auto") -> str:
        selected = kind
        if selected == "auto":
            if self.result.mode in ("lab_exact", "rotating_view"):
                selected = "field"
            elif self.result.mode == "rwa":
                selected = "drive"
            else:
                raise ValueError(
                    "input_kind='auto' is only defined for lab_exact, rotating_view, and rwa results."
                )
        if selected not in ("field", "drive"):
            raise ValueError("kind must be one of: auto, field, drive.")
        return selected

    def input_signal(self, *, kind: str = "auto") -> tuple[np.ndarray, str, str]:
        """返回用于频域分析的输入信号。

        `kind="auto"` 时，`lab_exact` 和 `rotating_view` 使用物理电场 `E(t)`，
        单位 `MV/cm`；`rwa` 使用慢变量 drive `g(t)`，单位 `fs^-1`。
        若 result 缺少对应输入信息，直接报错，不静默猜测。
        """

        selected = self._resolve_input_kind(kind)
        if selected == "field":
            field = self.result.field_MV_per_cm_values()
            if field is None:
                raise ValueError("field_MV_per_cm_values is required for field FFT analysis.")
            return np.asarray(field, dtype = float), "E(t)", "MV/cm"
        if selected == "drive":
            drive = self.result.drive_fs_inv_values()
            if drive is None:
                raise ValueError("drive_fs_inv_values is required for drive FFT analysis.")
            return np.asarray(drive, dtype = float), "g(t)", "fs^-1"
        raise AssertionError("unreachable input signal kind")

    def time_domain_dataframe(self, *, number_density_m3: float | None = None):
        """生成增强版时域 analysis dataframe。

        输出保留 `DynamicsResult.components_dataframe()` 的 dimension-aware 列，
        并新增通用 `dipole_expectation_D` 和 `full_polarization_C_per_m2` 列。
        """

        frame = self.result.components_dataframe().copy()
        dipole = self.dipole_expectation_D()
        full_polarization = self.full_polarization_C_per_m2(number_density_m3 = number_density_m3)
        frame["Re_dipole_expectation_D"] = dipole.real
        frame["Im_dipole_expectation_D"] = 0.0
        frame["abs_dipole_expectation_D"] = np.abs(dipole)
        frame["Re_full_polarization_C_per_m2"] = full_polarization.real
        frame["Im_full_polarization_C_per_m2"] = 0.0
        frame["abs_full_polarization_C_per_m2"] = np.abs(full_polarization)
        return frame

    def fft_response_dataframe(
            self,
            *,
            number_density_m3: float | None = None,
            pair: tuple[int, int] = (0, 1),
            input_kind: str = "auto",
            window: str | None = "hann",
            subtract_mean: bool = False,
            positive_only: bool = True,
            response_rel_epsilon: float = 1e-8,
    ):
        """计算 coherence 与通用 polarization 的 response-like 频域量。

        `rho_over_input = fft_coherence / fft_input` 是 coherence response-like quantity，
        不是 `chi` 或 absorption。`P_over_input = P_fft / fft_input` 是 polarization
        response-like quantity；`omega_Im_P_over_input` 给出常用于吸收功率方向判断
        的 `omega * Im[P_over_input]`，但仍依赖 Fourier convention 和线性响应条件。
        """

        pd = _require_pandas()
        t_fs = self.time_fs()
        if len(t_fs) < 2:
            raise ValueError("FFT requires at least two time points.")
        dt = np.diff(t_fs)
        if not np.allclose(dt, dt[0], rtol = 1e-5, atol = 1e-10):
            raise ValueError("FFT requires a uniformly sampled time axis.")

        coherence = np.asarray(self.coherence(pair), dtype = np.complex128)
        polarization = np.asarray(
            self.full_polarization_C_per_m2(number_density_m3 = number_density_m3),
            dtype = np.complex128,
        )
        input_values, input_name, input_unit = self.input_signal(kind = input_kind)
        input_values = np.asarray(input_values, dtype = float)
        if input_values.shape != t_fs.shape:
            raise ValueError(
                "input signal must have the same shape as time_fs. "
                f"Got input.shape={input_values.shape}, time_fs.shape={t_fs.shape}."
            )

        coherence_signal = coherence.copy()
        polarization_signal = polarization.copy()
        input_signal = input_values.astype(np.complex128)
        if subtract_mean:
            coherence_signal = coherence_signal - np.mean(coherence_signal)
            polarization_signal = polarization_signal - np.mean(polarization_signal)
            input_signal = input_signal - np.mean(input_signal)

        win = self._window_values(len(t_fs), window)
        fft_coherence = np.fft.fft(coherence_signal * win)
        fft_polarization = np.fft.fft(polarization_signal * win)
        fft_input = np.fft.fft(input_signal * win)
        frequency = np.fft.fftfreq(len(t_fs), d = float(dt[0]))
        angular = 2.0 * np.pi * frequency
        energy_eV = angular / ParaNormalizer.EV_TO_FS_INV
        coherence_over_input = _divide_with_mask(fft_coherence, fft_input, response_rel_epsilon)
        polarization_over_input = _divide_with_mask(fft_polarization, fft_input, response_rel_epsilon)
        omega_im_p_over_input = angular * np.imag(polarization_over_input)

        if positive_only:
            mask = frequency >= 0
            frequency = frequency[mask]
            angular = angular[mask]
            energy_eV = energy_eV[mask]
            fft_coherence = fft_coherence[mask]
            fft_polarization = fft_polarization[mask]
            fft_input = fft_input[mask]
            coherence_over_input = coherence_over_input[mask]
            polarization_over_input = polarization_over_input[mask]
            omega_im_p_over_input = omega_im_p_over_input[mask]

        data: dict[str, Any] = {
            "frequency_fs_inv"        : frequency,
            "angular_frequency_fs_inv": angular,
            "energy_eV"               : energy_eV,
            "input_signal_name"       : input_name,
            "input_signal_unit"       : input_unit,
            "input_kind_requested"    : input_kind,
            "input_kind_resolved"     : self._resolve_input_kind(input_kind),
        }
        data.update(_complex_columns("fft_coherence", fft_coherence))
        data.update(_complex_columns("fft_input", fft_input))
        data.update(_complex_columns("coherence_over_input", coherence_over_input))
        data.update(_complex_columns("P_fft", fft_polarization))
        data.update(_complex_columns("P_over_input", polarization_over_input))
        data["omega_Im_P_over_input"] = omega_im_p_over_input
        return pd.DataFrame(data)

    @staticmethod
    def _window_values(size: int, window: str | None) -> np.ndarray:
        if window is None or window == "none":
            return np.ones(size, dtype = float)
        if window == "hann":
            return np.hanning(size)
        raise ValueError("supported windows are: None, 'none', 'hann'.")

    def plot_polarization_time(self, *, ax = None, number_density_m3: float | None = None):
        """绘制通用时域 polarization `P(t)`。

        返回 `(fig, ax)`，不在此方法中保存文件。保存由 `save_outputs()` 完成。
        """

        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots(figsize = (6.0, 3.2))
        else:
            fig = ax.figure
        polarization = self.full_polarization_C_per_m2(number_density_m3 = number_density_m3)
        ax.plot(self.time_fs(), polarization.real, label = "Re P(t)")
        if np.max(np.abs(polarization.imag)) > 1e-14:
            ax.plot(self.time_fs(), polarization.imag, label = "Im P(t)")
            ax.legend()
        ax.set_xlabel("Time (fs)")
        ax.set_ylabel("P(t) (C/m^2)")
        ax.set_title("Full polarization")
        return fig, ax

    def plot_fft_response(
            self,
            *,
            ax = None,
            number_density_m3: float | None = None,
            pair: tuple[int, int] = (0, 1),
            input_kind: str = "auto",
            window: str | None = "hann",
            subtract_mean: bool = False,
            positive_only: bool = True,
            response_rel_epsilon: float = 1e-8,
    ):
        """绘制 `|P_over_input|` 频域响应强度。

        返回 `(fig, ax)`。该图展示 polarization response-like quantity，
        不直接命名为 `chi` 或 absorption。
        """

        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots(figsize = (6.0, 3.2))
        else:
            fig = ax.figure
        frame = self.fft_response_dataframe(
            number_density_m3 = number_density_m3,
            pair = pair,
            input_kind = input_kind,
            window = window,
            subtract_mean = subtract_mean,
            positive_only = positive_only,
            response_rel_epsilon = response_rel_epsilon,
        )
        input_name = frame["input_signal_name"].iloc[0] if len(frame) else "input"
        ax.plot(frame["energy_eV"], frame["abs_P_over_input"])
        ax.set_xlabel("Energy (eV)")
        ax.set_ylabel("|P_over_input|")
        ax.set_title(f"Polarization response-like spectrum vs {input_name}")
        return fig, ax

    def _analysis_request_dict(
            self,
            *,
            number_density_m3: float,
            pair: tuple[int, int],
            input_kind: str,
            window: str | None,
            subtract_mean: bool,
            positive_only: bool,
            response_rel_epsilon: float,
    ) -> dict[str, Any]:
        return _json_safe(
            {
                "result_mode"         : self.result.mode,
                "dimension"           : self.result.dimension(),
                "pair"                : list(pair),
                "number_density_m3"   : float(number_density_m3),
                "input_kind_requested": input_kind,
                "input_kind_resolved" : self._resolve_input_kind(input_kind),
                "fft_parameters"      : {
                    "window"              : window,
                    "subtract_mean"       : bool(subtract_mean),
                    "positive_only"       : bool(positive_only),
                    "response_rel_epsilon": float(response_rel_epsilon),
                },
            }
        )

    def _existing_metadata_matches(
            self,
            metadata: dict[str, Any],
            *,
            number_density_m3: float,
            pair: tuple[int, int],
            input_kind: str,
            window: str | None,
            subtract_mean: bool,
            positive_only: bool,
            response_rel_epsilon: float,
    ) -> bool:
        expected = self._analysis_request_dict(
            number_density_m3 = number_density_m3,
            pair = pair,
            input_kind = input_kind,
            window = window,
            subtract_mean = subtract_mean,
            positive_only = positive_only,
            response_rel_epsilon = response_rel_epsilon,
        )
        return metadata.get("analysis_request") == expected

    def metadata_dict(
            self,
            *,
            number_density_m3: float | None = None,
            pair: tuple[int, int] = (0, 1),
            input_kind: str = "auto",
            window: str | None = "hann",
            subtract_mean: bool = False,
            positive_only: bool = True,
            response_rel_epsilon: float = 1e-8,
            output_files: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """生成 analysis metadata。

        metadata 记录分析参数、单位约定、物理参数摘要和输出文件路径。code-unit
        细节仍由 checkpoint 或 `debug_meta.json` 负责。
        """

        density = _require_number_density(number_density_m3)
        physical = self.result.physical_params
        solver = self.result.solver_params
        input_values, input_name, input_unit = self.input_signal(kind = input_kind)
        analysis_request = self._analysis_request_dict(
            number_density_m3 = density,
            pair = pair,
            input_kind = input_kind,
            window = window,
            subtract_mean = subtract_mean,
            positive_only = positive_only,
            response_rel_epsilon = response_rel_epsilon,
        )
        return _json_safe(
            {
                "analysis_type"            : "DynamicsAnalysis",
                "example_name"             : self.example_name,
                "case_name"                : self.case_name,
                "result_mode"              : self.result.mode,
                "dimension"                : self.result.dimension(),
                "pair"                     : list(pair),
                "number_density_m3"        : density,
                "analysis_request"         : analysis_request,
                "polarization_definition"  : "P(t) = number_density_m3 * Tr[rho(t) mu_D] * DEBYE_TO_C_M",
                "coherence_over_input_note": (
                    "coherence_over_input is a coherence response-like quantity, not chi or absorption."
                ),
                "P_over_input_note"        : (
                    "P_over_input and omega_Im_P_over_input are polarization response-like quantities; "
                    "interpretation depends on Fourier convention and linear-response conditions."
                ),
                "time_unit"                : "fs",
                "density_matrix_unit"      : "dimensionless",
                "dipole_unit"              : "Debye",
                "polarization_unit"        : "C/m^2",
                "frequency_axis"           : {
                    "frequency_fs_inv"        : "cycles/fs",
                    "angular_frequency_fs_inv": "rad/fs",
                    "energy_eV"               : "hbar * angular_frequency",
                },
                "input_signal"             : {
                    "name"                : input_name,
                    "unit"                : input_unit,
                    "input_kind_requested": input_kind,
                    "input_kind_resolved" : self._resolve_input_kind(input_kind),
                    "n_points"            : len(input_values),
                },
                "fft_parameters"           : analysis_request["fft_parameters"],
                "input_field_class"        : self._input_field_class_name(),
                "physical_params"          : physical,
                "solver_params_fs_inv"     : None
                if solver is None
                else {
                    "time_scale_fs"                 : solver.time_scale_fs,
                    "energies_fs_inv"               : solver.energies_fs_inv,
                    "omega_L_fs_inv"                : solver.omega_L_fs_inv,
                    "detuning_fs_inv"               : solver.detuning_fs_inv,
                    "coupling_matrix_fs_inv"        : solver.coupling_matrix_fs_inv,
                    "relaxation_channels_fs_inv"    : solver.relaxation_channels_fs_inv,
                    "pure_dephasing_channels_fs_inv": solver.pure_dephasing_channels_fs_inv,
                },
                "output_files"             : output_files or {},
                "extra_metadata"           : self.metadata,
            }
        )

    def _input_field_class_name(self) -> str | None:
        """返回面向用户的输入场/drive 类名。"""

        physical = self.result.physical_params
        if physical is None:
            return None
        field_payload = physical.field.to_dict() if getattr(physical, "field", None) is not None else {}
        envelope = field_payload.get("envelope", "unknown")
        if self.result.mode == "rwa":
            return "gaussian_rwa_envelope" if envelope == "gaussian" else "constant_rwa_envelope"
        if getattr(physical, "field", None) is not None:
            return physical.field.__class__.__name__
        return None

    def save_outputs(
            self,
            output_dir: str | Path = "outputs/analysis",
            *,
            example_name: str | None = None,
            case_name: str | None = None,
            overwrite: bool = False,
            number_density_m3: float | None = None,
            pair: tuple[int, int] = (0, 1),
            input_kind: str = "auto",
            window: str | None = "hann",
            subtract_mean: bool = False,
            positive_only: bool = True,
            response_rel_epsilon: float = 1e-8,
            save_csv: bool = True,
            save_png: bool = True,
            save_json: bool = True,
            dpi: int = 160,
    ) -> dict[str, Path]:
        """保存 analysis CSV / PNG / JSON 输出。

        默认输出到 `outputs/analysis/<example_name>/<case_name>/`。若已有
        `analysis_metadata.json` 且 `overwrite=False`，只有当已有 metadata 的
        analysis 参数与当前请求一致时才复用；否则直接报错，要求显式传入
        `overwrite=True`。
        """

        density = _require_number_density(number_density_m3)
        local_example = _safe_name(example_name or self.example_name, self.resolved_example_name)
        local_case = _safe_name(case_name or self.case_name, self.resolved_case_name)
        root = Path(output_dir) / local_example / local_case
        root.mkdir(parents = True, exist_ok = True)
        metadata_path = root / "analysis_metadata.json"

        if metadata_path.exists() and not overwrite:
            metadata = json.loads(metadata_path.read_text(encoding = "utf-8"))
            if not self._existing_metadata_matches(
                    metadata,
                    number_density_m3 = density,
                    pair = pair,
                    input_kind = input_kind,
                    window = window,
                    subtract_mean = subtract_mean,
                    positive_only = positive_only,
                    response_rel_epsilon = response_rel_epsilon,
            ):
                raise ValueError(
                    "Existing analysis output was generated with different analysis parameters. "
                    "Use overwrite=True or choose another output directory."
                )
            existing = {
                key: root / value
                for key, value in metadata.get("output_files", {}).items()
                if isinstance(value, str)
            }
            existing["analysis_metadata"] = metadata_path
            return existing

        written: dict[str, Path] = {}
        if save_csv:
            components_path = root / "analysis_components.csv"
            fft_path = root / "fft_response.csv"
            self.time_domain_dataframe(number_density_m3 = density).to_csv(
                components_path,
                index = False,
            )
            self.fft_response_dataframe(
                number_density_m3 = density,
                pair = pair,
                input_kind = input_kind,
                window = window,
                subtract_mean = subtract_mean,
                positive_only = positive_only,
                response_rel_epsilon = response_rel_epsilon,
            ).to_csv(fft_path, index = False)
            written["analysis_components"] = components_path
            written["fft_response"] = fft_path

        if save_png:
            import matplotlib.pyplot as plt

            figs_dir = root / "figs"
            figs_dir.mkdir(parents = True, exist_ok = True)
            fig, _ax = self.plot_polarization_time(number_density_m3 = density)
            polarization_path = figs_dir / "polarization_time.png"
            fig.savefig(polarization_path, dpi = dpi)
            plt.close(fig)
            fig, _ax = self.plot_fft_response(
                number_density_m3 = density,
                pair = pair,
                input_kind = input_kind,
                window = window,
                subtract_mean = subtract_mean,
                positive_only = positive_only,
                response_rel_epsilon = response_rel_epsilon,
            )
            fft_path = figs_dir / "fft_response.png"
            fig.savefig(fft_path, dpi = dpi)
            plt.close(fig)
            written["polarization_time_png"] = polarization_path
            written["fft_response_png"] = fft_path

        if save_json:
            output_files = {key: path.relative_to(root).as_posix() for key, path in written.items()}
            metadata = self.metadata_dict(
                number_density_m3 = density,
                pair = pair,
                input_kind = input_kind,
                window = window,
                subtract_mean = subtract_mean,
                positive_only = positive_only,
                response_rel_epsilon = response_rel_epsilon,
                output_files = output_files,
            )
            metadata_path.write_text(json.dumps(metadata, indent = 2, ensure_ascii = False), encoding = "utf-8")
            written["analysis_metadata"] = metadata_path

        return written


__all__ = ["DynamicsAnalysis"]
