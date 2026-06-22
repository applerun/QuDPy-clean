"""Standardized TA result objects and default TA outputs."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field as dataclass_field, fields as dataclass_fields, is_dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np


def _json_safe(value: Any) -> Any:
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _json_safe(value.to_dict())
    if is_dataclass(value):
        return _json_safe({item.name: getattr(value, item.name) for item in dataclass_fields(value)})
    if isinstance(value, complex):
        return {"real": float(np.real(value)), "imag": float(np.imag(value))}
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if callable(value):
        return {"callable_serialized": False, "repr": repr(value)}
    return value


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(_json_safe(payload), indent=2, ensure_ascii=False), encoding="utf-8")
    return output


def write_rows(path: str | Path, rows: list[dict[str, Any]]) -> Path:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output


def safe_delay_label(delay_fs: float) -> str:
    value = 0.0 if abs(float(delay_fs)) < 1e-12 else float(delay_fs)
    text = f"{value:.6f}".rstrip("0").rstrip(".")
    if text == "-0":
        text = "0"
    return text.replace("-", "m").replace(".", "p")


def nearest_value(values: Sequence[float], target: float) -> float:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        raise ValueError("values must not be empty.")
    return float(array[int(np.argmin(np.abs(array - float(target))))])


def _complex_columns(prefix: str, value: complex | np.complexfloating) -> dict[str, float]:
    item = complex(value)
    return {
        f"Re_{prefix}": float(item.real),
        f"Im_{prefix}": float(item.imag),
        f"abs_{prefix}": float(abs(item)),
    }


def _interp_real(source_x: np.ndarray, source_y: np.ndarray, target_x: np.ndarray) -> np.ndarray:
    return np.interp(target_x, source_x, np.asarray(source_y, dtype=float))


def _interp_complex(source_x: np.ndarray, source_y: np.ndarray, target_x: np.ndarray) -> np.ndarray:
    values = np.asarray(source_y, dtype=np.complex128)
    return (
        np.interp(target_x, source_x, np.real(values))
        + 1j * np.interp(target_x, source_x, np.imag(values))
    )


@dataclass
class TASpectrum:
    """One absorption spectrum on one energy axis."""

    energy_eV: np.ndarray
    omega_fs_inv: np.ndarray
    absorption: np.ndarray
    omega_im_P_over_E: np.ndarray | None = None
    P_over_E: np.ndarray | None = None
    E_omega: np.ndarray | None = None
    P_omega: np.ndarray | None = None
    abs_E_omega: np.ndarray | None = None
    metadata: dict[str, Any] = dataclass_field(default_factory=dict)

    def __post_init__(self) -> None:
        self.energy_eV = np.asarray(self.energy_eV, dtype=float)
        self.omega_fs_inv = np.asarray(self.omega_fs_inv, dtype=float)
        self.absorption = np.asarray(self.absorption, dtype=float)
        self.omega_im_P_over_E = (
            self.absorption
            if self.omega_im_P_over_E is None
            else np.asarray(self.omega_im_P_over_E, dtype=float)
        )

        if self.energy_eV.ndim != 1:
            raise ValueError("energy_eV must be 1D.")
        for name, value in {
            "omega_fs_inv": self.omega_fs_inv,
            "absorption": self.absorption,
            "omega_im_P_over_E": self.omega_im_P_over_E,
        }.items():
            if value.shape != self.energy_eV.shape:
                raise ValueError(f"{name} must have the same shape as energy_eV.")

        if self.P_over_E is not None:
            self.P_over_E = np.asarray(self.P_over_E, dtype=np.complex128)
            if self.P_over_E.shape != self.energy_eV.shape:
                raise ValueError("P_over_E must have the same shape as energy_eV.")
        if self.E_omega is not None:
            self.E_omega = np.asarray(self.E_omega, dtype=np.complex128)
        if self.P_omega is not None:
            self.P_omega = np.asarray(self.P_omega, dtype=np.complex128)
        if self.abs_E_omega is not None:
            self.abs_E_omega = np.asarray(self.abs_E_omega, dtype=float)
        self.metadata = dict(self.metadata)

    @classmethod
    def from_response(cls, response: dict[str, Any], *, metadata: dict[str, Any] | None = None) -> "TASpectrum":
        merged = dict(response.get("metadata", {}) or {})
        if metadata:
            merged.update(metadata)
        return cls(
            energy_eV=response["energy_eV"],
            omega_fs_inv=response["omega_fs_inv"],
            absorption=response["absorption"],
            omega_im_P_over_E=response.get("omega_im_P_over_E", response["absorption"]),
            P_over_E=response.get("P_over_E"),
            E_omega=response.get("E_omega"),
            P_omega=response.get("P_omega"),
            abs_E_omega=response.get("abs_E_omega"),
            metadata=merged,
        )

    def on_axis(self, energy_axis_eV: np.ndarray, *, allow_interpolation: bool = True) -> "TASpectrum":
        target = np.asarray(energy_axis_eV, dtype=float)
        if self.energy_eV.shape == target.shape and np.allclose(self.energy_eV, target):
            return self
        if not allow_interpolation:
            raise ValueError("Spectrum energy axis differs and interpolation is disabled.")
        return TASpectrum(
            energy_eV=target,
            omega_fs_inv=_interp_real(self.energy_eV, self.omega_fs_inv, target),
            absorption=_interp_real(self.energy_eV, self.absorption, target),
            omega_im_P_over_E=_interp_real(self.energy_eV, self.omega_im_P_over_E, target),
            P_over_E=None if self.P_over_E is None else _interp_complex(self.energy_eV, self.P_over_E, target),
            E_omega=None if self.E_omega is None else _interp_complex(self.energy_eV, self.E_omega, target),
            P_omega=None if self.P_omega is None else _interp_complex(self.energy_eV, self.P_omega, target),
            abs_E_omega=None if self.abs_E_omega is None else _interp_real(self.energy_eV, self.abs_E_omega, target),
            metadata={**self.metadata, "interpolated_to_axis": True},
        )

    def to_rows(self, *, prefix: str = "") -> list[dict[str, Any]]:
        tag = f"{prefix}_" if prefix else ""
        rows = []
        for idx, energy in enumerate(self.energy_eV):
            row: dict[str, Any] = {
                "energy_eV": float(energy),
                "omega_fs_inv": float(self.omega_fs_inv[idx]),
                f"{tag}absorption": float(self.absorption[idx]),
                f"{tag}omega_im_P_over_E": float(self.omega_im_P_over_E[idx]),
            }
            if self.P_over_E is not None:
                row.update(_complex_columns(f"{tag}P_over_E", self.P_over_E[idx]))
            if self.abs_E_omega is not None:
                row[f"{tag}abs_E_omega"] = float(self.abs_E_omega[idx])
            rows.append(row)
        return rows

    def to_dict(self, *, include_arrays: bool = False) -> dict[str, Any]:
        data = {
            "n_points": int(self.energy_eV.size),
            "energy_range_eV": [] if self.energy_eV.size == 0 else [float(self.energy_eV[0]), float(self.energy_eV[-1])],
            "metadata": dict(self.metadata),
        }
        if include_arrays:
            data.update(
                {
                    "energy_eV": self.energy_eV,
                    "omega_fs_inv": self.omega_fs_inv,
                    "absorption": self.absorption,
                    "omega_im_P_over_E": self.omega_im_P_over_E,
                }
            )
        return data


@dataclass
class TADelayResult:
    delay_fs: float
    case_name: str
    pump_center_fs: float
    probe_center_fs: float
    field_metadata: dict[str, Any]
    pump_probe_spectrum: TASpectrum
    probe_only_spectrum_on_axis: TASpectrum
    ta_spectrum: TASpectrum
    pump_probe_result: Any | None = None
    metadata: dict[str, Any] = dataclass_field(default_factory=dict)

    def __post_init__(self) -> None:
        self.delay_fs = float(self.delay_fs)
        self.pump_center_fs = float(self.pump_center_fs)
        self.probe_center_fs = float(self.probe_center_fs)
        self.field_metadata = dict(self.field_metadata)
        self.metadata = dict(self.metadata)

    def to_spectrum_rows(self) -> list[dict[str, Any]]:
        rows = []
        ppe = self.pump_probe_spectrum.P_over_E
        pre = self.probe_only_spectrum_on_axis.P_over_E
        for idx, energy in enumerate(self.ta_spectrum.energy_eV):
            row = {
                "delay_fs": float(self.delay_fs),
                "case_name": self.case_name,
                "energy_eV": float(energy),
                "omega_fs_inv": float(self.ta_spectrum.omega_fs_inv[idx]),
                "S_TA": float(self.ta_spectrum.absorption[idx]),
                "S_pump_probe": float(self.pump_probe_spectrum.absorption[idx]),
                "S_probe_only": float(self.probe_only_spectrum_on_axis.absorption[idx]),
            }
            if ppe is not None:
                row.update(_complex_columns("P_over_E_pump_probe", ppe[idx]))
            if pre is not None:
                row.update(_complex_columns("P_over_E_probe_only", pre[idx]))
            rows.append(row)
        return rows

    def to_dict(self) -> dict[str, Any]:
        return {
            "delay_fs": float(self.delay_fs),
            "case_name": self.case_name,
            "pump_center_fs": float(self.pump_center_fs),
            "probe_center_fs": float(self.probe_center_fs),
            "field_metadata": self.field_metadata,
            "pump_probe_spectrum": self.pump_probe_spectrum.to_dict(),
            "probe_only_spectrum_on_axis": self.probe_only_spectrum_on_axis.to_dict(),
            "ta_spectrum": self.ta_spectrum.to_dict(),
            "metadata": dict(self.metadata),
        }


@dataclass
class TAResult:
    settings: Any
    probe_field_metadata: dict[str, Any]
    probe_only_spectrum: TASpectrum
    delay_results: list[TADelayResult]
    common_energy_eV: np.ndarray
    common_omega_fs_inv: np.ndarray
    delays_fs: np.ndarray
    ta_map: np.ndarray
    pump_probe_map: np.ndarray
    probe_only_spectrum_on_common_axis: np.ndarray
    probe_only_result: Any | None = None
    selected_case_results: dict[str, Any] = dataclass_field(default_factory=dict)
    metadata: dict[str, Any] = dataclass_field(default_factory=dict)

    def __post_init__(self) -> None:
        self.probe_field_metadata = dict(self.probe_field_metadata)
        self.delay_results = list(self.delay_results)
        self.common_energy_eV = np.asarray(self.common_energy_eV, dtype=float)
        self.common_omega_fs_inv = np.asarray(self.common_omega_fs_inv, dtype=float)
        self.delays_fs = np.asarray(self.delays_fs, dtype=float)
        self.ta_map = np.asarray(self.ta_map, dtype=float)
        self.pump_probe_map = np.asarray(self.pump_probe_map, dtype=float)
        self.probe_only_spectrum_on_common_axis = np.asarray(self.probe_only_spectrum_on_common_axis, dtype=float)
        self.selected_case_results = dict(self.selected_case_results)
        self.metadata = dict(self.metadata)

    def to_dict(self) -> dict[str, Any]:
        return {
            "settings": self.settings.to_dict() if hasattr(self.settings, "to_dict") else repr(self.settings),
            "probe_field_metadata": self.probe_field_metadata,
            "n_delays": int(self.delays_fs.size),
            "n_energy_points": int(self.common_energy_eV.size),
            "delays_fs": self.delays_fs,
            "common_energy_range_eV": [] if self.common_energy_eV.size == 0 else [float(self.common_energy_eV[0]), float(self.common_energy_eV[-1])],
            "probe_only_spectrum": self.probe_only_spectrum.to_dict(),
            "delay_results": [item.to_dict() for item in self.delay_results],
            "selected_case_keys": list(self.selected_case_results.keys()),
            "metadata": dict(self.metadata),
        }

    def save(self, output_dir: str | Path, *, io_settings: Any | None = None) -> dict[str, Any]:
        return TAResultIO(output_dir=output_dir, io_settings=io_settings).save(self)


def _common_energy_axis(delay_results: list[TADelayResult], *, policy: str, min_points: int) -> np.ndarray:
    if not delay_results:
        raise ValueError("delay_results must not be empty.")
    if policy == "first":
        return np.asarray(delay_results[0].ta_spectrum.energy_eV, dtype=float)
    if policy != "overlap":
        raise ValueError("policy must be 'overlap' or 'first'.")
    axes = [np.asarray(item.ta_spectrum.energy_eV, dtype=float) for item in delay_results]
    overlap_min = max(float(np.min(axis)) for axis in axes)
    overlap_max = min(float(np.max(axis)) for axis in axes)
    if overlap_max <= overlap_min:
        raise ValueError("Delay spectra do not share an overlapping energy range.")
    reference = axes[0]
    common = reference[(reference >= overlap_min) & (reference <= overlap_max)]
    if common.size < int(min_points):
        raise ValueError(f"Common energy axis has too few points: {common.size}; min_points={min_points}.")
    return common


def build_common_ta_map(
    delay_results: list[TADelayResult],
    *,
    common_axis_policy: str = "overlap",
    allow_energy_axis_interpolation: bool = True,
    min_common_energy_points: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return common_energy, common_omega, ta_map, pump_probe_map, probe_only_map."""

    common_energy = _common_energy_axis(
        delay_results,
        policy=common_axis_policy,
        min_points=int(min_common_energy_points),
    )
    ta_rows, pp_rows, pr_rows, omega_rows = [], [], [], []
    for item in delay_results:
        ta = item.ta_spectrum.on_axis(common_energy, allow_interpolation=allow_energy_axis_interpolation)
        pp = item.pump_probe_spectrum.on_axis(common_energy, allow_interpolation=allow_energy_axis_interpolation)
        pr = item.probe_only_spectrum_on_axis.on_axis(common_energy, allow_interpolation=allow_energy_axis_interpolation)
        ta_rows.append(ta.absorption)
        pp_rows.append(pp.absorption)
        pr_rows.append(pr.absorption)
        omega_rows.append(ta.omega_fs_inv)
    return common_energy, np.mean(np.vstack(omega_rows), axis=0), np.vstack(ta_rows), np.vstack(pp_rows), np.vstack(pr_rows)


class TAResultIO:
    """Default writer for standardized TA analysis products."""

    def __init__(self, *, output_dir: str | Path, io_settings: Any | None = None) -> None:
        self.output_dir = Path(output_dir)
        self.io_settings = io_settings

    def save(self, result: TAResult) -> dict[str, Any]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_files: dict[str, Any] = {}
        output_files.update(self.save_map_data(result))
        output_files["delay_spectra"] = self.save_delay_spectra(result)
        if self._should_save_preview_figures():
            output_files["figures"] = self.save_preview_figures(result)
        output_files["metadata"] = str(self.save_metadata(result, output_files))
        return output_files

    def _should_save_preview_figures(self) -> bool:
        return bool(getattr(self.io_settings, "save_ta_preview_figures", True))

    def _figure_dir(self) -> Path:
        name = str(getattr(self.io_settings, "ta_preview_dir_name", "figures"))
        return self.output_dir / name

    def _preview_dpi(self) -> int:
        return int(getattr(self.io_settings, "ta_preview_dpi", 180))

    def _energy_mask(self, energy: np.ndarray) -> np.ndarray:
        energy_range = getattr(self.io_settings, "ta_preview_energy_range_eV", None)
        if energy_range is None:
            return np.ones_like(energy, dtype=bool)
        lo, hi = energy_range
        mask = (energy >= float(lo)) & (energy <= float(hi))
        if not np.any(mask):
            return np.ones_like(energy, dtype=bool)
        return mask

    def save_map_data(self, result: TAResult) -> dict[str, str]:
        data_dir = self.output_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        for i_delay, delay in enumerate(result.delays_fs):
            for i_energy, energy in enumerate(result.common_energy_eV):
                rows.append(
                    {
                        "delay_fs": float(delay),
                        "energy_eV": float(energy),
                        "omega_fs_inv": float(result.common_omega_fs_inv[i_energy]),
                        "S_TA": float(result.ta_map[i_delay, i_energy]),
                        "S_pump_probe": float(result.pump_probe_map[i_delay, i_energy]),
                        "S_probe_only": float(result.probe_only_spectrum_on_common_axis[i_energy]),
                    }
                )
        ta_map_csv = write_rows(data_dir / "ta_map.csv", rows)
        ta_map_npz = data_dir / "ta_map.npz"
        np.savez_compressed(
            ta_map_npz,
            delays_fs=result.delays_fs,
            energy_eV=result.common_energy_eV,
            omega_fs_inv=result.common_omega_fs_inv,
            S_TA=result.ta_map,
            S_pump_probe=result.pump_probe_map,
            S_probe_only=result.probe_only_spectrum_on_common_axis,
        )
        probe_csv = write_rows(data_dir / "probe_reference_spectrum.csv", result.probe_only_spectrum.to_rows(prefix="probe_only"))
        output = {
            "ta_map_csv": str(ta_map_csv),
            "ta_map_npz": str(ta_map_npz),
            "probe_reference_spectrum_csv": str(probe_csv),
        }
        kinetic_path = self._save_kinetic_if_requested(result)
        if kinetic_path is not None:
            output["kinetic_trace_csv"] = str(kinetic_path)
        return output

    def _save_kinetic_if_requested(self, result: TAResult) -> Path | None:
        standardize = getattr(getattr(result, "settings", None), "standardize", None)
        kinetic_energy = None if standardize is None else getattr(standardize, "kinetic_energy_eV", None)
        if kinetic_energy is None:
            return None
        idx = int(np.argmin(np.abs(result.common_energy_eV - float(kinetic_energy))))
        actual_energy = float(result.common_energy_eV[idx])
        token = f"{actual_energy:.4f}".rstrip("0").rstrip(".").replace(".", "p")
        rows = [
            {
                "delay_fs": float(delay),
                "energy_eV": actual_energy,
                "S_TA": float(result.ta_map[i, idx]),
                "S_pump_probe": float(result.pump_probe_map[i, idx]),
                "S_probe_only": float(result.probe_only_spectrum_on_common_axis[idx]),
            }
            for i, delay in enumerate(result.delays_fs)
        ]
        return write_rows(self.output_dir / "data" / f"kinetic_trace_at_{token}_eV.csv", rows)

    def save_delay_spectra(self, result: TAResult) -> list[str]:
        spectra_dir = self.output_dir / "data" / "delay_spectra"
        spectra_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        for item in result.delay_results:
            path = spectra_dir / f"delay_{safe_delay_label(item.delay_fs)}_fs.csv"
            write_rows(path, item.to_spectrum_rows())
            paths.append(str(path))
        return paths

    def save_preview_figures(self, result: TAResult) -> dict[str, Any]:
        """Save TA-level preview figures.

        These figures are generated from the standardized TAResult arrays.  They
        are different from raw DynamicsResult previews, which should be produced
        from checkpoints through TAPlan.save_preview_from_checkpoints().
        """

        fig_dir = self._figure_dir()
        fig_dir.mkdir(parents=True, exist_ok=True)
        outputs: dict[str, Any] = {
            "ta_map": str(self._plot_ta_map(result, fig_dir)),
            "selected_delay_lineouts": str(self._plot_selected_delay_lineouts(result, fig_dir)),
        }
        kinetic = self._plot_kinetic_trace_if_requested(result, fig_dir)
        if kinetic is not None:
            outputs["kinetic_trace"] = str(kinetic)
        return outputs

    def _plot_ta_map(self, result: TAResult, fig_dir: Path) -> Path:
        import matplotlib.pyplot as plt

        energy = result.common_energy_eV
        delays = result.delays_fs
        values = result.ta_map
        mask = self._energy_mask(energy)
        cmap = str(getattr(self.io_settings, "ta_preview_cmap", "plasma"))

        fig, ax = plt.subplots(figsize=(6.8, 4.4))
        mesh = ax.pcolormesh(
            energy[mask],
            delays,
            values[:, mask],
            shading="auto",
            cmap=cmap,
        )
        ax.set_xlabel("Energy (eV)")
        ax.set_ylabel("Probe delay (fs)")
        ax.set_title("Intrinsic TA response")
        cbar = fig.colorbar(mesh, ax=ax)
        cbar.set_label("S_TA")
        fig.tight_layout()

        path = fig_dir / "ta_map.png"
        fig.savefig(path, dpi=self._preview_dpi())
        plt.close(fig)
        return path

    def _selected_lineout_delays(self, result: TAResult) -> list[float]:
        configured = tuple(getattr(self.io_settings, "selected_lineout_delays_fs", ()) or ())
        if configured:
            selected = [nearest_value(result.delays_fs, delay) for delay in configured]
        else:
            n = int(result.delays_fs.size)
            if n <= 7:
                selected = [float(item) for item in result.delays_fs]
            else:
                indices = np.linspace(0, n - 1, 7).round().astype(int)
                selected = [float(result.delays_fs[idx]) for idx in indices]
        unique: list[float] = []
        for value in selected:
            if not any(abs(value - existing) < 1e-9 for existing in unique):
                unique.append(float(value))
        return unique

    def _plot_selected_delay_lineouts(self, result: TAResult, fig_dir: Path) -> Path:
        import matplotlib.pyplot as plt

        energy = result.common_energy_eV
        mask = self._energy_mask(energy)
        selected = self._selected_lineout_delays(result)
        cmap = plt.get_cmap(str(getattr(self.io_settings, "ta_preview_cmap", "plasma")))
        colors = cmap(np.linspace(0.08, 0.92, max(len(selected), 1)))

        fig, ax = plt.subplots(figsize=(6.8, 4.2))
        for color, delay in zip(colors, selected):
            idx = int(np.argmin(np.abs(result.delays_fs - float(delay))))
            ax.plot(
                energy[mask],
                result.ta_map[idx, mask],
                linewidth=1.0,
                color=color,
                label=f"{result.delays_fs[idx]:g} fs",
            )

        ax.set_title("Selected TA lineouts")
        ax.set_xlabel("Energy (eV)")
        ax.set_ylabel("S_TA")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, ncols=2, fontsize=8)
        fig.tight_layout()

        path = fig_dir / "selected_delay_lineouts.png"
        fig.savefig(path, dpi=self._preview_dpi())
        plt.close(fig)
        return path

    def _plot_kinetic_trace_if_requested(self, result: TAResult, fig_dir: Path) -> Path | None:
        standardize = getattr(getattr(result, "settings", None), "standardize", None)
        kinetic_energy = None if standardize is None else getattr(standardize, "kinetic_energy_eV", None)
        if kinetic_energy is None:
            return None

        import matplotlib.pyplot as plt

        idx = int(np.argmin(np.abs(result.common_energy_eV - float(kinetic_energy))))
        actual_energy = float(result.common_energy_eV[idx])
        token = f"{actual_energy:.4f}".rstrip("0").rstrip(".").replace(".", "p")

        fig, ax = plt.subplots(figsize=(6.4, 4.0))
        ax.plot(result.delays_fs, result.ta_map[:, idx], marker="o", linewidth=1.2)
        ax.axhline(0.0, linewidth=0.8)
        ax.set_title(f"TA kinetics at {actual_energy:.3f} eV")
        ax.set_xlabel("Probe delay (fs)")
        ax.set_ylabel("S_TA")
        ax.grid(alpha=0.25)
        fig.tight_layout()

        path = fig_dir / f"kinetic_trace_at_{token}_eV.png"
        fig.savefig(path, dpi=self._preview_dpi())
        plt.close(fig)
        return path

    def save_metadata(self, result: TAResult, output_files: dict[str, Any]) -> Path:
        payload = result.to_dict()
        payload["output_files"] = output_files
        return write_json(self.output_dir / "meta.json", payload)


__all__ = [
    "TASpectrum",
    "TADelayResult",
    "TAResult",
    "TAResultIO",
    "build_common_ta_map",
    "nearest_value",
    "safe_delay_label",
    "write_json",
    "write_rows",
]
