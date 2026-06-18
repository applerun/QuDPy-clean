"""Result output and case-level data management helpers.

User-facing CSV and figure outputs should prefer physical units when available.
Internal code-unit diagnostics are still saved, but only under explicit names.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, fields as dataclass_fields, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np

from sjh_learn.utils.core.results import DynamicsResult


ResultLike = DynamicsResult

HC_EV_NM = 1239.8419843320026


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


def format_value_tag(value: float) -> str:
    return f"{value:.4f}".rstrip("0").rstrip(".").replace("-", "m").replace(".", "p")


def _physical_field_payload(physical: Any) -> dict[str, Any] | None:
    field = getattr(physical, "field", None)
    if field is None:
        return None
    if not hasattr(field, "to_dict"):
        raise TypeError("NLevelPhysicalParams.field must expose to_dict() for metadata export.")
    payload = field.to_dict()
    if not isinstance(payload, dict):
        raise TypeError("field.to_dict() must return a dict.")
    return _json_safe(payload)


def _field_metadata_value(physical: Any, key: str) -> Any:
    payload = _physical_field_payload(physical)
    if payload is None:
        return None
    value = payload.get(key)
    if value is not None:
        return value
    metadata = payload.get("metadata")
    if isinstance(metadata, dict):
        return metadata.get(key)
    return None


def _field_E0_MV_per_cm(physical: Any) -> float | None:
    value = _field_metadata_value(physical, "E0_MV_per_cm")
    return None if value is None else float(value)


def _field_laser_energy_eV(physical: Any, solver: Any | None = None) -> float | None:
    value = _field_metadata_value(physical, "laser_energy_eV")
    if value is not None:
        return float(value)
    omega = _field_metadata_value(physical, "omega_L_fs_inv")
    if omega is None and solver is not None:
        omega = getattr(solver, "omega_L_fs_inv", None)
    if omega is None:
        return None
    from sjh_learn.utils.core.normalization import ParaNormalizer

    return float(ParaNormalizer.fs_inv_to_energy_eV(float(omega)))


def _field_pulse_value(physical: Any, key: str, fallback_key: str) -> float | None:
    value = _field_metadata_value(physical, key)
    if value is None:
        value = _field_metadata_value(physical, fallback_key)
    return None if value is None else float(value)


def default_output_path(output_dir: Path, result: ResultLike) -> Path:
    physical = getattr(result, "physical_params", None)
    solver = getattr(result, "solver_params", None)
    parameters = getattr(result, "parameters", None)
    if physical is not None and solver is not None:
        amplitude = _field_E0_MV_per_cm(physical)
        if amplitude is None:
            raise ValueError("field metadata must contain E0_MV_per_cm to build default output path.")
        amplitude_tag = format_value_tag(amplitude)
        detuning = getattr(solver, "detuning_fs_inv", None)
        detuning_tag = "na" if detuning is None else format_value_tag(detuning)
    else:
        coupling = getattr(parameters, "coupling_matrix", None) or getattr(parameters, "dipole_matrix", ((0.0,),))
        amplitude_tag = format_value_tag(float(np.max(np.abs(np.asarray(coupling, dtype=np.complex128)))))
        detuning_tag = format_value_tag(getattr(parameters, "detuning", 0.0))
    return output_dir / f"comparison_E0_{amplitude_tag}_Delta_{detuning_tag}.png"


def save_parameter_summary(results: list[ResultLike], output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    data = [result.parameter_summary_dict() for result in results]
    output.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return output


def _append_results_csv_row(path: Path, header: list[str], row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            writer.writerow([row.get(column, "") for column in header])
        return

    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        existing_header = next(reader, None)
    if existing_header != header:
        raise ValueError(
            "results.csv header mismatch. Delete the old file or use another output directory.\n"
            f"existing: {existing_header}\n"
            f"expected: {header}"
        )

    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([row.get(column, "") for column in header])


def _safe_case_name(value: str) -> str:
    cleaned = []
    for char in str(value):
        cleaned.append(char if (char.isalnum() or char in ("-", "_")) else "_")
    return "".join(cleaned).strip("_") or "case"


def _case_name(result: ResultLike) -> str:
    physical = result.physical_params
    if physical is not None:
        field = _field_E0_MV_per_cm(physical)
        laser = _field_laser_energy_eV(physical, getattr(result, "solver_params", None))
        return _safe_case_name(
            f"{result.mode}_field_{format_value_tag(0.0 if field is None else field)}_"
            f"laser_{format_value_tag(0.0 if laser is None else laser)}"
        )
    return _safe_case_name(f"{result.mode}_N{result.dimension()}")


def _summary_row(case_name: str, result: ResultLike) -> dict[str, Any]:
    times = result.times_fs if result.times_fs is not None else result.times
    dimension = result.dimension()
    max_trace_error = result.max_trace_error()
    max_hermiticity_error = result.max_hermiticity_error()
    final_populations = ";".join(f"{float(value.real):.12g}" for value in result.populations()[-1])
    row: dict[str, Any] = {
        "case_name": case_name,
        "result_type": type(result).__name__,
        "mode": getattr(result, "mode", ""),
        "source_mode": getattr(result, "source_mode", ""),
        "time_start_fs": float(times[0]) if len(times) else "",
        "time_end_fs": float(times[-1]) if len(times) else "",
        "n_time_points": len(result.times),
        "dimension": dimension,
        "max_trace_error": max_trace_error,
        "max_hermiticity_error": max_hermiticity_error,
        "final_populations": final_populations,
    }
    physical = getattr(result, "physical_params", None)
    solver = getattr(result, "solver_params", None)
    row["field_MV_per_cm"] = "" if physical is None else (_field_E0_MV_per_cm(physical) or "")
    row["laser_energy_eV"] = "" if physical is None else (_field_laser_energy_eV(physical, solver) or "")
    detuning = None if solver is None else getattr(solver, "detuning_fs_inv", None)
    row["detuning_fs_inv"] = "" if detuning is None else detuning
    return row


def _trajectory_summary(result: ResultLike) -> dict[str, Any]:
    times = result.times_fs if result.times_fs is not None else result.times
    density = result.density_array()
    dimension = result.dimension()
    populations = result.populations().real
    final_populations = {f"rho_{index}{index}": float(populations[-1, index]) for index in range(dimension)}
    max_populations = {f"rho_{index}{index}": float(np.max(populations[:, index])) for index in range(dimension)}
    coherences: dict[str, Any] = {}
    for i in range(dimension):
        for j in range(i + 1, dimension):
            values = density[:, i, j]
            abs_values = np.abs(values)
            phase = np.angle(values)
            phase_unwrapped = np.unwrap(phase)
            if abs_values[-1] < 1e-8:
                final_phase = None
                final_phase_unwrapped = None
            else:
                final_phase = float(phase[-1])
                final_phase_unwrapped = float(phase_unwrapped[-1])
            coherences[f"rho_{i}{j}"] = {
                "final_abs": float(abs_values[-1]),
                "max_abs": float(np.max(abs_values)),
                "final_phase": final_phase,
                "final_phase_unwrapped": final_phase_unwrapped,
            }
    summary: dict[str, Any] = {
        "n_time_points": int(len(result.times)),
        "time_range_fs": [float(times[0]), float(times[-1])] if len(times) else [],
        "dimension": dimension,
        "final_populations": final_populations,
        "max_populations": max_populations,
        "final_coherences": coherences,
        "max_trace_error": result.max_trace_error(),
        "max_hermiticity_error": result.max_hermiticity_error(),
    }
    return summary


def _component_export_metadata(result: ResultLike) -> dict[str, Any]:
    dimension = result.dimension()
    return {
        "dimension": dimension,
        "component_indexing": "zero_based",
        "saved_populations": [f"rho_{index}{index}" for index in range(dimension)],
        "saved_coherences": [f"rho_{i}{j}" for i in range(dimension) for j in range(i + 1, dimension)],
        "saved_coherences_rule": "upper triangular off-diagonal elements only",
        "coherence_components": ["real", "imag", "abs", "phase", "phase_unwrapped"],
        "saved_observables": [],
        "observable_note": "Observable, polarization, FFT, and absorption-like quantities are analysis-layer outputs.",
        "density_npz": "full density matrix trajectory",
    }


def _channel_name(channel: Any) -> str | None:
    if isinstance(channel, dict):
        value = channel.get("name")
    else:
        value = getattr(channel, "name", None)
    return None if value is None else str(value)


def _channel_with_rate(channel: Any, solver_channels: tuple[dict[str, Any], ...] | list[dict[str, Any]]) -> dict[str, Any]:
    data = _json_safe(channel)
    if not isinstance(data, dict):
        return {"channel": data}
    if data.get("rate_fs_inv") is None:
        name = _channel_name(channel)
        for solver_channel in solver_channels:
            if name is not None and solver_channel.get("name") != name:
                continue
            if "rate_fs_inv" in solver_channel:
                data["rate_fs_inv"] = solver_channel["rate_fs_inv"]
                break
    return data


def _transition_table(physical: Any, solver: Any) -> list[dict[str, Any]]:
    if physical is None:
        return []
    energies_eV = np.asarray(getattr(physical, "energies_eV", ()), dtype=float)
    dipole_matrix = np.asarray(getattr(physical, "dipole_matrix_D", ()), dtype=np.complex128)
    n = int(len(energies_eV))
    table: list[dict[str, Any]] = []
    energies_fs_inv = None if solver is None else np.asarray(getattr(solver, "energies_fs_inv", ()), dtype=float)
    coupling_fs_inv = None if solver is None else np.asarray(getattr(solver, "coupling_matrix_fs_inv", ()), dtype=np.complex128)
    omega_L_fs_inv = None if solver is None else getattr(solver, "omega_L_fs_inv", None)
    laser_energy_eV = _field_laser_energy_eV(physical, solver)

    for i in range(n):
        for j in range(i + 1, n):
            energy_eV = float(energies_eV[j] - energies_eV[i])
            omega_fs_inv = None
            if energies_fs_inv is not None and energies_fs_inv.shape[0] > j:
                omega_fs_inv = float(energies_fs_inv[j] - energies_fs_inv[i])
            detuning_eV = None if laser_energy_eV is None else float(energy_eV - float(laser_energy_eV))
            detuning_fs_inv = None
            if omega_fs_inv is not None and omega_L_fs_inv is not None:
                detuning_fs_inv = float(omega_fs_inv - float(omega_L_fs_inv))
            dipole_D = None
            coupling = None
            if dipole_matrix.shape == (n, n):
                dipole_D = dipole_matrix[i, j]
            if coupling_fs_inv is not None and coupling_fs_inv.shape == (n, n):
                coupling = coupling_fs_inv[i, j]
            table.append(
                {
                    "from": i,
                    "to": j,
                    "label": f"{i}_to_{j}",
                    "energy_eV": energy_eV,
                    "omega_fs_inv": omega_fs_inv,
                    "laser_energy_eV": laser_energy_eV,
                    "detuning_eV": detuning_eV,
                    "detuning_fs_inv": detuning_fs_inv,
                    "dipole_D": dipole_D,
                    "coupling_fs_inv": coupling,
                    "dipole_coupled": bool(dipole_D is not None and abs(dipole_D) > 0),
                }
            )
    return table


def _system_metadata(physical: Any, solver: Any, result: ResultLike) -> dict[str, Any] | None:
    if physical is None:
        return None
    system = {
        "basis": physical.basis,
        "dimension": result.dimension(),
        "energies_eV": physical.energies_eV,
        "dipole_matrix_D": physical.dipole_matrix_D,
        "transition_table": _transition_table(physical, solver),
    }
    if result.dimension() == 2:
        system["energy_gap_eV"] = physical.energy_gap_eV
        laser_energy_eV = _field_laser_energy_eV(physical, solver)
        system["detuning_eV"] = None if laser_energy_eV is None else physical.energy_gap_eV - laser_energy_eV
    return system


def _field_metadata(result: ResultLike, physical: Any, solver: Any) -> dict[str, Any] | None:
    if physical is None:
        return None
    envelope = _field_envelope(physical)
    field_payload = _field_rebuild_metadata(result, physical)
    field_class = None if field_payload is None else field_payload.get("class")
    if field_class is None:
        field_class = "GaussianCarrierFieldPhysical" if envelope == "gaussian" else "CarrierFieldPhysical"

    field_E0 = _field_E0_MV_per_cm(physical)
    parameters = {
        "E0_MV_per_cm": field_E0,
        "peak_E_MV_per_cm": _field_metadata_value(physical, "peak_E_MV_per_cm"),
        "omega_L_fs_inv": _field_metadata_value(physical, "omega_L_fs_inv"),
        "laser_energy_eV": _field_laser_energy_eV(physical, solver),
        "phase_rad": _field_metadata_value(physical, "phase_rad") or 0.0,
        "envelope": envelope,
    }
    if envelope == "gaussian":
        parameters["pulse_center_fs"] = _field_pulse_value(physical, "pulse_center_fs", "center_fs")
        parameters["pulse_sigma_fs"] = _field_pulse_value(physical, "pulse_sigma_fs", "sigma_fs")
    return {
        "class": field_class,
        "expression": "E(t) = 2 E0 f(t) cos(omega_L t + phase)",
        "parameters": parameters,
        "units": {"field": "MV/cm", "time": "fs"},
        "amplitude_convention": "E0_MV_per_cm is E0 in E(t) = 2 E0 f(t) cos(omega_L t + phase).",
        "rebuildable": bool(field_payload is not None and field_payload.get("rebuildable", True)),
        "debug_details": "debug_meta.json",
    }


def _dissipation_metadata(physical: Any, solver: Any) -> dict[str, Any] | None:
    if physical is None:
        return None
    relaxation_solver = () if solver is None else solver.relaxation_channels_fs_inv
    dephasing_solver = () if solver is None else solver.pure_dephasing_channels_fs_inv
    return {
        "relaxation_channels": [
            _channel_with_rate(channel, relaxation_solver)
            for channel in getattr(physical, "relaxation_channels", ())
        ],
        "pure_dephasing_channels": [
            _channel_with_rate(channel, dephasing_solver)
            for channel in getattr(physical, "pure_dephasing_channels", ())
        ],
    }


def _time_grid_metadata(result: ResultLike, physical: Any) -> dict[str, Any]:
    times = result.times_fs if result.times_fs is not None else result.times
    if physical is not None:
        t_start = physical.t_start_fs
        t_end = physical.t_end_fs
        dt = physical.dt_fs
    elif len(times):
        t_start = float(times[0])
        t_end = float(times[-1])
        dt = float(np.diff(times)[0]) if len(times) > 1 else None
    else:
        t_start = t_end = dt = None
    return {
        "t_start_fs": t_start,
        "t_end_fs": t_end,
        "dt_fs": dt,
        "n_time_points": int(len(result.times)),
        "time_axis_unit": "fs",
    }


def _solver_metadata(result: ResultLike) -> dict[str, Any]:
    data = _solver_representation_metadata(result)
    if data.get("mode") == "lab_exact":
        data["interaction"] = "H_int(t) = -mu E(t)"
    return data


def _sanity_summary(result: ResultLike) -> dict[str, Any]:
    checks = getattr(result, "sanity_checks", {}) or {}
    trace = checks.get("trace_error_small", {}) if isinstance(checks, dict) else {}
    herm = checks.get("hermiticity_error_small", {}) if isinstance(checks, dict) else {}
    max_trace = result.max_trace_error()
    max_hermiticity = result.max_hermiticity_error()
    return {
        "trace_error_small": bool(trace.get("passed", max_trace < 1e-8)),
        "hermiticity_error_small": bool(herm.get("passed", max_hermiticity < 1e-8)),
        "max_trace_error": max_trace,
        "max_hermiticity_error": max_hermiticity,
        "details": "debug_meta.json",
    }


def _field_envelope(physical: Any) -> str:
    envelope = _field_metadata_value(physical, "envelope")
    return "unknown" if envelope is None else str(envelope)


def _field_rebuild_metadata(result: ResultLike, physical: Any) -> dict[str, Any] | None:
    """Return a JSON-safe, rebuild-oriented physical field payload for meta.json."""

    if physical is None:
        return None

    explicit_field = getattr(physical, "field", None)
    if explicit_field is not None and hasattr(explicit_field, "to_dict"):
        return _json_safe(explicit_field.to_dict())

    drive_dict = getattr(result, "drive_dict", None)
    if isinstance(drive_dict, dict):
        source_field = drive_dict.get("source_field")
        if source_field is not None:
            return _json_safe(source_field)

    return None


def _array_summary(value: Any) -> dict[str, Any]:
    """Summarize a long 1D numeric array for JSON metadata."""

    array = np.asarray(value, dtype=float).reshape(-1)
    if array.size == 0:
        return {"n_points": 0, "start": None, "end": None, "dt": None}
    if array.size == 1:
        return {
            "n_points": 1,
            "start": float(array[0]),
            "end": float(array[0]),
            "dt": None,
        }
    diffs = np.diff(array)
    first_dt = float(diffs[0])
    uniform = bool(np.allclose(diffs, first_dt, rtol=1e-9, atol=1e-12))
    return {
        "n_points": int(array.size),
        "start": float(array[0]),
        "end": float(array[-1]),
        "dt": first_dt if uniform else None,
        "is_uniform": uniform,
    }


def _replace_long_axis_with_summary(container: dict[str, Any], key: str, summary_key: str) -> None:
    """Replace a long array entry in-place with a compact summary."""

    if not isinstance(container, dict) or key not in container:
        return
    value = container.pop(key)
    try:
        container[summary_key] = _array_summary(value)
    except (TypeError, ValueError):
        container[summary_key] = {"summary_failed": True, "repr": repr(value)}


_TIME_AXIS_SUMMARY_KEYS = {
    "tlist": "tlist_code_summary",
    "tlist_code": "tlist_code_summary",
    "time_code": "time_code_summary",
    "times_code": "time_code_summary",
    "times": "time_code_summary",
    "time_fs": "time_fs_summary",
    "times_fs": "time_fs_summary",
}


def _compact_time_axes_recursive(value: Any) -> Any:
    """Replace full time-axis arrays in debug metadata with compact summaries.

    This keeps debug_meta.json readable and avoids duplicating time axes that are
    already stored losslessly in density.npz. Only known time-axis keys are
    compacted; other numeric arrays remain available for debug metadata.
    """

    if isinstance(value, dict):
        compacted: dict[str, Any] = {}
        for key, item in value.items():
            summary_key = _TIME_AXIS_SUMMARY_KEYS.get(str(key))
            if summary_key is not None and isinstance(item, (list, tuple, np.ndarray)):
                try:
                    compacted[summary_key] = _array_summary(item)
                except (TypeError, ValueError):
                    compacted[summary_key] = {"summary_failed": True, "repr": repr(item)}
                continue
            compacted[str(key)] = _compact_time_axes_recursive(item)
        return compacted
    if isinstance(value, list):
        return [_compact_time_axes_recursive(item) for item in value]
    return value


def _compact_debug_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Keep debug_meta.json useful without duplicating full time axes from density.npz."""

    return _compact_time_axes_recursive(metadata)


_MATTER_METADATA_KEYS = {"user_metadata", "transitions_eV", "dipoles_D", "system", "example"}


def _clean_field_payload_metadata(value: Any) -> Any:
    if isinstance(value, dict):
        cleaned: dict[str, Any] = {}
        for key, item in value.items():
            if key == "metadata" and isinstance(item, dict):
                cleaned_metadata = {
                    str(meta_key): _clean_field_payload_metadata(meta_value)
                    for meta_key, meta_value in item.items()
                    if str(meta_key) not in _MATTER_METADATA_KEYS
                }
                cleaned[key] = cleaned_metadata
                continue
            if str(key) in _MATTER_METADATA_KEYS:
                continue
            cleaned[str(key)] = _clean_field_payload_metadata(item)
        return cleaned
    if isinstance(value, list):
        return [_clean_field_payload_metadata(item) for item in value]
    return value


def _clean_debug_field_metadata(value: Any) -> Any:
    if isinstance(value, dict):
        cleaned: dict[str, Any] = {}
        for key, item in value.items():
            if key == "source_field":
                cleaned[key] = _clean_field_payload_metadata(item)
            elif key == "field" and isinstance(item, dict):
                cleaned[key] = _clean_field_payload_metadata(item)
            else:
                cleaned[str(key)] = _clean_debug_field_metadata(item)
        return cleaned
    if isinstance(value, list):
        return [_clean_debug_field_metadata(item) for item in value]
    return value


def _input_field_metadata(physical: Any, solver: Any) -> dict[str, Any] | None:
    if physical is None:
        return None
    envelope = _field_envelope(physical)
    field_metadata = _physical_field_payload(physical)
    if field_metadata is not None:
        envelope = field_metadata.get("envelope", envelope)
    field_class = None
    if field_metadata is not None:
        field_class = field_metadata.get("class")
    if field_class is None:
        field_class = "GaussianCarrierFieldPhysical" if envelope == "gaussian" else "CarrierFieldPhysical"
    data: dict[str, Any] = {
        "description": "Physical lab-frame optical field.",
        "class": field_class,
        "expression": "E(t) = 2 E0 f(t) cos(omega_L t + phase)",
        "E0_MV_per_cm": _field_E0_MV_per_cm(physical),
        "peak_E_MV_per_cm": _field_metadata_value(physical, "peak_E_MV_per_cm"),
        "omega_L_fs_inv": _field_metadata_value(physical, "omega_L_fs_inv"),
        "laser_energy_eV": _field_laser_energy_eV(physical, solver),
        "phase_rad": _field_metadata_value(physical, "phase_rad") or 0.0,
        "envelope": envelope,
        "field_unit": "MV/cm",
        "time_unit": "fs",
        "amplitude_convention": "E0_MV_per_cm is E0 in E(t) = 2 E0 f(t) cos(omega_L t + phase).",
    }
    if getattr(physical, "input_description", None) is not None:
        data["user_description"] = physical.input_description
    if getattr(physical, "input_metadata", None) is not None:
        data["user_metadata"] = dict(physical.input_metadata)
    if field_metadata is not None:
        data["field_metadata"] = field_metadata
    if envelope == "gaussian":
        data["pulse_center_fs"] = _field_pulse_value(physical, "pulse_center_fs", "center_fs")
        data["pulse_sigma_fs"] = _field_pulse_value(physical, "pulse_sigma_fs", "sigma_fs")
    return data


def _solver_representation_metadata(result: ResultLike) -> dict[str, Any]:
    mode = getattr(result, "mode", None)
    if mode == "lab_exact":
        return {
            "mode": "lab_exact",
            "description": "Direct lab-frame propagation with the full time-dependent optical carrier.",
            "hamiltonian_type": "time-dependent lab-frame Hamiltonian",
            "uses_lab_field_directly": True,
            "uses_rwa_drive": False,
            "uses_rotating_transform": False,
            "carrier_retained": True,
            "counter_rotating_terms_retained": True,
            "rwa_approximation_applied": False,
            "independently_solved_by_mesolve": True,
        }
    if mode == "rotating_view":
        return {
            "mode": "rotating_view",
            "description": "Derived view obtained by transforming lab-frame density matrices into a rotating frame.",
            "hamiltonian_type": "not independently solved",
            "uses_lab_field_directly": False,
            "uses_rwa_drive": False,
            "uses_rotating_transform": True,
            "source_mode": getattr(result, "source_mode", None) or "lab_exact",
            "carrier_retained": "not_applicable",
            "counter_rotating_terms_retained": "inherited_from_lab_exact",
            "rwa_approximation_applied": False,
            "independently_solved_by_mesolve": False,
        }
    if mode == "rwa":
        return {
            "mode": "rwa",
            "description": "Propagation with the effective RWA Hamiltonian.",
            "hamiltonian_type": "RWA effective Hamiltonian",
            "uses_lab_field_directly": False,
            "uses_rwa_drive": True,
            "uses_rotating_transform": False,
            "carrier_retained": False,
            "counter_rotating_terms_retained": False,
            "rwa_approximation_applied": True,
            "independently_solved_by_mesolve": True,
        }
    return {
        "mode": mode,
        "description": "Simulation representation is not specialized for this mode.",
        "independently_solved_by_mesolve": None,
    }


def _lab_frame_solver_metadata(result: ResultLike, physical: Any, solver: Any) -> dict[str, Any] | None:
    if getattr(result, "mode", None) != "lab_exact":
        return None
    envelope = _field_envelope(physical) if physical is not None else "unknown"
    field_class = "GaussianCarrierFieldPhysical" if envelope == "gaussian" else "CarrierFieldPhysical"
    if getattr(physical, "field", None) is not None:
        field_class = physical.field.__class__.__name__
    return {
        "description": "Direct lab-frame solver using the physical optical carrier.",
        "field_class": field_class,
        "field_expression": "E(t) = 2 E0 f(t) cos(omega_L t + phase)",
        "interaction": "H_int(t) = -mu E(t)",
        "carrier_retained": True,
        "counter_rotating_terms_retained": True,
        "hamiltonian_note": "The full lab-frame optical carrier is retained in the time-dependent Hamiltonian.",
    }


def _rotating_transform_metadata(result: ResultLike, solver: Any) -> dict[str, Any] | None:
    if getattr(result, "mode", None) != "rotating_view":
        return None
    return {
        "description": "Post-processing transformation applied to lab-frame density matrix trajectory.",
        "source_result_mode": getattr(result, "source_mode", None) or "lab_exact",
        "reference_frequency_fs_inv": None if solver is None else solver.omega_L_fs_inv,
        "reference_frequency_source": "laser_energy_eV",
        "transform_type": "density matrix unitary frame transformation",
        "independently_solved_by_mesolve": False,
        "rwa_approximation_applied": False,
    }


def _input_drive_metadata(result: ResultLike, physical: Any, solver: Any, parameters: Any) -> dict[str, Any] | None:
    mode = getattr(result, "mode", None)
    if mode != "rwa":
        return {
            "description": f"No RWA drive is used in {mode} mode.",
            "uses_rwa_drive": False,
        }
    envelope = _field_envelope(physical) if physical is not None else "constant"
    drive_class = "gaussian_rwa_envelope" if envelope == "gaussian" else "constant_rwa_envelope"
    amplitude = None if solver is None else solver.rabi_fs_inv
    if envelope == "gaussian":
        drive_expr = (
            None
            if solver is None
            else (
                f"g(t) = {solver.rabi_fs_inv:.6g} fs^-1 * "
                f"exp[-(t_fs - {solver.pulse_center_fs:.6g})^2 / (2 * {solver.pulse_sigma_fs:.6g}^2)]"
            )
        )
    else:
        drive_expr = None if amplitude is None else f"g(t) = {amplitude:.6g} fs^-1"
    return {
        "description": "Effective slow drive entering the RWA Hamiltonian.",
        "class": drive_class,
        "drive_name": getattr(result, "drive_name", None),
        "drive_symbol": "g(t)",
        "expression": "g(t) = mu E0 f(t) / hbar",
        "amplitude_fs_inv": None if solver is None else solver.rabi_fs_inv,
        "source": "derived from dipole_matrix_D and field reference_MV_per_cm",
        "domain": "RWA",
        "drive_unit_physical": "fs^-1",
        "envelope": envelope,
        "amplitude_convention": "input_drive is the slow RWA coupling after removing the optical carrier.",
        "drive_expr": drive_expr,
    }


def _human_metadata(
    result: ResultLike,
    *,
    example_name: str | None = None,
    condition_name: str | None = None,
    case_name: str | None = None,
    output_files: dict[str, str] | None = None,
) -> dict[str, Any]:
    physical = getattr(result, "physical_params", None)
    solver = getattr(result, "solver_params", None)
    meta: dict[str, Any] = {
        "result_type": type(result).__name__,
        "example_name": example_name,
        "condition_name": condition_name,
        "case_name": case_name,
        "mode": getattr(result, "mode", None),
        "source_mode": getattr(result, "source_mode", None),
    }

    if physical is not None:
        meta["user_input"] = {
            "description": physical.input_description,
            "metadata": physical.input_metadata,
        }
    else:
        meta["user_input"] = None

    meta["system"] = _system_metadata(physical, solver, result)
    meta["field"] = _field_metadata(result, physical, solver)
    meta["dissipation"] = _dissipation_metadata(physical, solver)
    meta["time_grid"] = _time_grid_metadata(result, physical)
    meta["solver"] = _solver_metadata(result)
    if getattr(result, "mode", None) == "rotating_view":
        meta["rotating_transform"] = _rotating_transform_metadata(result, solver)
    if getattr(result, "mode", None) == "rwa":
        meta["input_drive"] = _input_drive_metadata(result, physical, solver, getattr(result, "parameters", None))

    meta["trajectory_summary"] = _trajectory_summary(result)
    meta["sanity_summary"] = _sanity_summary(result)
    meta["component_export"] = _component_export_metadata(result)
    meta["output_files"] = output_files or {}
    return _json_safe(meta)


def _debug_metadata(
    result: ResultLike,
    *,
    example_name: str | None = None,
    condition_name: str | None = None,
    case_name: str | None = None,
) -> dict[str, Any]:
    metadata = result.metadata_dict()
    metadata["example_name"] = example_name
    metadata["condition_name"] = condition_name
    metadata["case_name"] = case_name
    return _compact_debug_metadata(_clean_debug_field_metadata(metadata))


def _relative_output_files(case_dir: Path, written: dict[str, Path]) -> dict[str, str]:
    names = {
        "density_npz": "density",
        "components_csv": "components",
        "populations_csv": "populations",
        "selected_elements_csv": "selected_elements",
        "preview": "preview",
        "full": "full",
        "debug_meta": "debug_metadata",
        "meta": "metadata",
    }
    output_files: dict[str, str] = {}
    for key, label in names.items():
        path = written.get(key)
        if path is not None:
            output_files[label] = path.relative_to(case_dir).as_posix()
    component_figures: dict[str, str] = {}
    for key, path in written.items():
        if not str(key).startswith("component_") or path is None:
            continue
        stem = str(key).removeprefix("component_")
        component_figures[stem] = path.relative_to(case_dir).as_posix()
    if component_figures:
        output_files["component_figures"] = component_figures
        output_files["component_figures_dir"] = "figs/component"
    return output_files


def _write_metadata_files(
    result: ResultLike,
    case_dir: Path,
    written: dict[str, Path],
    *,
    example_name: str | None = None,
    condition_name: str | None = None,
    case_name: str | None = None,
    save_human_meta: bool = True,
    save_debug_meta: bool = True,
) -> dict[str, Path]:
    metadata_paths: dict[str, Path] = {}
    debug_path = case_dir / "debug_meta.json"
    meta_path = case_dir / "meta.json"
    metadata_outputs = {**written}
    if save_debug_meta:
        metadata_outputs["debug_meta"] = debug_path
    output_files = _relative_output_files(case_dir, metadata_outputs)

    if save_debug_meta:
        debug_path.write_text(
            json.dumps(
                _debug_metadata(
                    result,
                    example_name=example_name,
                    condition_name=condition_name,
                    case_name=case_name,
                ),
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        metadata_paths["debug_meta"] = debug_path

    if save_human_meta:
        meta_path.write_text(
            json.dumps(
                _human_metadata(
                    result,
                    example_name=example_name,
                    condition_name=condition_name,
                    case_name=case_name,
                    output_files=output_files,
                ),
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        metadata_paths["meta"] = meta_path

    return metadata_paths


def save_figure(fig, output_path: str | Path, dpi: int = 120) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=int(dpi))
    return output


def save_result_data(
    result: ResultLike,
    output_dir: str | Path,
    *,
    save_npz: bool = True,
    save_csv: bool = True,
    save_populations_csv: bool = False,
    save_json: bool = True,
    save_human_meta: bool = True,
    save_debug_meta: bool = True,
    example_name: str | None = None,
    condition_name: str | None = None,
    case_name: str | None = None,
    selected_elements: dict[str, tuple[int, int]] | None = None,
) -> dict[str, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    if save_npz:
        npz_path = output / "density.npz"
        np.savez_compressed(npz_path, **result.to_npz_dict())
        written["density_npz"] = npz_path

    if save_csv:
        components_path = output / "components.csv"
        result.components_dataframe().to_csv(components_path, index=False)
        written["components_csv"] = components_path

        if save_populations_csv:
            populations_path = output / "populations.csv"
            result.populations_dataframe().to_csv(populations_path, index=False)
            written["populations_csv"] = populations_path

        if selected_elements:
            selected_path = output / "selected_elements.csv"
            result.selected_elements_dataframe(selected_elements).to_csv(selected_path, index=False)
            written["selected_elements_csv"] = selected_path

    if save_json:
        written.update(
            _write_metadata_files(
                result,
                output.parent,
                written,
                example_name=example_name,
                condition_name=condition_name,
                case_name=case_name,
                save_human_meta=save_human_meta,
                save_debug_meta=save_debug_meta,
            )
        )

    return written


def save_results_components_long(results: list[ResultLike], output_path: str | Path) -> Path:
    if not results:
        raise ValueError("results must not be empty.")
    frames = []
    for result in results:
        frame = result.components_dataframe().copy()
        frame.insert(0, "mode", result.mode)
        frames.append(frame)

    pd = __import__("pandas")
    combined = pd.concat(frames, ignore_index=True)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(path, index=False)
    return path

def save_result_case(
    result: ResultLike,
    output_dir: str | Path,
    *,
    output_data: bool = True,
    output_preview: bool = False,
    fig=None,
    preview_fig=None,
    preview_dpi: int = 120,
    full_fig=None,
    full_dpi: int = 300,
    save_npz: bool = True,
    save_csv: bool = True,
    save_populations_csv: bool = False,
    save_json: bool = True,
    save_human_meta: bool = True,
    save_debug_meta: bool = True,
    example_name: str | None = None,
    condition_name: str | None = None,
    case_name: str | None = None,
    selected_elements: dict[str, tuple[int, int]] | None = None,
    append_results_csv: bool = True,
    preview_population_axis_map: dict[int, str | int] | None = None,
    preview_population_split_threshold: float = 0.1,
    preview_component_pairs: list[tuple[int, int]] | tuple[tuple[int, int], ...] | None = None,
    preview_max_pairs: int | None = 6,
    preview_display_code_unit: bool = False,
) -> dict[str, Path]:
    root = Path(output_dir)
    name = _safe_case_name(case_name) if case_name is not None else _case_name(result)
    case_dir = root / name
    data_dir = case_dir / "data"
    figs_dir = case_dir / "figs"
    component_dir = figs_dir / "component"
    case_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {"case_dir": case_dir}

    if output_data:
        written.update(
            save_result_data(
                result,
                data_dir,
                save_npz=save_npz,
                save_csv=save_csv,
                save_populations_csv=save_populations_csv,
                save_json=False,
                selected_elements=selected_elements,
            )
        )

    generated_preview = False
    generated_component_figs = False

    local_preview_fig = fig if preview_fig is None else preview_fig
    local_component_figs = []

    if output_preview:
        from .plotting import (
            build_component_figures,
            build_preview_figure,
        )

        if local_preview_fig is None:
            local_preview_fig, _axes = build_preview_figure(
                result,
                coherences=preview_component_pairs,
                display_code_unit=preview_display_code_unit,
                max_pairs=preview_max_pairs,
                population_axis_map=preview_population_axis_map,
                split_population_axes_threshold=preview_population_split_threshold,
            )
            generated_preview = True

        local_component_figs = build_component_figures(
            result,
            coherences=preview_component_pairs,
            max_pairs=preview_max_pairs,
        )
        generated_component_figs = True

    if output_preview and local_preview_fig is not None:
        preview_path = figs_dir / "preview.png"
        written["preview"] = save_figure(local_preview_fig, preview_path, dpi=preview_dpi)
        if generated_preview:
            import matplotlib.pyplot as plt
            plt.close(local_preview_fig)

    if output_preview and local_component_figs:
        component_dir.mkdir(parents=True, exist_ok=True)
        import matplotlib.pyplot as plt

        for stem, comp_fig in local_component_figs:
            comp_path = component_dir / f"{stem}.png"
            written[f"component_{stem}"] = save_figure(comp_fig, comp_path, dpi=preview_dpi)
            if generated_component_figs:
                plt.close(comp_fig)

    if full_fig is not None:
        full_path = figs_dir / "full.png"
        written["full"] = save_figure(full_fig, full_path, dpi=full_dpi)

    if save_json:
        written.update(
            _write_metadata_files(
                result,
                case_dir,
                written,
                example_name=example_name,
                condition_name=condition_name,
                case_name=name,
                save_human_meta=save_human_meta,
                save_debug_meta=save_debug_meta,
            )
        )

    if append_results_csv:
        header = [
            "case_name",
            "result_type",
            "mode",
            "source_mode",
            "time_start_fs",
            "time_end_fs",
            "n_time_points",
            "dimension",
            "field_MV_per_cm",
            "laser_energy_eV",
            "detuning_fs_inv",
            "max_trace_error",
            "max_hermiticity_error",
            "final_populations",
        ]
        row = _summary_row(name, result)
        _append_results_csv_row(root / "results.csv", header, row)

    return written

@dataclass
class QuantumResultIO:
    outdir: str
    results_csv_name: str = "results.csv"

    def save_case(self, result: ResultLike, **kwargs) -> dict[str, Path]:
        return save_result_case(result, self.outdir, **kwargs)


ResultManager = QuantumResultIO


__all__ = [
    "QuantumResultIO",
    "ResultManager",
    "default_output_path",
    "save_parameter_summary",
    "save_result_data",
    "save_results_components_long",
    "save_figure",
    "save_result_case",
]
