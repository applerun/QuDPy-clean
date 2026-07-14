#!/usr/bin/env python3
"""Benchmark stable core mainline runtime, disk export, and plotting overhead.

这是开发 benchmark，不是正式 example。它只使用当前稳定 core 主线：

    NLevelPhysicalParams -> run_case -> DynamicsResult

并测量：

    solver runtime
    dataframe construction
    checkpoint / npz / csv / json 落盘
    matplotlib figure build / png save

输出默认写到 /tmp 下的独立目录，避免污染仓库。
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import platform
import statistics
import sys
import time
from dataclasses import dataclass, field as dataclass_field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from qudpy_sjh.utils.core import NLevelPhysicalParams, ParaNormalizer, run_case  # noqa: E402
from qudpy_sjh.utils.fields.carrier_envelope import make_gaussian_carrier_envelope_field  # noqa: E402
from qudpy_sjh.utils.io import save_figure  # noqa: E402
from qudpy_sjh.utils.plotting import plot_coherences, plot_populations  # noqa: E402


@dataclass(frozen=True)
class BenchmarkCase:
    """一个 core 主线 benchmark case。"""

    name: str
    t_start_fs: float
    t_end_fs: float
    dt_fs: float
    E0_MV_per_cm: float
    sigma_fs: float

    @property
    def n_time_points(self) -> int:
        return int(round((self.t_end_fs - self.t_start_fs) / self.dt_fs)) + 1


@dataclass
class Measurement:
    """单条 benchmark 测量记录。"""

    section: str
    name: str
    elapsed_s: float
    size_bytes: int | None = None
    metadata: dict[str, Any] = dataclass_field(default_factory=dict)

    def to_row(self) -> dict[str, Any]:
        return {
            "section": self.section,
            "name": self.name,
            "elapsed_s": f"{self.elapsed_s:.9f}",
            "size_bytes": "" if self.size_bytes is None else int(self.size_bytes),
            "metadata_json": json.dumps(self.metadata, ensure_ascii=False, sort_keys=True),
        }


def _measure(section: str, name: str, func: Callable[[], Any], *, metadata: dict[str, Any] | None = None) -> tuple[Any, Measurement]:
    start = time.perf_counter()
    value = func()
    elapsed = time.perf_counter() - start
    return value, Measurement(section=section, name=name, elapsed_s=elapsed, metadata=dict(metadata or {}))


def _file_size(path: Path) -> int:
    if not path.exists():
        raise FileNotFoundError(path)
    return int(path.stat().st_size)


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _environment_summary() -> dict[str, Any]:
    summary: dict[str, Any] = {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }
    for package_name in ("numpy", "scipy", "qutip", "pandas", "matplotlib"):
        try:
            module = __import__(package_name)
            summary[f"{package_name}_version"] = getattr(module, "__version__", "unknown")
        except ImportError:
            summary[f"{package_name}_version"] = None
    return summary


def _default_cases(profile: str) -> tuple[BenchmarkCase, ...]:
    if profile == "quick":
        return (
            BenchmarkCase(
                name="quick_two_level_201",
                t_start_fs=-80.0,
                t_end_fs=120.0,
                dt_fs=1.0,
                E0_MV_per_cm=0.01,
                sigma_fs=12.0,
            ),
        )
    if profile == "standard":
        return (
            BenchmarkCase(
                name="standard_two_level_401",
                t_start_fs=-120.0,
                t_end_fs=280.0,
                dt_fs=1.0,
                E0_MV_per_cm=0.01,
                sigma_fs=18.0,
            ),
            BenchmarkCase(
                name="standard_two_level_801",
                t_start_fs=-200.0,
                t_end_fs=600.0,
                dt_fs=1.0,
                E0_MV_per_cm=0.01,
                sigma_fs=25.0,
            ),
        )
    raise ValueError("profile must be 'quick' or 'standard'.")


def _make_params(case: BenchmarkCase) -> NLevelPhysicalParams:
    field = make_gaussian_carrier_envelope_field(
        E0_MV_per_cm=case.E0_MV_per_cm,
        laser_energy_eV=1.55,
        center_fs=0.0,
        sigma_fs=case.sigma_fs,
        phase_rad=0.0,
        name=f"{case.name}_field",
    )
    return NLevelPhysicalParams(
        basis=("g", "e"),
        energies_eV=(0.0, 1.55),
        dipole_matrix_D=((0.0, 5.0), (5.0, 0.0)),
        t_start_fs=case.t_start_fs,
        t_end_fs=case.t_end_fs,
        dt_fs=case.dt_fs,
        field=field,
        solver_mode="lab_exact",
        input_description="Core efficiency benchmark case.",
        input_metadata={"benchmark_case": case.name},
    )


def _run_solver_benchmark(case: BenchmarkCase, *, repeats: int) -> tuple[Any, list[Measurement]]:
    params = _make_params(case)
    normalizer = ParaNormalizer()
    results = []
    measurements: list[Measurement] = []
    for index in range(repeats):
        result, item = _measure(
            "solver",
            case.name,
            lambda: run_case(params, normalizer=normalizer),
            metadata={"repeat": index, "n_time_points": case.n_time_points},
        )
        results.append(result)
        measurements.append(item)
    return results[-1], measurements


def _benchmark_dataframes(result, case_dir: Path) -> list[Measurement]:
    measurements: list[Measurement] = []
    components, item = _measure("dataframe", "components_dataframe", result.components_dataframe)
    item.metadata["shape"] = list(components.shape)
    measurements.append(item)

    populations, item = _measure("dataframe", "populations_dataframe", result.populations_dataframe)
    item.metadata["shape"] = list(populations.shape)
    measurements.append(item)

    # 复用已经构造的数据帧，避免 CSV 落盘 benchmark 混入 DataFrame 构造时间。
    csv_dir = case_dir / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    components_path = csv_dir / "components.csv"
    populations_path = csv_dir / "populations.csv"

    _value, item = _measure("disk", "components_csv_write", lambda: components.to_csv(components_path, index=False))
    item.size_bytes = _file_size(components_path)
    measurements.append(item)

    _value, item = _measure("disk", "populations_csv_write", lambda: populations.to_csv(populations_path, index=False))
    item.size_bytes = _file_size(populations_path)
    measurements.append(item)
    return measurements


def _benchmark_disk(result, case_dir: Path) -> list[Measurement]:
    measurements: list[Measurement] = []
    disk_dir = case_dir / "disk"
    disk_dir.mkdir(parents=True, exist_ok=True)

    ckp_path = disk_dir / "result.ckp"
    _value, item = _measure("disk", "checkpoint_pickle_save", lambda: result.save_ckp(ckp_path))
    item.size_bytes = _file_size(ckp_path)
    measurements.append(item)

    npz_path = disk_dir / "density.npz"
    _value, item = _measure("disk", "density_npz_compressed_write", lambda: np.savez_compressed(npz_path, **result.to_npz_dict()))
    item.size_bytes = _file_size(npz_path)
    measurements.append(item)

    metadata_path = disk_dir / "metadata.json"
    _value, item = _measure("disk", "metadata_json_write", lambda: _write_json(metadata_path, result.metadata_dict()))
    item.size_bytes = _file_size(metadata_path)
    measurements.append(item)
    return measurements


def _benchmark_plotting(result, case_dir: Path) -> list[Measurement]:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    measurements: list[Measurement] = []
    plot_dir = case_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    fig_pop, item = _measure("plot", "plot_populations_build", lambda: plot_populations(result))
    measurements.append(item)
    pop_path = plot_dir / "populations.png"
    _value, item = _measure("plot", "plot_populations_png_save", lambda: save_figure(fig_pop[0], pop_path, dpi=120))
    item.size_bytes = _file_size(pop_path)
    measurements.append(item)
    plt.close(fig_pop[0])

    fig_coh, item = _measure("plot", "plot_coherences_build", lambda: plot_coherences(result))
    measurements.append(item)
    coh_path = plot_dir / "coherences.png"
    _value, item = _measure("plot", "plot_coherences_png_save", lambda: save_figure(fig_coh[0], coh_path, dpi=120))
    item.size_bytes = _file_size(coh_path)
    measurements.append(item)
    plt.close(fig_coh[0])
    return measurements


def _summary_by_name(measurements: list[Measurement]) -> dict[str, Any]:
    grouped: dict[str, list[float]] = {}
    for item in measurements:
        grouped.setdefault(f"{item.section}.{item.name}", []).append(float(item.elapsed_s))
    return {
        name: {
            "n": len(values),
            "min_s": min(values),
            "median_s": statistics.median(values),
            "max_s": max(values),
        }
        for name, values in sorted(grouped.items())
    }


def _write_measurements_csv(path: Path, measurements: list[Measurement]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = ["section", "name", "elapsed_s", "size_bytes", "metadata_json"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        for item in measurements:
            writer.writerow(item.to_row())
    return path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark stable QuDPy core runtime, disk IO, and plotting.")
    parser.add_argument("--profile", choices=("quick", "standard"), default="quick")
    parser.add_argument("--repeats", type=int, default=3, help="Solver repeats per case.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Benchmark output directory. Default: /tmp/qudpy_core_benchmark_<timestamp>",
    )
    parser.add_argument("--no-plots", action="store_true", help="Skip matplotlib build/save benchmark.")
    parser.add_argument("--no-disk", action="store_true", help="Skip checkpoint/npz/json/csv disk benchmark.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.repeats < 1:
        raise ValueError("--repeats must be >= 1.")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or Path("/tmp") / f"qudpy_core_benchmark_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    measurements: list[Measurement] = []
    cases = _default_cases(args.profile)
    case_summaries: dict[str, Any] = {}

    for case in cases:
        case_dir = output_dir / case.name
        case_dir.mkdir(parents=True, exist_ok=True)
        result, solver_measurements = _run_solver_benchmark(case, repeats=args.repeats)
        measurements.extend(solver_measurements)
        case_summaries[case.name] = {
            "n_time_points": case.n_time_points,
            "dimension": result.dimension(),
            "max_trace_error": result.max_trace_error(),
            "max_hermiticity_error": result.max_hermiticity_error(),
        }
        if not args.no_disk:
            measurements.extend(_benchmark_dataframes(result, case_dir))
            measurements.extend(_benchmark_disk(result, case_dir))
        if not args.no_plots:
            measurements.extend(_benchmark_plotting(result, case_dir))
        gc.collect()

    measurements_csv = _write_measurements_csv(output_dir / "benchmark_measurements.csv", measurements)
    summary = {
        "benchmark": "qudpy_core_efficiency",
        "created_at": timestamp,
        "profile": args.profile,
        "repeats": int(args.repeats),
        "output_dir": str(output_dir),
        "environment": _environment_summary(),
        "cases": case_summaries,
        "summary_by_name": _summary_by_name(measurements),
        "measurements_csv": str(measurements_csv),
    }
    summary_path = _write_json(output_dir / "benchmark_summary.json", summary)

    print("qudpy_core_benchmark_ok")
    print(f"output_dir: {output_dir}")
    print(f"summary_json: {summary_path}")
    print(f"measurements_csv: {measurements_csv}")
    for name, item in summary["summary_by_name"].items():
        print(f"{name}: median={item['median_s']:.6f}s n={item['n']}")


if __name__ == "__main__":
    main()
