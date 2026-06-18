# Solver 与 DynamicsResult API 说明

本文档说明当前 `sjh_learn.utils.core.solvers` 和 `sjh_learn.utils.core.results` 中被 examples / scratch 脚本直接使用的 active API。

当前主线是 `lab_exact`。RWA solver-unit drive path 已删除或禁用，不应为了 example 恢复。

## 推荐导入

```python
from sjh_learn.utils.core import (
    NLevelPhysicalParams,
    ParaNormalizer,
    run_case,
    run_cases,
    make_rotating_view,
    DynamicsResult,
)
```

`sjh_learn.utils.core.__init__` 会 re-export core 层公开入口。analysis observable 不应从 core 入口导出。

## run_case

```python
run_case(
    physical_params: NLevelPhysicalParams,
    normalizer: ParaNormalizer | None = None,
    rho0: Qobj | None = None,
    *,
    load_ckp: str | Path | None = None,
    save_ckp: str | Path | None = None,
    force_run: bool = False,
) -> DynamicsResult
```

### 作用

运行单个 physical case，返回一个 `DynamicsResult`。

主路径：

```text
NLevelPhysicalParams
-> ParaNormalizer.normalize()
-> NLevelSolverParams
-> build_lab_hamiltonian()
-> build_c_ops()
-> qutip.mesolve()
-> DynamicsResult(mode="lab_exact")
```

### 参数

| 参数 | 含义 |
|---|---|
| `physical_params` | 用户侧物理系统输入 |
| `normalizer` | 可选 `ParaNormalizer`；若为 `None`，内部创建默认 normalizer |
| `rho0` | 可选初始 density matrix；若为 `None`，通常使用默认初态 |
| `load_ckp` | 可选 checkpoint 路径；存在且 `force_run=False` 时直接读取 |
| `save_ckp` | 可选 checkpoint 保存路径 |
| `force_run` | 若为 `True`，忽略已有 checkpoint 并重新运行 |

### solver mode

当前主线只维护：

```python
solver_mode="lab_exact"
```

如果 `physical_params.solver_mode == "rwa"`，当前代码会进入 legacy guard 并报错。不要在新 example 中使用 RWA solver-unit drive。

### field 传递

`run_case()` 内部会把 `physical_params.field` 通过 `ParaNormalizer.make_code_field()` 转成单个 code-unit callable。`model.py / solvers.py` 只使用：

```python
args={"field": field}
```

不要恢复 `args["fields"]`，也不要让 solver 层接收多个 field callable。多脉冲组合必须已经由 `FieldPhySeries / TAField / TwoDESField` 在 physical field 层完成。

## run_cases

```python
run_cases(
    physical_params_list: Iterable[NLevelPhysicalParams],
    normalizer: ParaNormalizer | None = None,
) -> list[DynamicsResult]
```

这是轻量 convenience wrapper，本质上是对多个 `NLevelPhysicalParams` 逐个调用 `run_case()`。参数扫描、delay scan、power scan 和 batch workflow 仍属于上层 scripts/examples，不属于 core。

## make_rotating_view

```python
make_rotating_view(lab_result: DynamicsResult) -> DynamicsResult
```

### 作用

从 `lab_exact` 结果派生 rotating-frame view。它不重新调用 `mesolve`，只对已有 density matrix trajectory 做后处理变换。

### 限制

当前该函数只用于 N=2 `lab_exact` 结果：

```text
lab_result.mode == "lab_exact"
lab_result.dimension() == 2
```

不应把 `rotating_view` 当成 RWA solver 结果。它只是 lab-frame trajectory 的派生视图。

## DynamicsResult

`DynamicsResult` 是单轨迹结果对象。一个 `DynamicsResult` 只表示一次 simulation case 或一个派生 view。

### 字段

```python
DynamicsResult(
    mode: str,
    times: np.ndarray,
    times_fs: np.ndarray | None,
    states: list[Qobj],
    parameters: Any,
    physical_params: NLevelPhysicalParams | None = None,
    solver_params: SolverParams | None = None,
    metadata: dict[str, Any] = ...,
    source_mode: str | None = None,
    drive: Any | None = None,
    drive_dict: dict[str, Any] | None = None,
    drive_expr: str | None = None,
    drive_name: str | None = None,
    sanity_checks: dict[str, Any] = ...,
)
```

### 关键字段含义

| 字段 | 含义 |
|---|---|
| `mode` | `"lab_exact"`、`"rotating_view"` 或 legacy mode |
| `times` | solver code-unit 时间轴 |
| `times_fs` | physical 时间轴，单位 fs |
| `states` | 每个时间点的 density matrix，通常是 QuTiP `Qobj` |
| `parameters` | solver 内部参数 |
| `physical_params` | 原始 `NLevelPhysicalParams` |
| `solver_params` | `ParaNormalizer` 输出的 `SolverParams` |
| `metadata` | example / condition / case 备注 |
| `source_mode` | 派生 view 的来源 |
| `drive` | solver 实际使用的 field / drive callable |
| `drive_dict` | field / drive metadata |
| `sanity_checks` | trace / Hermiticity 等 sanity check 结果 |

## DynamicsResult 常用方法

### density_array

```python
rho_t = result.density_array()
```

返回 shape 为 `(n_time, N, N)` 的 complex ndarray。

这是 analysis 层计算 observable 的主要入口：

```python
P_t = np.einsum("tij,ji->t", rho_t, mu)
```

### dimension

```python
N = result.dimension()
```

返回 Hilbert space dimension。

### populations

```python
pops = result.populations()
```

返回 shape 为 `(n_time, N)` 的 population array。每列对应一个 diagonal density matrix element。

### matrix_element

```python
rho01 = result.matrix_element(0, 1)
```

返回指定 density matrix element 随时间的 complex array。

### matrix_elements

```python
items = result.matrix_elements([(0, 1), (0, 2)])
```

返回多个矩阵元。返回 dict 的 key 使用 `rho_ij` 命名。

### selected_elements

```python
items = result.selected_elements({
    "rho_ge": (0, 1),
    "rho_gf": (0, 2),
})
```

返回用户指定标签的矩阵元。

### field_MV_per_cm_values

```python
E_t = result.field_MV_per_cm_values()
```

对 `lab_exact` result，返回 solver 实际使用的 physical lab-frame field，单位 `MV/cm`。输出层应优先使用这个方法，而不是重新拼写 field 函数。

### drive_values / drive_code_values / drive_fs_inv_values

这些方法主要用于兼容 RWA / drive 展示。当前 `lab_exact` 主线分析光场时优先使用 `field_MV_per_cm_values()` 或直接调用原始 physical field。

### max_trace_error

```python
err = result.max_trace_error()
```

计算：

```text
max_t |Tr(rho(t)) - 1|
```

### max_hermiticity_error

```python
err = result.max_hermiticity_error()
```

计算 density matrix 的最大 Hermiticity deviation。

### summary_dict

```python
summary = result.summary_dict()
```

生成紧凑 summary，包括 mode、dimension、final populations、trace error、Hermiticity error 和 time range。

### metadata_dict

```python
metadata = result.metadata_dict()
```

生成 debug metadata。该方法会把 dataclass、ndarray、complex、Qobj 等转换为 JSON-safe 结构。

### to_npz_dict

```python
payload = result.to_npz_dict()
```

生成可保存到 NPZ 的数组字典，通常包含：

```text
time_fs
time_code
density
field_MV_per_cm
```

如果是 legacy drive 结果，也可能包含 `drive_code` 或 `drive_fs_inv`。

### components_dataframe

```python
df = result.components_dataframe()
```

生成包含所有 populations 和 upper-triangular coherences 的 dataframe。需要 pandas。

### populations_dataframe

```python
df = result.populations_dataframe()
```

生成 population-only dataframe。

### selected_elements_dataframe

```python
df = result.selected_elements_dataframe({"rho_ge": (0, 1)})
```

生成指定矩阵元 dataframe。

### plot_times_and_label

```python
times, xlabel = result.plot_times_and_label()
```

如果 `times_fs` 存在，返回 physical fs 时间轴和 `"Time (fs)"`；否则返回 solver code time。

## checkpoint

`DynamicsResult.save_ckp(file)` 会用 pickle 保存完整 result。`DynamicsResult.from_ckp(file)` 会读取 checkpoint。

`.ckp` 是内部 checkpoint / 后处理缓存，不保证跨版本长期稳定；不要把它当成替代 `density.npz`、`components.csv`、`meta.json` 和 `debug_meta.json` 的归档格式。不要加载不可信来源的 pickle 文件。

## 与 analysis 层的边界

`DynamicsResult` 保存 density matrix trajectory 和通用访问接口，但不负责：

```text
dipole expectation
polarization
FFT spectrum
TA response
absorption-like response
publication figure workflow
```

这些属于 `sjh_learn.utils.spectroscopy` 或上层 examples/scripts。

典型 analysis 用法：

```python
from sjh_learn.utils.spectroscopy import polarization_C_per_m2, lab_frame_fft_response_legacy

rho_t = result.density_array()
P_t = polarization_C_per_m2(
    rho_t,
    result.physical_params.dipole_matrix_D,
    number_density_m3 = 1.0e24,
)

E_t = result.field_MV_per_cm_values()

response = lab_frame_fft_response_legacy(
    t_fs = result.times_fs,
    E_MV_per_cm = E_t,
    P_C_per_m2 = P_t,
    rhoij = result.matrix_element(0, 1),
)
```
