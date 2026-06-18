# Solver 与 DynamicsResult 说明

本文档说明当前 `qudpy_sjh.utils.core.solvers` 和 `qudpy_sjh.utils.core.results` 的主线接口。

当前主线是 full-window `lab_exact` solver。标准结果对象是普通 `DynamicsResult`。

## 推荐导入

```python
from qudpy_sjh.utils.core import (
    run_case,
    run_cases,
    make_rotating_view,
    DynamicsResult,
)
```

## run_case

当前主入口：

```python
run_case(
    physical_params,
    normalizer=None,
    rho0=None,
    *,
    load_ckp=None,
    save_ckp=None,
    force_run=False,
) -> DynamicsResult
```

输入：

```text
NLevelPhysicalParams
```

输出：

```text
DynamicsResult
```

当前主路径：

```text
NLevelPhysicalParams
-> ParaNormalizer.normalize()
-> _optical_codeparams_from_solverparams()
-> build_lab_hamiltonian()
-> build_c_ops()
-> qutip.mesolve()
-> DynamicsResult(mode="lab_exact")
```

## full-window lab_exact

当前 solver 主线是：

```text
solver_mode="lab_exact"
```

它在完整时间窗口内直接传播 lab-frame time-dependent Hamiltonian，保留 optical carrier。

当前主线不使用：

```text
piecewise propagation
dark propagation
active/dark sequence
materialization
```

## RWA 状态

`solver_mode="rwa"` 当前应视为 legacy guard。

新示例和新文档不应把 RWA 写成默认 fallback，也不应恢复旧 solver-unit drive classes。

`make_rotating_view(lab_result)` 只是从 `lab_exact` 结果派生 rotating-frame view，不是独立 RWA solver。

## run_cases

`run_cases(...)` 是薄封装：

```python
run_cases(physical_params_list, normalizer=None) -> list[DynamicsResult]
```

它对列表逐个调用 `run_case(...)`。

复杂 parameter scan、delay scan、power scan、batch workflow 不属于 core，应在上层 examples / scripts 中组织。

## DynamicsResult

`DynamicsResult` 表示一条 density-matrix trajectory。

主要字段：

```text
mode
times
times_fs
states
parameters
physical_params
solver_params
metadata
source_mode
drive
drive_dict
drive_expr
drive_name
sanity_checks
```

含义：

- `mode`：例如 `"lab_exact"` 或 derived `"rotating_view"`；
- `times`：solver code time；
- `times_fs`：physical time axis，单位 fs；
- `states`：QuTiP `Qobj` density matrices；
- `parameters`：solver 内部参数；
- `physical_params`：用户侧 physical input；
- `solver_params`：normalizer 输出的 `SolverParams`；
- `metadata`：运行摘要；
- `drive`：solver 实际使用的 code field callable；
- `drive_dict` / `drive_expr` / `drive_name`：field / drive metadata；
- `sanity_checks`：trace、Hermiticity 等检查结果。

## 常用方法

```python
density_array()
dimension()
populations()
matrix_element(i, j)
matrix_elements(pairs)
selected_elements(elements)
field_MV_per_cm_values()
drive_values(times=None)
max_trace_error()
max_hermiticity_error()
summary_dict()
metadata_dict()
to_npz_dict()
components_dataframe()
populations_dataframe()
selected_elements_dataframe(elements)
plot_times_and_label()
save_ckp(path)
from_ckp(path)
```

## density_array

```python
rho_t = result.density_array()
```

返回 shape：

```text
(T, N, N)
```

其中 `T` 是时间点数量，`N` 是能级数量。

## populations

```python
pop = result.populations()
```

返回所有 diagonal populations：

```text
shape = (T, N)
```

## matrix_element

```python
rho_01 = result.matrix_element(0, 1)
```

返回指定 density matrix element 的时间序列。

## field_MV_per_cm_values

```python
E = result.field_MV_per_cm_values()
```

对 `lab_exact` result，该方法返回 solver 实际使用的 physical lab-frame field，单位 `MV/cm`。

该方法优先复用 internal code field adapter 绑定的 physical field，避免输出层重新拼写场函数。

## components_dataframe

```python
df = result.components_dataframe()
```

包含：

```text
time_fs
所有 diagonal populations
所有 upper-triangular off-diagonal coherences
field / drive diagnostic columns
```

coherence 通常包含：

```text
real
imag
abs
phase
phase_unwrapped
```

## metadata_dict

`metadata_dict()` 返回较完整的 debug metadata，可能包含 code-unit 参数、physical 参数、field metadata、sanity checks 等。

面向普通用户的紧凑 metadata 应由 IO 层写入 `meta.json`。

## checkpoint

`DynamicsResult` 支持：

```python
result.save_ckp(path)
result = DynamicsResult.from_ckp(path)
```

`.ckp` 是 pickle-based 内部 checkpoint / 后处理缓存，不保证跨版本长期稳定。不要加载不可信来源的 `.ckp` 文件。

`.ckp` 不应替代正式归档输出：

```text
density.npz
components.csv
meta.json
debug_meta.json
```

## DynamicsResult 不负责的内容

`DynamicsResult` 不负责：

```text
求解微分方程
绘图
保存 case 文件
判断物理结论
parameter scan
delay scan
dipole expectation
polarization
FFT response
absorption response
TA response
```

这些职责属于 solver、plotting、IO、workflow 或 spectroscopy 层。

## 示例

```python
from qudpy_sjh.utils.core import run_case

result = run_case(params)

rho_t = result.density_array()
pop = result.populations()
E = result.field_MV_per_cm_values()
meta = result.metadata_dict()
```
