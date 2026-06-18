# IO 与 Metadata 说明

本文档说明当前 `qudpy_sjh.utils.io` 的普通 full-window `DynamicsResult` 输出逻辑和 metadata 边界。

当前 IO 只应面向普通 `DynamicsResult`。不要在本层宣传 piecewise series、dark propagation 或 `materialize_full`。

## 推荐导入

```python
from qudpy_sjh.utils.io import (
    save_result_data,
    save_result_case,
    save_figure,
    save_parameter_summary,
    save_results_components_long,
    QuantumResultIO,
)
```

## ResultLike

当前 `ResultLike` 应理解为：

```text
DynamicsResult
```

即普通单轨迹结果对象。

不要写成支持：

```text
PieceDynamicsResultSeries
piecewise series
active/dark series
```

## save_result_data

```python
save_result_data(
    result,
    output_dir,
    *,
    save_npz=True,
    save_csv=True,
    save_populations_csv=False,
    save_json=True,
    save_human_meta=True,
    save_debug_meta=True,
    example_name=None,
    condition_name=None,
    case_name=None,
    selected_elements=None,
)
```

典型输出：

```text
data/
  density.npz
  components.csv
  populations.csv
  selected_elements.csv
```

以及 metadata：

```text
meta.json
debug_meta.json
```

具体 metadata 写入位置取决于调用层传入的目录结构。

## save_result_case

```python
save_result_case(
    result,
    output_dir,
    *,
    output_data=True,
    output_preview=False,
    fig=None,
    preview_fig=None,
    full_fig=None,
    save_npz=True,
    save_csv=True,
    save_populations_csv=False,
    save_json=True,
    selected_elements=None,
    append_results_csv=True,
)
```

典型 case 输出：

```text
outdir/
├─ results.csv
└─ case_name/
   ├─ data/
   │  ├─ density.npz
   │  ├─ components.csv
   │  ├─ populations.csv
   │  └─ selected_elements.csv
   ├─ figs/
   │  ├─ preview.png
   │  ├─ full.png
   │  └─ component/
   ├─ meta.json
   └─ debug_meta.json
```

`output_preview=True` 时，IO 层可以调用 plotting helper 生成低清 preview 和 component figures。`result` 对象本身不负责绘图。

## density.npz

`density.npz` 来自：

```python
result.to_npz_dict()
```

常见字段：

```text
time_fs
time_code
density
drive_code
drive_fs_inv
field_MV_per_cm
```

其中：

- `time_fs` 是物理时间轴；
- `time_code` 是 solver 内部时间轴；
- `density` 是完整 density matrix trajectory；
- `field_MV_per_cm` 只对 lab-frame physical field result 有意义。

## components.csv

`components.csv` 来自：

```python
result.components_dataframe()
```

包含：

```text
time_fs
所有 diagonal populations
所有 upper-triangular off-diagonal coherences
field / drive diagnostic columns
```

coherence 保存：

```text
real
imag
abs
phase
phase_unwrapped
```

`components.csv` 不应保存：

```text
dipole expectation
polarization
FFT response
absorption response
TA response
```

这些属于 spectroscopy / analysis 层，应另存为单独分析文件。

## populations.csv

`populations.csv` 来自：

```python
result.populations_dataframe()
```

保存所有 diagonal elements。

## selected_elements.csv

如果传入：

```python
selected_elements={
    "rho_01": (0, 1),
}
```

则可保存指定 density matrix elements 的时间序列。

## meta.json

`meta.json` 是人类可读的紧凑 metadata，优先使用 physical units。

推荐顶层 block：

```text
result_type
example_name
condition_name
case_name
mode
source_mode
user_input
system
field
dissipation
time_grid
solver
trajectory_summary
sanity_summary
component_export
output_files
```

`field` block 只描述 optical input field，不应混入 matter system 信息。

不应放入 `field` block 的内容：

```text
dipole_matrix_D
transitions_eV
transition_table
relaxation_channels
pure_dephasing_channels
user_metadata
```

这些应分别放入：

```text
system
dissipation
user_input
```

## debug_meta.json

`debug_meta.json` 保存更完整的 debug payload，可能包括：

```text
physical_params
solver_params_fs_inv
solver_params_code
parameters_code
drive_dict
drive_expr
sanity_checks
```

它适合调试，不应作为面向普通读者的唯一 metadata。

## results.csv

`save_result_case(..., append_results_csv=True)` 会在根输出目录追加 case summary。

典型字段：

```text
case_name
result_type
mode
source_mode
time_start_fs
time_end_fs
n_time_points
dimension
field_MV_per_cm
laser_energy_eV
detuning_fs_inv
max_trace_error
max_hermiticity_error
final_populations
```

## save_figure

```python
save_figure(fig, output_path, dpi=120)
```

只负责保存已经构造好的 matplotlib figure。

plotting 函数负责生成 figure；IO 只负责写文件。

## QuantumResultIO

`QuantumResultIO` 是简单封装：

```python
io = QuantumResultIO(outdir="outputs")
io.save_case(result)
```

当前它仍应只处理普通 `DynamicsResult`。

## 不支持的旧输出边界

当前文档不应写：

```text
save_result_case supports PieceDynamicsResultSeries
save_result_case supports piecewise series
materialize_full before saving
dark materialization output
active/dark output schema
long-window piecewise benchmark output
```

如果未来需要这些能力，应新增独立 schema，并明确与普通 `DynamicsResult` 输出的边界。
