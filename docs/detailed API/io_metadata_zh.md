# IO 与 Metadata 说明

本文档说明 `qudpy_sjh.utils.io` 对单次 `DynamicsResult` 的数据导出、
图像保存和 metadata 文件结构。

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

`ResultLike` 在本模块中表示单次动力学结果：

```text
DynamicsResult
```

即一个完整时间轴上的 density-matrix trajectory。

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

`components.csv` 不包含以下 spectroscopy / analysis 层结果：

```text
dipole expectation
polarization
FFT response
absorption response
TA response
```

这些量由 spectroscopy / analysis 层生成，并保存为单独的分析文件。

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

`meta.json` 是 human-facing v2 schema：它只保存去重后的人类可读摘要，
优先使用 physical units。完整 raw/grouped 参数不会完整出现在
`meta.json` 顶层；这类信息属于 `debug_meta.json` 或 raw serialization
路径。

顶层 block 固定为：

```text
schema
identity
params
results
stats
exports
```

当前结构：

```text
schema                         # metadata schema 标识
  name                         # schema 名称，固定为 "qudpy_result_metadata"
  version                      # schema 版本号，当前为 2

identity                       # 结果身份信息与用户备注
  result_type                  # result 对象类型，例如 "DynamicsResult"
  example_name                 # 示例或工作流名称
  condition_name               # 实验/扫描条件名称
  case_name                    # 当前 case 名称
  mode                         # result 表示模式，例如 "lab_exact"
  source_mode                  # 派生 result 的来源模式；原始结果通常为 null
  user_notes                   # 用户输入的说明与自定义 metadata
    description                # 来自 NLevelPhysicalParams.input_description
    metadata                   # 来自 NLevelPhysicalParams.input_metadata

params                         # 物理输入与求解配置摘要
  system                       # matter system 定义
    basis                      # basis state 名称
    dimension                  # Hilbert space 维度
    energies_eV                # 各能级能量，单位 eV
    dipole_matrix_D            # optical polarization 投影后的偶极矩矩阵，单位 Debye
    dissipation                # 开放系统通道
      relaxation_channels      # population relaxation channels
      pure_dephasing_channels  # pure dephasing channels
  field                        # optical input field 摘要
    class                      # field class 名称
    expression                 # 场表达式说明
    parameters                 # 场参数，例如 E0、omega、phase、envelope
    units                      # field/time 单位
    amplitude_convention       # 场幅值约定
    rebuildable                # field payload 是否可重建
    debug_details              # 更完整 field 信息所在文件
  solve                        # 求解配置与 solver 表示
    time_grid                  # 物理时间网格摘要
    solver                     # solver representation 摘要

results                        # 主要结果摘要
  trajectory_summary           # populations/coherences/trace 等轨迹摘要

stats                          # 数值检查摘要
  sanity_summary               # trace、Hermiticity 等 sanity checks

exports                        # 导出文件说明
  component_export             # components.csv 的列语义
  output_files                 # 当前 case 写出的文件路径
```

`identity.user_notes` 来自 `NLevelPhysicalParams.input_description` 和
`NLevelPhysicalParams.input_metadata`。这里命名为 `user_notes`，避免和
optical input field 混淆。

`params.system` 只描述 matter system。开放系统通道必须放在
`params.system.dissipation` 下，而不是顶层 `dissipation` 或
`params.dissipation`。

`params.system` 未包含以下 system-field 关系或归一化派生诊断量：

```text
transition_table
detuning_eV
detuning_fs_inv
laser_energy_eV
coupling_fs_inv
```

这些量不是 matter system 自身的 raw 参数；更完整的诊断信息位于
`debug_meta.json`，未来也可以由独立的 diagnostics / light_matter block
表达。

`params.field` 只描述 optical input field，不混入 matter system 信息。
`field.parameters.envelope` 使用 `"constant"`、`"gaussian"`、`"sech"` 等
简洁字符串，而不是 dict 或字符串化 dict。

`params.field` 未包含以下 matter system 信息：

```text
dipole_matrix_D
energies_eV
transition_table
relaxation_channels
pure_dephasing_channels
user_metadata
```

`params.solve.time_grid` 保存时间网格；`params.solve.solver` 保存 solver
representation 摘要。`meta.json` 不重复保存完整的 `grouped_params["solve"]`。

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

它适合调试；面向普通阅读和结果浏览时，优先查看 `meta.json`。

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
