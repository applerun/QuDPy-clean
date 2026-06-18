# QuDPy Metadata Schema 规范

版本：0.1  
状态：建议作为当前基线  
范围：`save_result_case()` 为 `DynamicsResult` 生成的 `meta.json` 与 `debug_meta.json`

## 1. 设计目标

QuDPy 的 metadata 分成两层：

- `meta.json`：紧凑、便于人阅读、优先使用物理单位的单个 simulation result 摘要。
- `debug_meta.json`：更冗长的内部调试记录，用于 debugging、可复现性、code-unit 转换、solver adapter 信息和完整 sanity-check 细节。

默认 metadata schema 必须同时支持二能级体系和一般 N 能级体系。schema 不能假设体系只有一个 transition、一个 energy gap、一个 detuning、一个 Rabi frequency 或一个 dephasing rate。

## 2. 顶层结构规则

`meta.json` 的顶层结构应固定为：

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

除非确实出现了新的概念层，否则不要增加新的顶层 block。

以下旧顶层 block 不应再出现在 `meta.json` 中：

```text
inputs_physical
input_field
input_field_rebuild
derived_physical
solver_representation
lab_frame_solver
rotating_transform
input_drive
solver_code_summary
```

如果这些旧 block 中的信息仍然有用，应移动到上面规定的 block 中，或者只放入 `debug_meta.json`。

## 3. `user_input`

用途：保留用户提供的描述和可选用户 metadata。

推荐结构：

```json
"user_input": {
  "description": "...",
  "metadata": {
    "...": "..."
  }
}
```

规则：

- 该 block 可以保留用户手写备注，例如 `transitions_eV` 或 `dipoles_D`。
- 这些用户备注不是规范化的机器可读系统描述。
- 规范化的体系信息必须放在 `system` 中。
- 不要把 `user_input.metadata` 复制进 `field`、debug field adapter 或 field rebuild 相关结构。

## 4. `system`

用途：描述量子物质体系。

必需字段：

```text
basis
dimension
energies_eV
dipole_matrix_D
transition_table
```

推荐结构：

```json
"system": {
  "basis": ["0", "1", "2"],
  "dimension": 3,
  "energies_eV": [0.0, 1.5, 1.75],
  "dipole_matrix_D": [[...]],
  "transition_table_rule": "all i < j transitions; dipole_coupled indicates nonzero dipole",
  "transition_table": [...]
}
```

### 4.1 能级

`energies_eV` 是物理单位下的规范能级列表。

对 N 能级体系，transition energy 应由下式得到：

```text
transition_energy_eV(i, j) = energies_eV[j] - energies_eV[i], for i < j
```

### 4.2 偶极矩矩阵

`dipole_matrix_D` 是 Debye 单位下的物质-光场耦合矩阵。

规则：

- 它属于 `system`。
- 应保存为完整矩阵。
- 如果支持复数偶极矩，矩阵元素应使用 JSON-safe 的复数格式：
  `{"real": ..., "imag": ...}`。

### 4.3 Transition table

`transition_table` 应从 `energies_eV`、`dipole_matrix_D`、field metadata 中可用的
`laser_energy_eV` / `omega_L_fs_inv`，以及 solver 推导出的 coupling 自动生成。
如果 field 不是单频 carrier 或没有对应 metadata，`laser_energy_eV` /
`detuning_fs_inv` 可以为 `null`。

每一行建议包含：

```json
{
  "from": 0,
  "to": 1,
  "label": "0_to_1",
  "energy_eV": 1.5,
  "omega_fs_inv": 2.278901173214266,
  "laser_energy_eV": 1.625,
  "detuning_eV": -0.125,
  "detuning_fs_inv": -0.18990843110118893,
  "dipole_D": {"real": 3.0, "imag": 0.0},
  "coupling_fs_inv": {"real": 0.0009489083473202662, "imag": 0.0},
  "dipole_coupled": true
}
```

规则：

- 默认包含所有上三角 pair，即所有 `i < j`。
- 当 dipole 为 0 时，设 `dipole_coupled = false`。
- 不要默认删除 dark transition，除非之后明确添加 filtering 选项。
- 如果包含所有 transition，应写入 `transition_table_rule` 说明规则。

### 4.4 二能级兼容字段

当且仅当 `dimension == 2` 时，`system` 可以额外包含：

```text
energy_gap_eV
detuning_eV
```

规则：

- 这两个字段只是二能级兼容和快速阅读字段。
- 只允许在 `dimension == 2` 时写入。
- 对 `dimension > 2`，不能写入这些 singular 字段。

## 5. `field`

用途：只描述输入光场。

推荐结构：

```json
"field": {
  "class": "GaussianCarrierFieldPhysical",
  "expression": "E(t) = 2 E0 f(t) cos(omega_L t + phase)",
  "parameters": {
    "E0_MV_per_cm": 0.1,
    "peak_E_MV_per_cm": 0.2,
    "omega_L_fs_inv": 2.468809604315455,
    "laser_energy_eV": 1.625,
    "phase_rad": 0.0,
    "envelope": "gaussian",
    "pulse_center_fs": 0.0,
    "pulse_sigma_fs": 5.0
  },
  "units": {
    "field": "MV/cm",
    "time": "fs"
  },
  "amplitude_convention": "E0_MV_per_cm is E0 in E(t) = 2 E0 f(t) cos(omega_L t + phase).",
  "rebuildable": true,
  "debug_details": "debug_meta.json"
}
```

规则：

- `field` 不能包含 `dipole_matrix_D`、`dipoles_D`、`transitions_eV`、`transition_table`、relaxation channels 或 pure dephasing channels。
- `field` 不能复制 `user_input.metadata`。
- `field` 只描述光场参数和单位。
- 完整 rebuild 细节可以放入 `debug_meta.json`。

## 6. `dissipation`

用途：用物理单位描述 relaxation 和 pure dephasing channels。

推荐结构：

```json
"dissipation": {
  "relaxation_channels": [
    {
      "name": "relaxation_1_to_0",
      "from_level": 1,
      "to_level": 0,
      "T1_fs": 800.0,
      "rate_fs_inv": 0.00125
    }
  ],
  "pure_dephasing_channels": [
    {
      "name": "pure_dephasing_level_1",
      "level": 1,
      "Tphi_fs": 250.0,
      "rate_fs_inv": 0.004
    }
  ]
}
```

规则：

- 使用物理单位。
- 不要在 `meta.json` 中保存 code-unit rate。
- code-unit rate 属于 `debug_meta.json`。

## 7. `time_grid`

用途：用物理单位描述 simulation time axis。

推荐结构：

```json
"time_grid": {
  "t_start_fs": -1000.0,
  "t_end_fs": 1000.0,
  "dt_fs": 0.05,
  "n_time_points": 40001,
  "time_axis_unit": "fs"
}
```

规则：

- 优先使用 fs 单位的物理时间。
- code-unit time grid 属于 `debug_meta.json`。
- 不要在 JSON metadata 中重复保存完整时间数组。

## 8. `solver`

用途：描述 simulation representation 和 solver 语义。

`lab_exact` 的推荐结构：

```json
"solver": {
  "mode": "lab_exact",
  "description": "Direct lab-frame propagation with the full time-dependent optical carrier.",
  "hamiltonian_type": "time-dependent lab-frame Hamiltonian",
  "uses_lab_field_directly": true,
  "uses_rwa_drive": false,
  "uses_rotating_transform": false,
  "carrier_retained": true,
  "counter_rotating_terms_retained": true,
  "rwa_approximation_applied": false,
  "independently_solved_by_mesolve": true,
  "interaction": "H_int(t) = -mu E(t)"
}
```

规则：

- 旧的 `solver_representation` 和 `lab_frame_solver` 语义合并到这里。
- 对 `lab_exact`，不要再写单独的 `input_drive` 顶层 block。
- 对 `rwa`，RWA drive 细节可以放在这里或 mode-specific subsection 中，但不要重新引入旧的顶层 `input_drive`。

## 9. `trajectory_summary`

用途：紧凑总结 density matrix trajectory。

推荐字段：

```text
n_time_points
time_range_fs
dimension
final_populations
max_populations
final_coherences
max_trace_error
max_hermiticity_error
```

规则：

- 包含所有 population。
- 包含上三角非对角 coherence。
- coherence summary 可以包含 `final_abs`、`max_abs`、`final_phase` 和 `final_phase_unwrapped`。
- 大数组不属于这里。

## 10. `sanity_summary`

用途：在 `meta.json` 中暴露关键数值 sanity-check 结果。

推荐结构：

```json
"sanity_summary": {
  "trace_error_small": true,
  "hermiticity_error_small": false,
  "max_trace_error": 2.22e-16,
  "max_hermiticity_error": 2.28e-7,
  "details": "debug_meta.json"
}
```

规则：

- 保持紧凑。
- 完整 sanity-check 细节放入 `debug_meta.json`。
- 如果 sanity check 失败，必须能从 `meta.json` 直接看出来。

## 11. `component_export`

用途：描述保存了哪些 density matrix components。

推荐结构：

```json
"component_export": {
  "dimension": 3,
  "component_indexing": "zero_based",
  "saved_populations": ["rho_00", "rho_11", "rho_22"],
  "saved_coherences": ["rho_01", "rho_02", "rho_12"],
  "saved_coherences_rule": "upper triangular off-diagonal elements only",
  "coherence_components": ["real", "imag", "abs", "phase", "phase_unwrapped"],
  "saved_observables": [],
  "observable_note": "Observable, polarization, FFT, and absorption-like quantities are analysis-layer outputs.",
  "density_npz": "full density matrix trajectory"
}
```

规则：

- 该 block 描述导出的 density-matrix components，不描述 spectroscopy post-processing。
- 如果以后增加 spectra 或 polarization 输出，应单独设计对应记录。

## 12. `output_files`

用途：列出当前 case 生成的文件。

推荐结构：

```json
"output_files": {
  "density": "data/density.npz",
  "components": "data/components.csv",
  "populations": "data/populations.csv",
  "preview": "figs/preview.png",
  "debug_metadata": "debug_meta.json",
  "component_figures": {
    "rho_01": "figs/component/rho_01.png",
    "rho_02": "figs/component/rho_02.png",
    "rho_12": "figs/component/rho_12.png"
  },
  "component_figures_dir": "figs/component"
}
```

规则：

- `component_figures` 应包含所有生成的 component figure。
- 如果生成了 component figures，应写入 `component_figures_dir`。
- 路径应相对于 case directory。

## 13. `debug_meta.json`

用途：保存冗长的内部调试和可复现性记录。

允许内容：

```text
summary
sanity_checks
result_metadata
drive_dict
parameters_code
physical_params
solver_params_fs_inv
solver_params_code
example_name
condition_name
case_name
```

规则：

- `debug_meta.json` 可以包含 code-unit quantities。
- 可以包含 adapter 内部信息。
- 可以包含完整 solver parameters。
- 不应重复完整时间数组，应使用 summary。
- Field adapter metadata 不应包含 matter-system metadata，例如 `dipoles_D` 或 `transitions_eV`。
- 不应写入 `source_field.metadata.user_metadata`。
- 如果需要保存用户输入 metadata，应放在 `physical_params.input_metadata` 或顶层 user-input 相关记录中，不应塞进 field adapter。

## 14. 明确不做的事

metadata schema 不应该：

- 改变 solver 数值行为。
- 改变 Hamiltonian 构造。
- 改变 Lindblad channel 构造。
- 改变 spectroscopy post-processing。
- 改变 plotting style。
- 把 N 能级体系隐式当作二能级体系。
- 对 `dimension > 2` 使用一个全局的 `energy_gap_eV`、`detuning_eV`、`rabi_fs_inv` 或 `gamma2_fs_inv`。

## 15. 回归检查建议

### 15.1 compact 顶层结构

对 `meta.json`，检查顶层 keys 至少应限制在：

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

### 15.2 field block 清洁度

检查 `field` 中不包含：

```text
dipoles_D
dipole_matrix_D
transitions_eV
transition_table
user_metadata
relaxation_channels
pure_dephasing_channels
```

### 15.3 多能级 singular 字段排除

对 `dimension > 2`，检查 `system` 中不包含：

```text
energy_gap_eV
detuning_eV
```

并检查 `meta.json` 中不存在旧顶层 `derived_physical`。

### 15.4 二能级兼容性

对 `dimension == 2`，检查：

```text
system.energy_gap_eV exists
system.detuning_eV exists
system.transition_table exists
len(system.transition_table) == 1
```

### 15.5 多能级 transition table

对 `dimension == N`，除非显式启用过滤，否则检查：

```text
len(system.transition_table) == N * (N - 1) / 2
```

### 15.6 component figure 输出文件

如果生成 component figures，检查：

```text
output_files.component_figures exists
output_files.component_figures_dir exists
```

### 15.7 sanity summary

检查 `sanity_summary` 暴露：

```text
trace_error_small
hermiticity_error_small
max_trace_error
max_hermiticity_error
details
```

## 16. 建议 commit message

```bash
git commit -m "Document compact metadata schema"
```
