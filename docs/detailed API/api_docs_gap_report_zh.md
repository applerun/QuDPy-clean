# QuDPy API 文档缺口审计

本文档记录在当前 `field_zh.md`、`normalizer_zh.md`、`current_architecture_zh.md` 和 metadata schema 之外，为支持 TA / absorption diagnostic 脚本还需要补充的 API 文档。

当前已有文档已经覆盖：

- physical-field-first 主线；
- `FieldPhyRoot / FieldPhySeries / TAField / TwoDESField`；
- `ParaNormalizer` 的归一化职责；
- metadata schema 的顶层结构；
- 不恢复 solver-unit field、RWA drive 和 core `ParameterSweep` 的边界。

但目前脚本已经直接使用了一些更底层的 active API。如果这些 API 不写清楚，后续很容易出现“按 metadata dict 写入 core 参数”或“重新实现 FFT”的问题。

## 建议新增文档

### 1. `parameters_zh.md`

优先级：高。

原因：`NLevelPhysicalParams` 是用户侧正式入口，但当前文档还没有像 `field_zh.md` 一样系统说明它的字段、单位和 channel 类型。最近脚本报错的根本原因就是把 `relaxation_channels` 写成了 dict，而源码期望的是 `RelaxationChannel` 对象。

应覆盖：

- `NLevelPhysicalParams`
- `RelaxationChannel`
- `PureDephasingChannel`
- `SolverParams`
- `NLevelSolverParams`
- `dimension` / `energy_gap_eV`
- channel 中 `T1_fs` / `Tphi_fs` 与 `rate_fs_inv` 的优先级
- 不再支持的旧 top-level optical scalar，例如 `field_MV_per_cm`、`laser_energy_eV`、`pulse_center_fs`、`pulse_sigma_fs`

### 2. `solver_result_zh.md`

优先级：高。

原因：TA / absorption / window diagnostic 脚本会直接调用 `run_case()`，并从 `DynamicsResult` 读取 `times_fs`、`density_array()`、`matrix_element()`、`max_trace_error()` 等。现有架构文档有概念说明，但缺少可运行脚本需要的精确接口表。

应覆盖：

- `run_case()`
- `run_cases()`
- `make_rotating_view()`
- checkpoint 参数 `load_ckp` / `save_ckp` / `force_run`
- `DynamicsResult` 字段
- `DynamicsResult` 常用方法
- `DynamicsResult` 与 analysis observable 的边界

### 3. `spectroscopy_zh.md`

优先级：高。

原因：当前 TA intrinsic response 和 window diagnostic 都依赖 `sjh_learn.utils.spectroscopy`。尤其需要写清楚 `lab_frame_fft_response()` 已经负责 FFT、frequency axis、positive-frequency mask、small-denominator mask，不应在 example 中重复实现。

应覆盖：

- `dipole_expectation_D()`
- `polarization_C_per_m2()`
- `apply_time_window()`
- `safe_complex_ratio()`
- `lab_frame_fft_response()`
- `rwa_fft_response()` 的 legacy diagnostic 定位
- `window` 当前只支持 `None` / `"none"` / `"hann"`；如果需要 `"hamming"`，应扩展公共 helper，而不是在脚本中手写第二套 window 逻辑
- `lab_frame_fft_response()` 返回的是 mask 后的有效频率轴，因此不同输入场或不同 window 下 `energy_eV` 长度可能不同

## 可后续补充的文档

### `io_plotting_zh.md`

优先级：中。

当 examples 开始稳定保存 `save_result_case()`、preview figure、CSV/NPZ 输出时再补。当前 TA/window 脚本主要是 scratch / diagnostic，不必先扩展。

### `examples_workflow_zh.md`

优先级：中。

当 TA intrinsic response 从 scratch 变成稳定 example 时，再记录 example 组织规范，例如 output_dir、metadata、diagnostic figure、delay scan、probe-only cache、sanity tests。

## 文档边界建议

不要在这些新文档中恢复以下旧路径：

```text
CodeCarrierField
CodeGaussianCarrierField
CodeCompositeField
CodeConstantDrive
CodeGaussianDrive
solver_input_from_dict
args["fields"]
PhysicalParameterSweep
ParameterSweep
RWA solver-unit drive mainline
```

这些新增文档只应补齐 active API 的“怎么调用”和“不要怎么调用”，不改变 core 架构。
