# 当前文档总览与更新摘要

本文档合并旧的 `docs/document_updated.md` 与 `docs/document_update_summary.md`，用于记录 `QuDPy-clean` 当前文档主线、active code path、模块边界和需要替换的旧表述。

当前代码包路径为：

```text
qudpy_sjh/
```

旧文档中若仍出现 `sjh_learn/`，应视为历史命名残留。后续中文文档建议统一使用 `qudpy_sjh/`。

## 当前主线

当前 active code path 是：

```text
FieldPhyRoot / FieldPhySeries / TAField / TwoDESField
-> NLevelPhysicalParams(..., field=field)
-> ParaNormalizer
-> _CodeFieldAdapter
-> lab_exact full-window solver
-> DynamicsResult
-> spectroscopy / IO / plotting / examples
```

核心约定：

1. 用户侧光场使用 physical units：时间 `fs`，电场 `MV/cm`，能量 `eV`，偶极矩 `Debye`。
2. 正式入口是 `NLevelPhysicalParams(..., field=field)`。
3. `field` 必须是 `FieldPhyRoot` 或其子类 / wrapper。
4. `NLevelPhysicalParams` 不再把 `field_MV_per_cm`、`laser_energy_eV`、`pulse_center_fs`、`pulse_sigma_fs` 作为顶层主接口。
5. `ParaNormalizer` 只负责单位换算和 code-unit scaling，不定义物理模型。
6. 多脉冲叠加只发生在 physical field 层，例如 `FieldPhySeries`、`TAField`、`TwoDESField`。
7. solver 主路径只接收一个 code-unit callable：`args={"field": field}`。
8. 当前主线是 full-window `lab_exact`。
9. 标准结果对象是 `DynamicsResult`。
10. 一个 `DynamicsResult` 只表示一条 density-matrix trajectory。
11. 参数扫描、delay scan、power scan 和 batch run 属于上层 examples/scripts/workflow，不属于 core。
12. dipole expectation、polarization、FFT response、absorption response 和 TA response 属于 spectroscopy / analysis 层，不属于 solver/result core。

## 当前目录理解

```text
qudpy_sjh/
├─ utils/
│  ├─ core/
│  │  ├─ __init__.py
│  │  ├─ parameters.py
│  │  ├─ normalization.py
│  │  ├─ model.py
│  │  ├─ solvers.py
│  │  ├─ results.py
│  │  └─ config.py
│  ├─ fields/
│  │  ├─ __init__.py
│  │  ├─ lab_fields.py
│  │  ├─ field_series.py
│  │  └─ specific/
│  │     ├─ __init__.py
│  │     ├─ basic_fields.py
│  │     ├─ ta_fields.py
│  │     └─ twodes_fields.py
│  ├─ spectroscopy/
│  │  ├─ __init__.py
│  │  ├─ observables.py
│  │  ├─ absorption_spectra.py
│  │  └─ theory.py
│  ├─ plotting.py
│  ├─ io.py
│  └─ checks.py
```

若本地目录或旧文档仍保留 `utils/model.py`、`utils/solvers.py`、`utils/results.py`、`utils/normalization.py` 或 `utils/analysis/observables.py` 这类旧口径，应视为历史命名或迁移残留。新文档应以 `utils/core/` 和 `utils/spectroscopy/` 为准。

## 文档分工

建议当前中文文档采用以下结构：

```text
README.md
docs/document_summary.md
docs/deprecated_boundaries_zh.md
docs/detailed API/field_zh.md
docs/detailed API/parameters_zh.md
docs/detailed API/normalizer_zh.md
docs/detailed API/solver_result_zh.md
docs/detailed API/spectroscopy_zh.md
docs/detailed API/io_metadata_zh.md
docs/detailed API/workflow_examples_zh.md
```

其中：

- `README.md` 面向第一次打开仓库的人；
- `document_summary.md` 记录当前总架构、文档口径和更新摘要；
- `deprecated_boundaries_zh.md` 明确旧架构边界；
- `docs/detailed API/` 下的 7 个文档记录专题 API 与 workflow 边界。

## fields 层摘要

`utils/fields/` 描述用户侧 lab-frame physical field。所有内置 field 都应继承或符合 `FieldPhyRoot` 接口。

当前结构：

```text
qudpy_sjh/utils/fields/
  __init__.py
  lab_fields.py
  field_series.py
  specific/
    __init__.py
    basic_fields.py
    ta_fields.py
    twodes_fields.py
```

职责边界：

- `lab_fields.py`：基础 lab-frame field 抽象和 wrapper；
- `field_series.py`：多个 physical field 的线性叠加与 scan helper；
- `specific/basic_fields.py`：基础载波场和高斯载波场；
- `specific/ta_fields.py`：TA / pump-probe field helper；
- `specific/twodes_fields.py`：2DES field helper；
- `fields/__init__.py`：用户侧 public API re-export。

核心对象：

```text
FieldPhyRoot
TimeShiftedField
FieldPhyCustomed
FieldPhySeries
CarrierFieldPhysical
GaussianCarrierFieldPhysical
TAField
TwoDESField
```

核心 helper：

```text
make_default_carrier_field
make_default_gaussian_carrier_field
make_pump_probe_field_from_templates
make_ta_field_from_templates
make_ta_gaussian_field
make_twodes_gaussian_field
```

`TimeShiftedField` 的约定是：

```text
field.time_shifted(shift_fs)
shift_fs > 0 表示场整体向更晚时间移动
E_shifted(t) = E_original(t - shift_fs)
```

pump-probe template helper 的约定是：

```text
probe_center_fs 默认 0
pump_center_fs = probe_center_fs - delay_fs
```

需要注意：`make_ta_gaussian_field(...)` 当前是另一套直接 Gaussian 构造语义：

```text
probe_center_fs = pump_center_fs + probe_delay_fs
```

因此文档中应把 template-based helper 和 direct Gaussian helper 分开说明。

## parameters 层摘要

`utils/core/parameters.py` 只定义数据容器，不做单位换算、不构造 Hamiltonian、不调用 QuTiP。

用户侧正式输入：

```python
NLevelPhysicalParams(..., field=field)
```

常用 dataclass：

```text
NLevelPhysicalParams
RelaxationChannel
PureDephasingChannel
SolverParams
NLevelSolverParams
```

`RelaxationChannel` 描述 population relaxation：

```text
C_{to <- from} = sqrt(rate) |to><from|
```

`PureDephasingChannel` 描述 level projector pure dephasing：

```text
C_level^phi = sqrt(rate) |level><level|
```

`SolverParams` 是 `ParaNormalizer.normalize()` 输出的 solver 参数摘要。普通用户不应手动构造。

`NLevelSolverParams` 是 solver 内部 code-unit 参数容器。普通 example 应构造 `NLevelPhysicalParams`，再由 `ParaNormalizer` 生成内部参数。

## normalizer 层摘要

`ParaNormalizer` 负责真实物理单位到 solver code unit 的转换：

```text
NLevelPhysicalParams
-> validate physical units and shapes
-> derive fs^-1 quantities
-> choose time_scale_fs
-> build SolverParams
-> make one _CodeFieldAdapter
```

关键原则：

- 从 `field.reference_MV_per_cm` 读取电场参考尺度；
- 从 `field.normalization_rate_candidates_fs_inv` 读取 auto-scale 候选速率；
- 不从 `field.to_dict()` 读取 core 数值参数；
- 不根据 `Gaussian` / `CW` / `FieldSeries` / `TAField` / `TwoDESField` 写特殊分支。

field 缩放关系：

```text
coupling_matrix_fs_inv = mu_D * reference_MV_per_cm * constant
E_code(t) = E_phys(t) / reference_MV_per_cm
```

## solver / result 层摘要

当前求解入口：

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

主路径：

```text
NLevelPhysicalParams
-> ParaNormalizer.normalize()
-> _optical_codeparams_from_solverparams()
-> build_lab_hamiltonian()
-> build_c_ops()
-> qutip.mesolve()
-> DynamicsResult(mode="lab_exact")
```

`solver_mode="rwa"` 当前应视为 legacy guard；新 example 不应依赖它。`make_rotating_view()` 只是由 `lab_exact` 结果派生的 rotating-frame view，不是 RWA solver。

`DynamicsResult` 保存单条 density-matrix trajectory。它不负责绘图、保存文件、FFT response、absorption response、TA response 或参数扫描。

## spectroscopy 层摘要

`utils/spectroscopy/` 是当前谱学 observable 和 frequency-domain response 的位置。它只处理已经求解得到的 `rho(t)`、输入场 `E(t)`、polarization 和频域响应，不构造 Hamiltonian，也不调用 solver。

当前推荐主入口：

```python
lab_frame_absorption_response(...)
```

主输出字段：

```text
energy_eV
omega_fs_inv
absorption
omega_im_P_over_E
```

旧的较底层 helper 当前应写作：

```python
lab_frame_fft_response_legacy(...)
```

它不应作为当前 absorption-only 主入口宣传。

当前 `apply_time_window()` 支持：

```text
None
"none"
"hann"
```

若需要 `"hamming"` 或其它 window，应先扩展公共 helper，再让所有脚本调用同一套逻辑。

## IO / metadata 层摘要

IO 层只保存已经构造好的普通 `DynamicsResult` 数据、metadata 和 figure。典型函数：

```python
save_result_data(result, output_dir, save_npz=True, save_csv=True, save_json=True)
save_figure(fig, output_path, dpi=120)
save_result_case(result, output_dir, output_data=True, output_preview=False, ...)
QuantumResultIO
```

典型 case 输出：

```text
case_dir/
├─ data/
│  ├─ density.npz
│  ├─ components.csv
│  └─ populations.csv
├─ figs/
│  ├─ preview.png
│  └─ full.png
├─ meta.json
└─ debug_meta.json
```

`components.csv` 只保存 density matrix、population、coherence 和输入 field / drive 相关列。dipole expectation、polarization、FFT response 和 absorption quantities 应由 spectroscopy / analysis 层输出到单独文件。

当前 IO 不应写 piecewise series 输出，不应写 `materialize_full`。

## 本次合并更新摘要

本次文档口径更新重点：

1. 将旧的 `sjh_learn/` 口径更新为当前 `qudpy_sjh/`。
2. 将旧的 `utils/model.py`、`utils/solvers.py`、`utils/results.py`、`utils/analysis/observables.py` 口径更新为 `utils/core/` 与 `utils/spectroscopy/`。
3. 明确用户侧正式入口为 `NLevelPhysicalParams(..., field=field)`。
4. 明确 `field` 是 lab-frame physical field，而不是 solver-unit drive。
5. 补充 `TimeShiftedField` 的语义：`shift_fs > 0` 表示整体向更晚时间移动。
6. 区分 `make_ta_gaussian_field(...)` 与 template-based pump-probe helper 的 delay convention。
7. 将 spectroscopy 主入口更新为 `lab_frame_absorption_response(...)`。
8. 将 `lab_frame_fft_response_legacy(...)` 标记为较底层 / 历史兼容 helper。
9. 明确 solver 主线是 full-window `lab_exact`。
10. 明确标准结果是普通 `DynamicsResult`。
11. 明确 parameter scan / delay scan / batch workflow 属于 examples/scripts 层。
12. 明确普通 IO 只保存 full-window `DynamicsResult`。
13. 移除或降级 piecewise / dark / materialization / long-window benchmark 相关表述。

## 当前不进入主线的旧架构

以下旧架构不应写入 README 或主线文档：

```text
PieceDynamicsResultSeries
dark_propagation
piecewise_propagation
ActiveWindow
PropagationPiece
execute_piece_sequence
materialize_full
run_case(piecewise=...)
save_result_case(piecewise series)
long-window piecewise benchmark
complex transient_absorption scaffold
```
