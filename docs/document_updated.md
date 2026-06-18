# sjh_learn 当前架构记录

本文档记录 `sjh_learn` 当前 active code path 的职责边界和常用 API。它是顶层架构说明，不替代更细的 API 文档。field、normalizer、parameters、solver/result、spectroscopy 和 metadata schema 的详细说明应拆到对应专题文档中维护。

当前主线是：

```text
FieldPhyRoot / FieldPhySeries / TAField / TwoDESField
-> NLevelPhysicalParams(field=...)
-> ParaNormalizer
-> _CodeFieldAdapter
-> lab_exact solver
-> DynamicsResult
-> spectroscopy / analysis-layer observables
```

## 目录结构

当前建议按以下逻辑理解目录：

```text
sjh_learn/
├─ bin/
│  ├─ multilevel_demo.py
│  └─ n2_equivalence_check.py
├─ examples/
│  ├─ absorption/
│  │  ├─ absorption_01_chi_analytic.py
│  │  ├─ cw_pulse_absorption_compare.py
│  │  └─ three_level_absorption_lab_exact.py
│  └─ ta/
│     └─ ta_fieldseries_intrinsic_response.py
├─ scratch/
│  ├─ validate_*.py
│  └─ window_effect_absorption_compare.py
└─ utils/
   ├─ core/
   │  ├─ __init__.py
   │  ├─ parameters.py
   │  ├─ normalization.py
   │  ├─ model.py
   │  ├─ solvers.py
   │  ├─ results.py
   │  └─ config.py
   ├─ fields/
   │  ├─ __init__.py
   │  ├─ lab_fields.py
   │  └─ field_series.py
   ├─ spectroscopy/
   │  ├─ __init__.py
   │  ├─ observables.py
   │  ├─ spectra.py
   │  └─ rwa.py
   ├─ plotting.py
   ├─ io.py
   └─ checks.py
```

若本地目录仍保留旧位置，例如 `utils/model.py`、`utils/solvers.py`、`utils/results.py`、`utils/normalization.py` 或 `utils/analysis/observables.py`，这些应视为历史命名或迁移中痕迹；新文档应以 `utils/core/` 和 `utils/spectroscopy/` 为准。

## 核心约定

1. 用户侧光场使用 physical units：时间 `fs`，电场 `MV/cm`，能量 `eV`，偶极矩 `Debye`。
2. 正式入口是 `NLevelPhysicalParams(..., field=field)`；`field` 必须是 physical field 对象。
3. `NLevelPhysicalParams` 不再把 `field_MV_per_cm`、`laser_energy_eV`、`pulse_center_fs`、`pulse_sigma_fs` 作为顶层主接口。
4. `ParaNormalizer` 只负责单位换算和 code-unit scaling，不定义物理模型。
5. `model.py` / `solvers.py` 主路径只接收一个 `args["field"]` code-unit callable。
6. 多脉冲叠加只发生在 physical field 层，例如 `FieldPhySeries`、`TAField`、`TwoDESField`。
7. 标准结果对象是 `DynamicsResult`；一个 `DynamicsResult` 只表示一条 density-matrix trajectory。
8. 当前主线是 `lab_exact`；RWA solver-unit drive path 是 breaking-change 后的 legacy 路径，不应恢复。
9. 参数扫描、delay scan、power scan 和 batch run 属于上层 examples/scripts/workflow，不属于 core。
10. dipole expectation、polarization、FFT response、absorption-like spectrum 和 TA response 属于 spectroscopy / analysis 层，不属于 solver/result core。

## 文档分工

建议把文档拆成以下层次：

```text
current_architecture_zh.md      # 当前总架构和 breaking changes
document.md                     # 本文件：顶层职责边界和常用 API 索引
field_zh.md                     # physical field API
normalizer_zh.md                # ParaNormalizer 和 code-unit scaling
parameters_zh.md                # NLevelPhysicalParams / channel dataclass
solver_result_zh.md             # run_case / DynamicsResult
spectroscopy_zh.md              # observable / FFT response / window / mask
qudpy_metadata_schema_zh.md     # meta.json / debug_meta.json schema
```

其中 `document.md` 不应塞入所有字段细节；它只记录 active architecture 和脚本开发时最容易踩错的接口边界。具体字段、返回值和示例放到对应专题文档。

## fields/

`utils/fields/` 描述用户侧 lab-frame physical field。所有内置 field 都应继承或符合 `FieldPhyRoot` 接口。

### FieldPhyRoot 关键接口

```python
physical_E_MV_per_cm(t_fs)
reference_MV_per_cm
normalization_rate_candidates_fs_inv
to_dict()
```

含义：

- `physical_E_MV_per_cm(t_fs)` 返回真实 lab-frame 电场，单位 `MV/cm`。
- `reference_MV_per_cm` 是 normalizer 使用的电场归一化参考尺度，是 core 数值接口。
- `normalization_rate_candidates_fs_inv` 提供 auto-scale 的候选速率，只影响数值缩放，不改变物理模型。
- `to_dict()` 只用于 metadata / debug / rebuild，不是 core 数值接口。

不要从 `field.to_dict()` 读取 `reference_MV_per_cm` 或其它 core 数值参数。

### 内置 physical field

主要对象：

- `CarrierFieldPhysical`: `E(t_fs) = 2 E0 cos(omega_L t_fs + phase)`。
- `GaussianCarrierFieldPhysical`: `E(t_fs) = 2 E0 exp[-(t_fs-center)^2/(2 sigma^2)] cos(omega_L t_fs + phase)`。
- `FieldPhySeries`: physical-layer 多 field 线性叠加。
- `TAField`: pump-probe / transient absorption 常用 field container。
- `TwoDESField`: 2DES 常用 three-pulse field container。

常用 helper：

```python
make_default_carrier_field(...)
make_default_gaussian_carrier_field(...)
make_ta_gaussian_field(...)
make_twodes_gaussian_field(...)
iter_ta_gaussian_fields(...)
iter_twodes_gaussian_fields(...)
```

### TAField 约定

`TAField` 是 physical-layer 多脉冲 field。它至少应满足：

```text
1. 有 name 为 "probe" 的 subfield；
2. 有 probe_delay_fs / probe_delay property，表示 t_probe - t_pump；
3. TAField["probe"] 能提取 probe 子场；
4. TAField 本身仍是 FieldPhyRoot，可直接传入 NLevelPhysicalParams(field=...）。
```

`make_ta_gaussian_field()` 当前语义是：

```text
probe_center_fs = pump_center_fs + probe_delay_fs
```

即默认 pump 固定、probe 随 delay 移动。如果某个 workflow 希望 probe 固定、pump 移动，不需要修改 helper；在脚本中设置：

```python
probe_center_fs = 0.0
pump_center_fs = probe_center_fs - probe_delay_fs
field = make_ta_gaussian_field(
    probe_delay_fs=probe_delay_fs,
    pump_center_fs=pump_center_fs,
    ...
)
```

这属于 workflow 调用策略，不应反向修改 `TAField` 或 `make_ta_gaussian_field()` 的核心语义。

## parameters.py

`utils/core/parameters.py` 只定义数据容器，不做单位换算、不构造 Hamiltonian、不调用 QuTiP。

推荐导入：

```python
from qudpy_sjh.utils.core import (
    NLevelPhysicalParams,
    RelaxationChannel,
    PureDephasingChannel,
    SolverParams,
    NLevelSolverParams,
)
```

### NLevelPhysicalParams

`NLevelPhysicalParams` 是当前用户侧标准物理系统输入。two-level system 是普通 `N=2` system，multilevel system 是普通 `N>2` system。

核心字段：

```python
NLevelPhysicalParams(
    energies_eV=(...),
    dipole_matrix_D=((...), ...),
    t_start_fs=...,
    t_end_fs=...,
    dt_fs=...,
    field=field,
    basis=(...) or None,
    relaxation_channels=(...),
    pure_dephasing_channels=(...),
    solver_mode="lab_exact",
    input_description=...,
    input_metadata={...},
)
```

单位：

| 字段 | 单位 / 类型 |
|---|---|
| `energies_eV` | eV |
| `dipole_matrix_D` | Debye |
| `t_start_fs`, `t_end_fs`, `dt_fs` | fs |
| `field` | `FieldPhyRoot` |
| `basis` | optional state labels |
| `relaxation_channels` | tuple of `RelaxationChannel` |
| `pure_dephasing_channels` | tuple of `PureDephasingChannel` |
| `solver_mode` | 当前主线为 `"lab_exact"` |

`dimension` 返回能级数量 N。`energy_gap_eV` 是 0->1 兼容属性；对 N>2，它不代表唯一能隙。

### RelaxationChannel

`RelaxationChannel` 描述 population relaxation：

```text
C_{to <- from} = sqrt(rate) |to><from|
```

active API 使用 dataclass 对象，不是 dict：

```python
RelaxationChannel(
    name="relaxation_1_to_0",
    from_level=1,
    to_level=0,
    T1_fs=1000.0,
)
```

或：

```python
RelaxationChannel(
    name="relaxation_1_to_0",
    from_level=1,
    to_level=0,
    rate_fs_inv=1.0e-3,
)
```

`T1_fs` 和 `rate_fs_inv` 都可指定；二者都给出时，normalizer 优先使用 `rate_fs_inv`。

错误写法：

```python
relaxation_channels=(
    {"from_level": 1, "to_level": 0, "T1_fs": 1000.0},
)
```

dict 形式适合 JSON metadata，不是 `NLevelPhysicalParams` 的 active input。

### PureDephasingChannel

`PureDephasingChannel` 描述 level projector pure dephasing：

```text
C_level^phi = sqrt(rate) |level><level|
```

active API 使用 dataclass 对象：

```python
PureDephasingChannel(
    name="pure_dephasing_level_1",
    level=1,
    Tphi_fs=120.0,
)
```

或：

```python
PureDephasingChannel(
    name="pure_dephasing_level_1",
    level=1,
    rate_fs_inv=1.0 / 120.0,
)
```

`Tphi_fs` 和 `rate_fs_inv` 都可指定；二者都给出时，normalizer 优先使用 `rate_fs_inv`。

### SolverParams 与 NLevelSolverParams

`SolverParams` 是 `ParaNormalizer.normalize()` 输出的 solver 参数摘要，包含 fs^-1 和 code-unit 两套量，主要用于 solver 构造、debug metadata 和 sanity check。普通用户不应手动构造。

`NLevelSolverParams` 是 solver 内部 code-unit 参数容器。普通 example 应构造 `NLevelPhysicalParams`，再由 `ParaNormalizer` 生成内部参数，不应直接构造 `NLevelSolverParams` 作为用户入口。

## normalizer.py

`ParaNormalizer` 负责真实物理单位到 solver code unit 的转换。它不定义物理模型，不根据 field 类型写特殊物理分支。

主职责：

```text
NLevelPhysicalParams
-> validate physical units and shapes
-> derive fs^-1 quantities
-> choose time_scale_fs
-> build SolverParams
-> make one _CodeFieldAdapter
```

关键原则：

- 从 `field.reference_MV_per_cm` 读取电场参考尺度。
- 从 `field.normalization_rate_candidates_fs_inv` 读取 auto-scale 候选速率。
- 不从 `field.to_dict()` 读取 core 数值参数。
- 不根据 `Gaussian` / `CW` / `FieldSeries` / `TAField` / `TwoDESField` 写特殊分支。
- `omega_L_fs_inv` 和 `detuning_fs_inv` 可作为 optional metadata/debug 信息，但不应成为 `lab_exact` 主路径依赖。

`field.reference_MV_per_cm` 与 code field 缩放必须成对使用：

```text
coupling_matrix_fs_inv = mu_D * reference_MV_per_cm * constant
E_code(t) = E_phys(t) / reference_MV_per_cm
```

否则 Hamiltonian 中的 `mu * E(t)` 会重复缩放或漏缩放。

## model.py / solvers.py

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

run_cases(physical_params_list, normalizer=None) -> list[DynamicsResult]

make_rotating_view(lab_result) -> DynamicsResult
```

`run_case()` 主路径：

```text
NLevelPhysicalParams
-> ParaNormalizer.normalize()
-> _optical_codeparams_from_solverparams()
-> build_lab_hamiltonian()
-> build_c_ops()
-> qutip.mesolve()
-> DynamicsResult(mode="lab_exact")
```

`model.py` / `solvers.py` 只接收单个 code-unit callable：

```python
args={"field": field}
```

不要恢复：

```text
args["fields"]
fields/solver_inputs
fields/legacy_solver_inputs
lab_fields_code
rwa_drives_code
```

`solver_mode="rwa"` 当前应视为 legacy guard；新 example 不应依赖它。`make_rotating_view()` 只是由 `lab_exact` 结果派生的 rotating-frame view，不是 RWA solver。

checkpoint 说明：

- `load_ckp` 存在且 `force_run=False` 时直接读取。
- `save_ckp` 或 `load_ckp` 可作为保存路径。
- `.ckp` 是内部 checkpoint / 后处理缓存，不保证跨版本长期稳定，不替代 `density.npz`、`components.csv`、`meta.json` 和 `debug_meta.json`。
- 不要加载不可信来源的 pickle checkpoint。

## results.py

`DynamicsResult` 保存单条 density-matrix trajectory。

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

常用方法：

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

`DynamicsResult` 不负责：

```text
求解微分方程
绘图
保存文件
判断物理结论
固定 two-level 的 rho_11 / rho_01 语义
dipole expectation
polarization
FFT response
absorption-like response
TA response
```

这些属于 solver、plotting、io 或 spectroscopy / example 层。

## spectroscopy/

`utils/spectroscopy/` 是当前谱学 observable 和 frequency-domain response 的主要位置。它只处理已经求解得到的 `rho(t)`、输入场 `E(t)` / drive、polarization 和频域响应，不构造 Hamiltonian，也不调用 solver。

推荐导入：

```python
from qudpy_sjh.utils.spectroscopy import (
    dipole_expectation_D,
    polarization_C_per_m2,
    apply_time_window,
    safe_complex_ratio,
    lab_frame_fft_response_legacy,
)
```

### observables.py

`dipole_expectation_D(rho_t, dipole_matrix_D)` 计算：

```text
p(t) = Tr[rho(t) mu] = sum_ij rho_ij(t) mu_ji
```

实现约定等价于：

```python
np.einsum("tij,ji->t", rho_t, dipole_matrix_D)
```

`polarization_C_per_m2(rho_t, dipole_matrix_D, number_density_m3)` 计算：

```text
P_macro(t) = number_density_m3 * p_single(t) * DEBYE_TO_C_M
```

`number_density_m3` 必须显式传入，单位 `m^-3`。observable 必须使用用户侧物理偶极矩 `dipole_matrix_D`，不能使用 `coupling_matrix_code`。

### spectra.py

`lab_frame_fft_response()` 是 lab-frame pulse response 的公共谱计算 helper。它已经负责：

```text
uniform dt 检查
subtract mean
window
zero padding
FFT
frequency axis
omega axis
energy axis
P_fft / E_fft
rhoij_fft / E_fft
positive-frequency selection
small-denominator mask
```

因此 examples / scratch 脚本不应重复实现 FFT、frequency axis、positive-frequency selection 或小分母 mask。

当前接口：

```python
lab_frame_fft_response(
    *,
    t_fs,
    E_MV_per_cm,
    P_C_per_m2,
    rhoij=None,
    rho12=None,
    window="hann",
    subtract_mean=True,
    rel_threshold=1e-6,
    zero_padding_factor=4,
)
```

当前 `apply_time_window()` 支持：

```text
None
"none"
"hann"
```

若需要 `"hamming"` 或其它 window，应先扩展 `apply_time_window()` 这个公共 helper，再让所有脚本调用同一套逻辑；不要在单个 diagnostic 脚本里手写第二套 window 实现。

`lab_frame_fft_response()` 返回的是 mask 后的有效正频率轴，而不是原始完整 FFT 轴。因为它使用：

```text
mask = positive_frequency & valid_E
```

其中 `valid_E` 来自 `safe_complex_ratio()` 对 `E_fft` 的小分母筛选。不同输入场、不同 window、不同 field 时间位置或不同 `rel_threshold` 下，返回的 `energy_eV` 长度可能不同。

常用返回键：

```text
omega_fs_inv
energy_eV
E_fft
P_fft
rhoij_fft
P_over_E
rhoij_over_E
abs_E_fft
abs_rhoij_over_E
im_rhoij_over_E
omega_im_rhoij_over_E
neg_omega_im_P_over_E
```

其中 `neg_omega_im_P_over_E` 是：

```text
- omega * Im[P(omega) / E(omega)]
```

如果 example 使用：

```text
omega * Im[P(omega) / E(omega)]
```

应在 example metadata 中明确符号约定。

`rwa_fft_response()` 只保留 legacy RWA-like trajectory 的诊断性后处理，不是 core 主线依赖。

## plotting.py

绘图函数只返回 matplotlib 句柄，不直接保存文件：

```python
plot_drive(result, ax=None, ...)
plot_field(field, times, ax=None, ...)
plot_populations(result, ax=None, ...)
plot_coherences(result, ax=None, ...)
plot_density_components(result, axes=None, include_drive=False, ...)
plot_multilevel_components(result, axes=None, ...)
build_preview_figure(result, ...)
```

如果传入 `ax/axes`，函数画到传入坐标轴；否则内部创建 figure。保存 figure 属于 `io.py` 或上层脚本职责。

## io.py

IO 层只保存已经构造好的数据、metadata 和 figure。典型函数：

```python
save_result_data(result, output_dir, save_npz=True, save_csv=True, save_json=True)
save_figure(fig, output_path, dpi=120)
save_result_case(result, output_dir, output_data=True, output_preview=False, preview_fig=None, full_fig=None, ...)
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

`components.csv` 只保存 density matrix、population、coherence 和输入 field/drive 相关列。dipole expectation、polarization、FFT response 和 absorption-like quantities 应由 spectroscopy / analysis 层输出到单独文件，例如 `analysis_components.csv` 或 `fft_response.csv`。

## Metadata

`meta.json` 应使用紧凑、物理单位优先的结构；`debug_meta.json` 保存 code-unit 和内部诊断信息。默认 schema 不应假设只有一个 transition、一个 energy gap、一个 detuning、一个 Rabi frequency 或一个 dephasing rate。

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

旧顶层 block 不应再出现在 `meta.json`：

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

`field` block 只描述 optical input field，不能包含：

```text
dipole_matrix_D
dipoles_D
transitions_eV
transition_table
relaxation_channels
pure_dephasing_channels
user_metadata
```

规范化的 matter system 信息应放在 `system`；dissipation 信息应放在 `dissipation`；用户手写备注可放在 `user_input.metadata`。

## 顶层 examples / workflow

### absorption examples

`examples/absorption/` 放相对稳定的 absorption 示例。它们可以调用 `lab_frame_fft_response()` 生成 absorption-like spectrum，但不应在 example 里重复实现 FFT 轴、window、mask 或 complex division。

### TA intrinsic response example

TA intrinsic response 属于模拟结果的直接 spectroscopy observable，适合放在 QuDPy example / scratch，而不是 UFANSYS 后处理库。

定义：

```text
S_TA(omega, delay)
=
omega * Im[P_pump_probe(omega, delay) / E_probe(omega, delay)]
-
omega * Im[P_probe_only(omega, delay) / E_probe(omega, delay)]
```

其中：

```text
P(t) = Tr[rho(t) mu]
E_probe(t) = TAField["probe"](t_fs)
```

第一版 workflow：

```text
for each delay:
    construct TAField
    run pump+probe with full TAField
    run probe-only with TAField["probe"]
    compute P_pump_probe(t), P_probe_only(t)
    call lab_frame_fft_response() with the same E_probe(t)
    subtract spectra
```

更稳定的 delay map workflow 建议：

```text
固定 probe center
通过 pump_center_fs = probe_center_fs - probe_delay_fs 扫描 delay
probe-only 只跑一次
所有 delay 共用同一 time grid
E_probe(omega) 和 valid_E mask 尽量只定义一次
```

第一版不需要 pump-only。pump-only 可作为后续诊断项，用于检查 pump-induced residual polarization background。

该 TA response 是材料本征响应，忽略传播效应、样品厚度、transmitted intensity 和 detector response。这些实验链路因素不属于当前 QuDPy 模型范围。

### window diagnostic

比较 window 对 absorption-like spectrum 的影响时，应复用同一个 `lab_frame_fft_response()`，只改变 `window` 参数。不要在 diagnostic 脚本中手写第二套 FFT 或 window 逻辑。当前公共 helper 只支持 `None` / `"none"` / `"hann"`；若要比较 Hamming，应先扩展 `apply_time_window()`。

### workflow 边界

参数扫描、delay scan、power scan、batch run、figure layout、summary CSV 和 map 生成属于上层 examples/scripts/workflow。不要把这些需求反向塞进 core：

```text
不要恢复 ParameterSweep
不要恢复 run_parameter_sweep
不要让 solver 接收多个 field callable
不要让 normalizer 根据具体 field class 写 workflow 特例
```

## Unit Conventions

- 时间：用户输入 `fs`；solver 内部为 dimensionless code time。
- 能量：用户输入 `eV`；normalizer 转为 `fs^-1` 再转为 code unit。
- 偶极矩：用户输入 `Debye`。
- 电场：用户输入 `MV/cm`。
- Rabi / coupling：由 `mu_D * reference_MV_per_cm / hbar` 得到 `fs^-1`，再转为 code unit。
- density matrix、population、coherence 均为无量纲。
- collapse operator rate 由 `T1_fs` / `Tphi_fs` 或 `rate_fs_inv` 得到，再转为 code unit。
- Plotting、CSV export 和 example summary 默认使用 physical units。
- Code-unit 输出只应作为 metadata 或显式 diagnostic 字段保存。

## RWA Legacy Status

RWA solver-unit drive path 已删除或禁用。当前主线只维护 lab-frame / exact solver。

不要恢复：

```text
CodeCarrierField
CodeGaussianCarrierField
CodeCompositeField
CodeConstantDrive
CodeGaussianDrive
solver_input_from_dict
fields/solver_inputs
fields/legacy_solver_inputs
lab_fields_code
rwa_drives_code
PhysicalParameterSweep
run_physical_parameter_sweep
ParameterSweep
run_parameter_sweep
parameter_sweep
args["fields"]
```

如果旧 absorption compare example 中还有 RWA compare 语义，不应为了它恢复 RWA；后续可以单独改成 lab_exact-only example 或继续标记为 legacy。

## Multi-Level Compatibility Notes

- `NLevelPhysicalParams -> ParaNormalizer -> NLevelSolverParams -> model.py -> solvers.py -> DynamicsResult` 是统一 N-level route。
- two-level system 是普通 `N=2`；multilevel system 是普通 `N>2`。
- `components_dataframe()` 和 `components.csv` 是 dimension-aware，保存所有 diagonal populations 和所有 upper-triangular coherences。
- `populations.csv` 包含所有 diagonal elements。
- `density.npz` 包含完整 density-matrix trajectory。
- `DynamicsResult` 不应暴露 two-level-only `components()` helper。
- two-level demos 或 legacy comparison scripts 可以在 example-local helper 中提取 `rho_11` / `rho_01`，但不要把该语义放进 generic result export。
- 对 `dimension > 2`，metadata 不应使用单一全局 `energy_gap_eV`、`detuning_eV`、`rabi_fs_inv` 或 `gamma2_fs_inv` 作为完整系统描述。

## 当前开发建议

近期与 TA / absorption diagnostic 相关的开发，应优先在 example / scratch 层完成：

```text
1. 修正 TA intrinsic response 脚本的 probe-anchored 调用方式。
2. probe-only reference 尽量只运行一次。
3. 增加 diagnostic figure：E_probe(t)、P(t)、|E_probe(omega)|、mask、lineout。
4. window sensitivity 通过 lab_frame_fft_response(window=...) 比较。
5. 若要支持 hamming window，先扩展 spectroscopy.apply_time_window()。
6. 不改 lab_exact Hamiltonian、Lindblad 逻辑、ParaNormalizer 主接口或 FieldPhyRoot 主接口。
```
