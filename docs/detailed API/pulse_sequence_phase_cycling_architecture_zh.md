# Pulse Sequence 与 Phase Cycling 架构说明

本文档描述 QuDPy 中 TA / multi-pulse / phase-cycling 的长期分层架构。
当前已经建立通用 multi-pulse single-run scaffold，并在 Milestone 2 中
为 `CarrierEnvelopeField` 增加正式 field-level phase override API。
Milestone 3 已新增 generic `SingleRunPlan / ReadoutSpec`，用于把一次
concrete pulse sequence 接入 `run_case` 并执行可选通用 readout。
Milestone 4 已新增 generic `PhaseGrid / PhaseCyclingPlan`，用于对任意
`SingleRunPlan` 执行 phase grid 并按 `target_phase_vector` 做 Fourier
projection。Milestone 4.5 已新增 generic projected-result bundle，用于
把 Fourier-projected signal 与非投影 axis metadata 配对。具体 TA /
2DES experiment recipe 仍属于后续阶段。

## 总体原则

默认模拟流程只负责一次给定 field configuration 的传播与 readout：

```text
field construction
-> system propagation
-> polarization / output field
-> spectrum / readout
```

默认流程不内置 phase cycling，也不默认假设 TA、2DES 或其他具体实验。
phase cycling 属于更高层实验逻辑，由独立 cycler / experiment runner
负责。底层保留 pulse 的 `phase_tag`、`phase_rad` 和
`independent_phase` metadata；当 field backend 支持 phase override 时，
`phase_rad` 会真实改变构造出的场。

## 三层结构

### Layer 1: generic multi-pulse single-run simulation

Layer 1 只描述一次具体 field configuration：

```text
PulseSpec / FieldGroupSpec / PulseSequenceSpec
-> concrete FieldPhySeries
-> run_case
-> readout / spectrum
```

本层不关心 delay grid、TA subtraction、2DES target channel 或 phase
projection。`PulseSequenceSpec.build_field(...)` 只把任意数量的 pulse /
field group 构造成一次可传给 `NLevelPhysicalParams.field` 的
`FieldPhySeries`。

### Layer 2: generic phase cycler

Layer 2 建立在 Layer 1 的 phase metadata 上：

```text
collect phase_tags
-> generate phase grid
-> apply phase vector to pulse sequence
-> run many single-run cases
-> Fourier projection by target_phase_vector
```

generic cycler 不写死 TA 或 2DES。它只根据 `phase_tags`、
`target_phase_vector` 和 phase grid 重复调用 `SingleRunPlan`，提取指定
readout array，并做 Fourier projection。

### Layer 3: experiment recipes

Layer 3 定义具体非线性光谱实验：

- TA recipe：pump/probe pulse sequence、pump-on/off 或 probe-only
  reference、delay-energy map、`S_TA = S_pump_probe - S_probe_only`。
- 2DES recipe：pulse1/pulse2/pulse3/readout sequence、rephasing /
  non-rephasing / double-quantum target vectors、`t1/t2/t3` grid 和二维
  Fourier transform。
- 其他 nonlinear spectroscopy recipe 可复用同一 pulse-sequence 与
  phase-cycling 基础层。

## 为什么本轮先做 pulse-sequence scaffold

phase cycling 依赖底层 pulse sequence 的 phase_tag 语义。如果没有
`PulseSpec / FieldGroupSpec / PulseSequenceSpec`，phase cycler 只能是
孤立的数学工具，无法可靠接入真实 field construction。

本轮先让框架能够表达：

```text
任意 pulse sequence 的一次模拟
```

后续 cycler 再对这些 pulse 或 field group 的 `phase_tag` 做扫描和
Fourier projection。

## 当前 TA prototype 的定位

`qudpy_sjh/experiments/ta/ta_settings.py`、
`ta_case_plan.py`、`ta_result.py` 是 experimental TA recipe v1 prototype。
它们已经包含 pump/probe、probe-only reference、
`S_TA = S_pump_probe - S_probe_only`、TA map 和 TA IO 等具体 TA 语义。

这些文件的定位是：

- 当前未被 phase-cycling validation demo 使用；
- 作为 historical reference、IO/export behavior reference、migration
  comparison 和 regression reference；
- 不是 generic pulse-sequence simulation framework；
- 不是当前 TA recipe v2 主线；
- 作为 frozen reference 暂时保留；
- 当前不默认执行 phase cycling；
- 新开发优先使用 `qudpy_sjh.experiments.pulse_sequence` 和
  `qudpy_sjh.experiments.ta.ta_recipe_v2`。

## Legacy TA v1 prototype audit / deprecation policy

legacy TA v1 prototype 文件包括：

- `qudpy_sjh/experiments/ta/ta_settings.py`
- `qudpy_sjh/experiments/ta/ta_case_plan.py`
- `qudpy_sjh/experiments/ta/ta_result.py`

当前策略：

- v1 保留为 legacy / frozen reference；
- v1 不删除、不移动、不重构执行逻辑；
- v1 不在运行时发 `DeprecationWarning`，避免污染测试输出；
- v2 主线位于 `qudpy_sjh/experiments/ta/ta_recipe_v2.py`，并复用 generic
  pulse-sequence / single-run / phase-cycling framework；
- 待 v2 IO/export 和 old demo migration 完成后，再考虑把 v1 移到
  `ta/legacy/` 或删除。

功能覆盖关系：

| 功能 | v1 prototype | v2 current status | 处理建议 |
| --- | --- | --- | --- |
| settings | `TASettings` 已有完整 v1 配置 | v2 以 plan/spec/dataclass 小对象表达当前最小主线 | 新开发优先 v2；v1 作为配置参考 |
| single-run orchestration | `TAPlan` / `TADelayCasePlan` 直接编排 v1 delay case | `SingleRunPlan` + `TASingleDelayPlan` 已覆盖单次 concrete run 编排 | 新开发走 generic single-run + v2 recipe |
| probe-only reference | v1 已内置 probe-only reference | v2 `TASingleDelayPlan.make_probe_only_plan()` 已覆盖 | v2 主线继续沿用显式 reference |
| pump-probe pair | v1 已内置 pump-probe case | v2 `TASingleDelayPairResult` 已覆盖 | v2 主线继续扩展 |
| TA subtraction | v1 已内置 `pump_probe - probe_only` | v2 `compute_ta_contrast()` 已覆盖单 delay subtraction | 新验证优先 v2 |
| delay scan | v1 已有 full delay scan 与输出策略 | v2 已有 delay scan scaffold 和 delay × energy map stack | v1 暂作 full-output reference；v2 后续补 IO/export |
| phase cycling | v1 不默认执行 | v2 TA 尚未接入 phase cycling；generic `PhaseCyclingPlan` 已存在 | 后续单独设计 optional phase-cycling TA |
| IO/export | v1 `TAResultIO` 已有默认导出 | v2 暂无 TAResultIO v2 | v1 作为 IO/export reference |
| old demo migration | v1 demo 仍使用旧路径 | v2 尚未迁移旧 demo | v2 IO/export 完成后再迁移 |

package-level API 导出策略：

- legacy aliases：`LegacyTASettings`、`LegacyTAPlan`、
  `LegacyTADelayScanPlan`、`LegacyTAResult`、`LegacyTAResultIO`；
- v1 原裸名保持稳定：例如 package 顶层 `TADelayScanPlan` 继续指向
  `ta_case_plan.TADelayScanPlan`；
- v2 scan 显式 aliases：`TADelayScanPlanV2`、`TADelayScanResultV2`、
  `TADelayScanMapV2`；
- 不允许 v2 在 package 顶层静默覆盖已有 legacy 裸名；
- 如需 v2 原类名，可直接从 `qudpy_sjh.experiments.ta.ta_recipe_v2`
  导入。

## phase_tag 语义

`phase_tag` 属于 `PulseSpec` 或 `FieldGroupSpec`，默认不属于单个 carrier
component。

默认规则：

- 一个 physical pulse 对应一个 `phase_tag`；
- `FieldGroupSpec` 可以包含多个 carrier / component / pulse template；
- 多载波或宽带 field group 不自动产生多个 `phase_tag`；
- 只有用户显式声明 `independent_phase=True` 的 pulse 或 group，才作为
  上层 cycler 的独立相位控制对象；
- 如果 `FieldGroupSpec.phase_tag` 非空，group-level phase tag 优先，
  group 内部多个 component 作为一个 coherent physical group 共享该 tag；
  此时内部 pulse 不再暴露独立 `phase_tag`，以避免 group phase 与
  component phase 同时进入同一个 `phase_vector`；
- `target_phase_vector` 是 cycler 的 phase-channel 控制入口；
- 当前阶段只预留 `target_phase_vector` 语义，不实现完整 runner。

因此：

- 宽带 pulse；
- FROG 真实包络；
- 多载波 pump/probe；
- frequency-comb-like field group；

默认都作为一个相干 physical pulse / field group 处理，共享一个 global
phase tag。只有实验设计上具有独立相位控制的 component 才显式拆分。

## system metadata 与 phase cycling 的边界

`detuning` 是 system-field 关系，不是 matter system 自身属性。
`transition_table` 和 `coupling_fs_inv` 是诊断或归一化派生信息。
phase cycling 是 experiment-level Fourier projection。

因此 human-facing `meta.json` 中的 bottom-level `params.system` 只保存
matter system 定义和 dissipation。默认 propagation 只接受当前 concrete
field configuration，不关心 phase projection。

## TA recipe 示例

TA recipe 可以由 Layer 3 定义：

```text
pump pulse
probe pulse
delay_fs
probe-only reference 或 pump-on/off reference
S_TA(omega, delay) = S_pump_probe(omega, delay) - S_probe_only(omega)
```

phase cycling 可作为上层可选 recipe：

```text
target_phase_vector = {"pump": 0, "probe": 1}
```

或按具体实验约定使用 pump net 0 + probe +1 channel。这个 target vector
属于 recipe / cycler 层，不属于 `run_case` 或 `DynamicsResult`。

## 2DES recipe 示例

2DES recipe 可以定义：

```text
pulse1, pulse2, pulse3, readout
phase tags: phi1, phi2, phi3, phi_lo
t1, t2, t3 grid
```

常见 target vectors：

```text
rephasing       = {"pulse1": -1, "pulse2": +1, "pulse3": +1, "readout": -1}
non_rephasing   = {"pulse1": +1, "pulse2": -1, "pulse3": +1, "readout": -1}
double_quantum  = {"pulse1": +1, "pulse2": +1, "pulse3": -1, "readout": -1}
```

具体符号约定由 2DES recipe 文档固定。二维 Fourier transform 属于 2DES
readout/output 层，不属于 bottom-level pulse sequence。

## 当前阶段未实现的内容

当前阶段不实现：

- phase-cycled TA subtraction / full phase-cycling TA recipe；
- TAResultIO v2；
- OD / relative difference / normalization variants；
- interpolation / resampling / smoothing；
- TA recipe v2 plotting；
- TA recipe v2 npz / csv export；
- 2DES recipe；
- rephasing / non-rephasing / double-quantum recipe；
- 2D Fourier transform；
- phase-case checkpoint path builder；
- piecewise / dark propagation；
- `materialize_full`；
- `DynamicsResult` 重写；
- `run_case` 行为改变；
- 旧 TA demo 迁移；
- 默认 TAPlan phase cycling；
- 将 `qudpy_sjh/experiments/ta/` prototype 改造成通用框架。

## 当前新增 scaffold

当前新增模块：

```text
qudpy_sjh/experiments/pulse_sequence/
  __init__.py
  phase_cycling.py
  pulse_sequence.py
  single_run.py
qudpy_sjh/experiments/ta/
  ta_recipe_v2.py
```

主要 API：

```text
validate_phase_tag
validate_pulse_name
normalize_phase_vector
is_supported_phase_backend
supports_phase_override
PulseSpec
FieldGroupSpec
PulseSequenceSpec
SingleRunFieldPlan
ReadoutSpec
SingleRunReadoutResult
SingleRunCheckpointSettings
SingleRunPlan
SingleRunResult
compute_single_run_readout
PhaseGrid
PhaseProjectionSpec
PhaseCaseRecord
PhaseCyclingPlan
PhaseCyclingResult
AxisMetadataSpec
ProjectedReadoutBundle
normalize_target_phase_vector
phase_projection_weight
fourier_project_phase_cases
build_uniform_phase_grid
extract_single_run_quantity
build_projected_readout_bundle
TADelayCenters
TAReadoutBundle
TASingleDelayPlan
TASingleDelayPairResult
extract_ta_absorption_bundle
TASubtractionSpec
TAContrastResult
validate_ta_readout_bundle_axes
compute_ta_contrast
TAPhaseCyclingSpec
TAPhaseCycledPumpProbeResult
build_ta_pump_probe_phase_cycling_plan
build_ta_phase_cycled_pump_probe_bundle
ta_recipe_v2.TADelayScanMap
validate_ta_contrast_axes_for_scan
build_ta_delay_scan_map
ta_recipe_v2.TADelayScanPlan
ta_recipe_v2.TADelayScanResult
package alias: TADelayScanMapV2
package alias: TADelayScanPlanV2
package alias: TADelayScanResultV2
```

这些 API 不依赖 matplotlib，也不把框架写死为 TA / 2DES。除
`SingleRunPlan.execute()`、`TASingleDelayPlan.execute_pair()` 和
`ta_recipe_v2.TADelayScanPlan.execute()` 这些显式执行入口会调用
`run_case` 外，其余 helper 主要负责构造 field、组织 case、验证 axis
或打包结果。

`FieldPhyRoot` 仍是通用抽象接口；`CarrierEnvelopeField` 是当前
phase-aware workflow 的首个正式支持 backend。它提供：

```text
positive_frequency_E_MV_per_cm(t_fs)
with_phase(phase_rad)
phase_shifted(delta_phase_rad)
```

`with_phase(phase_rad)` 使用 absolute phase 语义，返回新的 field 对象。
正频复场满足：

```text
E_positive(t; phi) = exp(i * phi) * E_positive(t; 0)
```

real lab-frame field 由 shifted positive-frequency field 取实部得到：

```text
E_real(t) = 2 * Re[E_positive(t)]
```

`PulseSpec.shifted(...)` 在 backend 支持 `with_phase` 时会真实应用
phase override，并在 metadata 中记录：

```text
phase_rad
base_phase_rad
phase_tag
phase_override_applied
phase_override_note
```

对不支持正式 phase API 的自定义 field，默认 strict 行为会拒绝非零
phase override，避免 phase-cycling workflow 中不同 phase run 的真实
waveform 完全相同却被误认为已经投影。

## Milestone 3：generic SingleRunPlan / ReadoutSpec

当前 `single_run.py` 提供一个实验无关的 single-run 执行层：

```text
SingleRunFieldPlan
-> build concrete FieldPhySeries
-> replace(base_params, field=field)
-> run_case
-> optional generic readout
-> SingleRunResult
```

`SingleRunPlan` 只替换 `NLevelPhysicalParams.field` 以及用户记录用的
`input_description / input_metadata`。它不改变 matter system、time grid、
solver mode、relaxation、pure dephasing、normalization 或 solver 行为。

`input_metadata["single_run_workflow"]` 记录本次 single-run 的
`case_name`、`field_plan`、`readout`、`phase_vector` 和 `centers_fs`，
用于后续 recipe / cycler 追踪一次具体 field configuration。

`ReadoutSpec` 当前支持三种模式：

- `mode="none"`：不执行 readout，`SingleRunResult.readout` 为 `None`；
- `mode="polarization"`：计算宏观 `polarization_C_per_m2(t)`；
- `mode="absorption"`：计算 polarization，并使用指定 readout field 作为
  denominator 调用 `lab_frame_absorption_response`。

`readout_field_name=None` 表示使用 total field；非空字符串表示从
`FieldPhySeries` 中按名称选择 subfield，例如未来 TA recipe 可以选择
`readout_field_name="probe"`。generic single-run 层不把 `"probe"` 作为
默认值，也不假设 probe-only / pump-probe reference 存在。

当前 absorption-like readout 只输出通用谱响应：

```text
polarization_C_per_m2
readout_field_MV_per_cm
lab_frame_absorption_response(...)
```

它不计算：

- `S_TA = S_pump_probe - S_probe_only`；
- phase average；
- phase grid；
- Fourier projection；
- 2DES spectrum。

这些操作属于后续 Layer 2 / Layer 3。

## Milestone 4：generic PhaseCyclingPlan / Fourier projection

当前 `phase_cycling.py` 提供一个实验无关的 phase-grid runner：

```text
PhaseGrid
-> phase vectors
-> clone SingleRunPlan per phase case
-> SingleRunPlan.execute()
-> extract selected readout array
-> stack phase_cases x ...
-> Fourier projection by target_phase_vector
-> PhaseCyclingResult
```

`PhaseGrid` 只表达 phase tags 到 phase samples 的笛卡尔积。`PhaseCyclingPlan`
只重复调用 `SingleRunPlan`，并通过 `PhaseProjectionSpec.quantity` 指定要
投影的 readout array，例如：

```text
readout.polarization_C_per_m2
readout.readout_field_MV_per_cm
readout.spectrum.absorption
```

Fourier projection 的默认约定为：

```text
weight = exp(-i * sum_k target_phase_vector[k] * phase_vector[k])
```

`target_phase_vector` 是 phase-channel 控制入口。它的系数必须是整数；
例如 `{"probe": 1}` 表示提取随 probe phase 带 `exp(+i phi_probe)` 的
分量。projection 保留 complex 结果，不强制取实部。

`PhaseCyclingPlan` 当前不保存文件，也不自动生成 checkpoint path。若
`base_plan.checkpoint.enabled=True`，当前实现直接拒绝，以避免多个 phase
case 默认覆盖或误读同一个 checkpoint。

generic phase cycler 当前不计算：

- `S_TA = S_pump_probe - S_probe_only`；
- pump-probe / probe-only orchestration；
- delay-energy map；
- 2DES rephasing / non-rephasing / double-quantum recipe；
- `t1/t2/t3` grid；
- 2D Fourier transform。

这些操作属于后续 Layer 3 recipe。未来 TA recipe 负责 delay scan、
probe-only reference 和 pump-probe difference；未来 2DES recipe 负责
`t1/t2/t3` grid 和二维 FFT。

## Milestone 4.5：projected-result bundle 与 axis metadata

`PhaseCyclingResult` 负责保存 phase-cycling runner 的核心结果：

- 每个 phase case 的 values；
- Fourier-projected signal；
- phase vectors；
- target phase vector；
- projection metadata；
- phase grid metadata。

recipe 输出通常还需要把 projected signal 与非投影 axis metadata 配对。
例如 absorption-like projection 后，常见 bundle 是：

```text
projected_signal = projected absorption
axes["energy_eV"] = energy_eV
axes["omega_fs_inv"] = omega_fs_inv
```

或 polarization projection 后：

```text
projected_signal = projected polarization_C_per_m2
axes["time_fs"] = time_fs
```

`ProjectedReadoutBundle` 用于保存这种通用配对结果：

```text
signal_name
signal_quantity
projected_signal
axes
phase_result_summary
metadata
```

`AxisMetadataSpec` 描述 axis metadata 从哪里来：

- `source="first_case"`：只从第一个 phase case 提取 axis；
- `source="validate_all_cases"`：从所有 phase cases 提取 axis，并用
  `np.allclose` 验证一致。

`validate_all_cases` 适合 `energy_eV`、`omega_fs_inv` 这类每个 phase case
都应相同的频率轴。若不同 phase case 的 axis shape 或数值不一致，会直接
报错，避免把不匹配的频率轴和 projected signal 绑定在一起。

axis metadata 不做 Fourier projection。`target_phase_vector` 只作用于
`PhaseProjectionSpec.quantity` 指定的 signal quantity。recipe 层决定哪些
axes 应与 projected signal 配对；TA / 2DES 的物理约定仍由各自 recipe
固定。

当前 bundle helper 仍不实现：

- TA subtraction；
- delay scan；
- probe-only / pump-probe orchestration；
- 2DES `t1/t2/t3` grid；
- 2D Fourier transform；
- explicit LO/readout phase recipe；
- phase-case checkpoint path builder。

## Milestone 5.1：TA recipe v2 minimal single-delay scaffold

当前 `ta_recipe_v2.py` 提供一个最小 TA recipe v2 外壳，用 Layer 3 的
TA 语义组织已有 generic 层：

```text
pump PulseSpec
probe PulseSpec
single delay centers
-> pump-probe SingleRunPlan
-> probe-only SingleRunPlan
-> probe-channel absorption-like readout bundle
```

最小 TA 物理语义：

- physical pulses: pump pulse 与 probe pulse；
- delay 定义：`delay_fs = probe_center_fs - pump_center_fs`；
- 正 delay 表示 pump before probe；
- readout / detection：probe-channel absorption-like readout；
- `ReadoutSpec(mode="absorption", readout_field_name=probe.name)`；
- reference cases：pump-probe response 与 probe-only response；
- 后续 TA signal 约定：

```text
S_TA(omega, delay) = S_pump_probe(omega, delay) - S_probe_only(omega)
```

当前 readout 不是第三个激发脉冲。probe 既是 physical probe pulse，也是
readout field reference。最小 TA recipe 不引入 independent LO phase tag；
如果后续显式建模 heterodyne observable，再在 recipe 层定义 LO / readout
phase 约定。

`TASingleDelayPlan` 当前只生成两条 `SingleRunPlan`：

- `make_pump_probe_plan()`：sequence 包含 pump 和 probe；
- `make_probe_only_plan()`：sequence 只包含 probe。

`extract_ta_absorption_bundle(...)` 从 `SingleRunResult.readout.spectrum`
提取：

- `absorption`；
- `energy_eV`；
- 可选 `omega_fs_inv`。

它只打包单条 response，不做 subtraction、不做 phase projection。

`TASingleDelayPlan` 与 `extract_ta_absorption_bundle(...)` 本身不做
subtraction；subtraction 由 Milestone 5.2 的 `compute_ta_contrast(...)`
显式完成。

当前单 delay scaffold 本身不实现：

- phase-cycling TA；
- TAResultIO v2；
- old demo migration。

## Milestone 5.2：TA subtraction 与 energy-axis alignment

当前 `ta_recipe_v2.py` 已支持单 delay TA contrast：

```text
TASingleDelayPlan
-> execute_pair()
-> pump-probe SingleRunPlan.execute()
-> probe-only SingleRunPlan.execute()
-> extract_ta_absorption_bundle()
-> compute_ta_contrast()
-> TAContrastResult
```

subtraction convention 固定为：

```text
S_TA(omega, delay) = S_pump_probe(omega, delay) - S_probe_only(omega)
```

当前不做 sign flip，不做 OD / delta OD 转换，不做 relative difference，不做
normalization variants。

`TASubtractionSpec` 只控制：

- output signal name；
- energy / omega axis validation tolerance；
- 是否验证 `omega_fs_inv` axis。

axis policy：

- `absorption` shape 必须一致；
- `energy_eV` shape 必须一致，且 `np.allclose`；
- 如果 `validate_omega_axis=True`，两边 `omega_fs_inv` 要么都不存在，要么
  shape 一致且 `np.allclose`；
- 如果 `validate_omega_axis=False`，不验证 omega axis，并优先沿用
  pump-probe bundle 的 `omega_fs_inv`；
- 当前不插值、不重采样、不平滑、不静默截断。

`TAContrastResult` 保存单 delay 的：

- `delta_absorption`；
- `energy_eV`；
- 可选 `omega_fs_inv`；
- subtraction spec summary；
- source case names；
- axis validation metadata。

当前 Milestone 5.2 的单 delay subtraction 本身不实现：

- phase-cycling TA；
- TAResultIO v2；
- old demo migration。

## Milestone 5.3：TA delay scan scaffold

当前 `ta_recipe_v2.py` 已支持最小 TA delay scan scaffold：

```text
TADelayScanPlan
-> make_single_delay_plans()
-> 每个 delay 执行 TASingleDelayPlan.execute_pair()
-> TASingleDelayPairResult.compute_contrast()
-> build_ta_delay_scan_map()
-> TADelayScanResult
```

当前支持范围：

- 单 delay pump-probe / probe-only 编排；
- 单 delay `S_TA = S_pump_probe - S_probe_only` contrast；
- 多 delay scaffold；
- 将多个单 delay contrast 堆叠为 delay × energy map；
- fake executor 测试入口，用于不触发 solver 的 scan 级验证。

delay / axis policy：

- delay 轴严格保留输入顺序；
- 不按 delay 数值排序；
- 每个 delay 的 `energy_eV` shape 必须一致；
- 每个 delay 的 `energy_eV` 数值必须 `np.allclose`；
- 默认要求每个 delay 的 `omega_fs_inv` 要么都存在且一致，要么都不存在；
- 如果 `validate_omega_axis=False`，不验证 omega 轴，并沿用第一个
  contrast 的 `omega_fs_inv`；
- 当前不做 interpolation、resampling、smoothing、截断或自动修轴。

`TADelayScanMap` 保存：

- `delays_fs`；
- `energy_eV`；
- `delta_absorption`，shape 为 `n_delay × n_energy`；
- 可选 `omega_fs_inv`；
- source contrast case names；
- axis validation metadata。

`TADelayScanPlan` 仍拒绝 `checkpoint.enabled=True`，避免多个 delay case
默认复用同一个 checkpoint 路径导致覆盖或误读。当前 scaffold 不保存大输出。

当前 Milestone 5.3 仍不实现：

- phase-cycling TA；
- TAResultIO v2；
- old demo migration；
- plotting；
- npz / csv export；
- interpolation / resampling；
- OD / delta OD / relative difference / normalization variants。

## Milestone 5.4：optional pump-probe phase-cycling scaffold

当前 `ta_recipe_v2.py` 已支持可选 pump-probe phase-cycling scaffold：

```text
TASingleDelayPlan
-> make_pump_probe_plan()
-> build_ta_pump_probe_phase_cycling_plan()
-> PhaseCyclingPlan.execute(...)
-> build_ta_phase_cycled_pump_probe_bundle()
-> ProjectedReadoutBundle
```

当前支持范围：

- 使用 `TAPhaseCyclingSpec` 显式描述 phase grid、target phase vector、
  projection quantity、projection sign 和 axis metadata；
- 只对 pump-probe readout quantity 做 Fourier projection；
- target phase vector 必须由用户或上层 recipe 显式传入；
- 支持把 `PhaseCyclingResult` 打包为 pump-probe projected readout bundle；
- 支持 fake executor 测试，不要求默认运行真实 multi-phase solver。

The optional TA phase-cycling scaffold projects the pump-probe readout quantity
using an explicitly supplied `target_phase_vector`. It does not define a
universal TA phase convention. A population-like example may use pump net phase
0 and probe +1, but this remains recipe-level convention, not a bottom-level
rule.

语义边界：

- no universal TA phase target is hard-coded；
- `target_phase_vector` 属于 recipe / cycler 层，不属于 `run_case`；
- `target_phase_vector` 不属于 `DynamicsResult`；
- readout 不是第三个激发脉冲；
- 在当前 minimal TA recipe 中，probe 既是 physical probe pulse，也是
  readout field reference；
- 本轮不引入 independent LO / readout phase tag；
- 如果后续显式建模 heterodyne observable，再由 recipe 层定义 LO /
  readout phase convention。

当前 Milestone 5.4 仍不实现：

- phase-cycled pump-probe minus probe-only subtraction；
- probe-only phase projection；
- delay-scan phase cycling；
- TAResultIO v2；
- old demo migration；
- explicit LO/readout phase convention；
- OD / delta OD / relative difference / normalization variants；
- interpolation / resampling / smoothing / sign flip。

Phase-cycled TA contrast 保留给后续 recipe-level decision；需要先固定 TA
phase convention，再决定 probe-only reference 是否参与 projection，以及
如何与 pump-probe projected signal 相减。

## Migration plan

### Milestone 1：已完成

- 架构文档；
- generic multi-pulse / pulse-sequence single-run scaffold；
- tests / smoke；
- 标注 `qudpy_sjh/experiments/ta/` 是未接入的 TA recipe v1 prototype；
- 不改变现有 TA 行为；
- phase cycling 只做文档设计，不实现完整 runner。

### Milestone 2：已完成

- 为 `CarrierEnvelopeField` 实现 `with_phase` / `phase_shifted` API；
- `FieldPhyRoot` 保持通用抽象，不把框架写死为只支持
  `CarrierEnvelopeField`；
- `PulseSpec.shifted(...)` 显式接入 `CarrierEnvelopeField` 作为当前首个
  phase-aware backend；
- 支持任意 pulse sequence 生成带真实 phase override 的 `FieldPhySeries`；
- 没有改变 solver。

### Milestone 3：已完成低风险 generic single-run plan 与 readout

- 已实现 generic `SingleRunPlan / ReadoutSpec`；
- 已标准化一次 concrete field propagation + 可选 readout；
- 已支持 no readout、polarization readout、absorption-like spectral readout；
- 已保留 `readout_field_name` 接口，供后续 TA / 2DES recipe 指定 total
  field 或 named subfield；
- 未实现 phase grid、Fourier projection、TA subtraction 或 2DES readout。

### Milestone 4：已完成低风险 generic PhaseCycler runner

- 已实现 `PhaseGrid / PhaseCyclingPlan / PhaseCyclingResult`；
- 已实现 `normalize_target_phase_vector`、`phase_projection_weight` 和
  `fourier_project_phase_cases`；
- 已支持从 `SingleRunResult.readout` 中提取指定 array 并投影；
- 已支持 fake executor 测试，不要求默认运行多次 solver；
- checkpoint enabled 时有明确 guard；
- 未实现 TA subtraction、TA recipe v2、2DES recipe 或 phase-case
  checkpoint path builder。

### Milestone 4.5：已完成低风险 projected-result bundle

- 已实现 `AxisMetadataSpec`；
- 已实现 `ProjectedReadoutBundle`；
- 已实现 `build_projected_readout_bundle`；
- 已支持从 first case 提取 axis metadata；
- 已支持从 all cases 提取并验证 axis metadata；
- 已支持 `readout.time_fs` quantity；
- axis metadata 不做 Fourier projection；
- 未实现 TA subtraction、delay scan 或 2DES FFT。

### Milestone 5.1：已完成低风险 TA recipe v2 minimal scaffold

- 已实现 `TADelayCenters`；
- 已实现 `TAReadoutBundle`；
- 已实现 `TASingleDelayPlan`；
- 已实现 `TASingleDelayPairResult`；
- 已实现 `extract_ta_absorption_bundle`；
- 已支持生成单 delay 的 pump-probe `SingleRunPlan`；
- 已支持生成单 delay 的 probe-only `SingleRunPlan`；
- 默认 readout 是 probe-channel absorption-like readout；
- 此阶段未实现 TA subtraction、delay scan、phase-cycling TA 或 TAResultIO v2。

### Milestone 5.2：已完成低风险 TA subtraction 与 energy-axis alignment

- 已实现 `TASubtractionSpec`；
- 已实现 `TAContrastResult`；
- 已实现 `validate_ta_readout_bundle_axes`；
- 已实现 `compute_ta_contrast`；
- 已支持 `TASingleDelayPairResult.compute_contrast()`；
- 已固定 `S_TA = S_pump_probe - S_probe_only`；
- 已验证 pump-probe / probe-only 的 energy axis 对齐；
- 已新增单 delay TA execute smoke；
- 此阶段未实现 delay scan、phase-cycling TA、TAResultIO v2、OD/relative
  difference、interpolation 或 resampling。

### Milestone 5.3：已完成低风险 TA delay scan scaffold

- 已实现 `TADelayScanMap`；
- 已实现 `validate_ta_contrast_axes_for_scan`；
- 已实现 `build_ta_delay_scan_map`；
- 已实现 `TADelayScanPlan`；
- 已实现 `TADelayScanResult`；
- 已支持对多个 delay 生成 `TASingleDelayPlan`；
- 已支持收集 delay-energy map；
- delay 输入顺序保留，不排序；
- 所有 delay 的 energy axis 必须对齐；
- checkpoint enabled 时有明确 guard；
- 已支持 fake executor 测试，不要求默认运行真实 multi-delay solver scan；
- 不改变 single-run / solver 行为；
- 未实现 phase-cycling TA、TAResultIO v2、绘图、npz/csv export、旧 demo
  迁移、OD/ΔOD variants、interpolation 或 resampling。

### Milestone 5.4：已完成低风险 optional pump-probe phase-cycling scaffold

- 已实现 `TAPhaseCyclingSpec`；
- 已实现 `TAPhaseCycledPumpProbeResult`；
- 已实现 `build_ta_pump_probe_phase_cycling_plan`；
- 已实现 `build_ta_phase_cycled_pump_probe_bundle`；
- 已支持从 `TASingleDelayPlan` 构造 pump-probe `PhaseCyclingPlan`；
- `target_phase_vector` 必须显式传入；
- 不硬编码 TA phase target vector；
- 不把 phase cycling 设为默认流程；
- 当前只 project pump-probe readout；
- 未实现 phase-cycled TA subtraction、probe-only phase projection、
  delay-scan phase cycling、TAResultIO v2 或 old demo migration。

### Milestone 5.5：decide TA phase-cycled contrast convention

- 固定或继续推迟 TA phase-cycled contrast convention；
- 决定 probe-only reference 是否参与 phase projection；
- 明确 pump-probe projected signal 与 reference 的 subtraction 规则。

### Milestone 5.6：TAResultIO v2

- 新增 TA recipe v2 的保存格式；
- 支持 single-delay contrast、delay scan map 与 optional projected bundle；
- 不复用 legacy v1 IO 的含义作为隐式默认。

### Milestone 5.7：old demo migration

- 复现旧 demo 的关键输出；
- 明确 `phase_stack`、`phase_avg`、`phase_rms`、selected-delay spectra、
  selected-energy kinetics 的 v2 对应关系。

### Milestone 6：新增 2DES recipe

- pulse1/pulse2/pulse3/readout；
- rephasing / non-rephasing / double-quantum target vectors；
- `t1/t2/t3` grid；
- 2D Fourier transform 和 2D spectra output。
