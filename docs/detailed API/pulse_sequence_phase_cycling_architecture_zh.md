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
- 可作为后续 TA recipe v2 的参考和回归基准；
- 不是 generic pulse-sequence simulation framework；
- 当前不默认执行 phase cycling；
- 未来 TA recipe v2 可以逐步调用 generic pulse-sequence / phase-cycling
  基础层。

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

- TA subtraction；
- full delay scan；
- phase-cycling TA；
- TAResultIO v2；
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
```

这些 API 不依赖 TA / 2DES / solver / matplotlib，只负责构造一次 concrete
`FieldPhySeries`。

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

它只打包单条 response，不做 subtraction、不做 phase projection、不假设
delay scan。

当前 TA recipe v2 minimal scaffold 不实现：

- TA subtraction；
- delay scan；
- phase-cycling TA；
- TAResultIO v2；
- old demo migration。

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
- 未实现 TA subtraction、delay scan、phase-cycling TA 或 TAResultIO v2。

### Milestone 5.2：TA subtraction 与 energy-axis alignment

- 定义 `TASubtractionSpec`；
- 定义 `TAContrastResult`；
- 实现 `S_TA = S_pump_probe - S_probe_only`；
- 验证 pump-probe / probe-only 的 energy axis 对齐。

### Milestone 5.3：TA delay scan

- 对多个 delay 生成 `TASingleDelayPlan`；
- 收集 delay-energy map；
- 不改变 single-run / solver 行为。

### Milestone 5.4：optional phase-cycling TA

- 将 generic `PhaseCyclingPlan` 接入 TA recipe；
- 固定 TA recipe 的 target phase vector 约定；
- 不把 phase cycling 设为默认流程。

### Milestone 5.5：TAResultIO v2 与 demo migration

- 新增 `TAPhaseCyclingRecipe / TAPhaseCyclingResult / TAPhaseCyclingIO`；
- 复现旧 demo 的 `phase_stack`、`phase_avg`、`phase_rms`、
  selected-delay spectra、selected-energy kinetics。

### Milestone 6：新增 2DES recipe

- pulse1/pulse2/pulse3/readout；
- rephasing / non-rephasing / double-quantum target vectors；
- `t1/t2/t3` grid；
- 2D Fourier transform 和 2D spectra output。
