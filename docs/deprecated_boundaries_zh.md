# Deprecated Boundaries

本文档记录当前 `QuDPy-clean` 不进入主线的旧架构和旧 API 边界。目的不是删除历史经验，而是防止旧仓库中的实验性设计被误写进 README、架构文档、field 文档、solver 文档或 IO 文档。

## 当前主线

当前主线是：

```text
lab-frame physical field
-> NLevelPhysicalParams(..., field=field)
-> ParaNormalizer
-> full-window lab_exact solver
-> ordinary DynamicsResult
-> spectroscopy / IO / workflow
```

field 主线是：

```text
FieldPhyRoot
FieldPhyCustomed
TimeShiftedField
FieldPhySeries
carrier_envelope.CarrierEnvelopeField
```

carrier-envelope 主线是：

```text
CarrierSpec
EnvelopeSpec
GaussianEnvelopeSpec
SechEnvelopeSpec
ConstantEnvelopeSpec
CarrierEnvelopeField
make_gaussian_carrier_envelope_field
make_pump_probe_field_series
```

以下内容不属于当前主线。

## 不在主线的旧架构清单

当前不应宣传、恢复或默认支持：

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

这些名称如果出现在旧文档、旧脚本说明或旧注释中，应被标记为 deprecated、archived 或 historical note，不应写成当前能力。

## 不再作为推荐 field API 的旧接口

以下旧 field 文件、类或 helper 不应写成当前 public API 主线：

```text
specific/basic_fields.py
specific/ta_fields.py
specific/twodes_fields.py
CarrierFieldPhysical
GaussianCarrierFieldPhysical
TAField
TwoDESField
make_default_carrier_field
make_default_gaussian_carrier_field
make_ta_gaussian_field
make_pump_probe_field_from_templates
make_ta_field_from_templates
make_twodes_gaussian_field
```

推荐替代表述：

```text
Use CarrierEnvelopeField or make_gaussian_carrier_envelope_field(...) for finite optical pulses.
Use FieldPhySeries(..., sub_field_names=(...)) to express pump / probe / LO roles.
Keep delay scan, phase cycling and TA map construction in workflow scripts.
```

## 为什么不应放进 README

README 面向第一次打开仓库的人，应只描述当前稳定主线。

如果 README 宣传 piecewise / dark / complex TA scaffold，会造成三个问题：

1. 用户会误以为当前 solver 支持 piecewise propagation；
2. 用户会误以为 IO 可以保存 piecewise result series；
3. 后续开发者可能绕过当前 `DynamicsResult` 主线，重新引入旧仓库的复杂状态对象。

当前 README 应只强调：

```text
lab-frame physical field
carrier-envelope finite pulse
FieldPhySeries pump/probe composition
ParaNormalizer
full-window lab_exact solver
ordinary DynamicsResult
lab_frame_absorption_response
ordinary IO / metadata
workflow-level delay scan and phase cycling
```

## 为什么不应放进 core 主线

当前 core 的职责是：

```text
parameters
normalization
Hamiltonian / c_ops construction
full-window solver
single-trajectory result container
```

piecewise / dark propagation 会引入额外概念：

```text
active window
dark window
piece sequence
piece materialization
series-level result object
partial trajectory stitching
```

这些概念会改变 solver、result、IO 和 metadata 的边界。若没有新的设计文档，不应直接混入当前 core。

## 为什么不应放进 IO 主线

当前 IO 针对普通 `DynamicsResult`：

```text
density.npz
components.csv
populations.csv
meta.json
debug_meta.json
preview.png
full.png
results.csv
```

一个 case 对应一条普通 trajectory。

如果引入 `PieceDynamicsResultSeries` 或 piecewise series 输出，IO 需要新增独立 schema，例如：

```text
piece index
piece start / end
piece type
stitching rule
materialization rule
full-window reconstruction rule
```

当前代码未确认这些 schema。因此文档不应写：

```text
save_result_case supports piecewise series
```

也不应写：

```text
materialize_full before saving
```

## 为什么不应恢复 materialize_full

`materialize_full` 暗示存在一个 piecewise series，需要被重新拼接成 full-window result。当前主线已经直接使用 full-window `lab_exact` solver，标准输出就是普通 `DynamicsResult`。

因此当前文档应避免：

```text
materialize_full
materialized result
dark materialization
piece materialization
```

如未来确实需要 long-window acceleration，应重新设计，而不是恢复旧名称。

## 为什么不应恢复 run_case(piecewise=...)

当前 `run_case(...)` 的边界应保持简单：

```text
run_case(NLevelPhysicalParams, normalizer=None, rho0=None, ...)
-> DynamicsResult
```

不应新增或宣传：

```python
run_case(piecewise=True)
run_case(piecewise=...)
run_case(active_dark=...)
```

如果未来需要 piecewise workflow，应优先作为独立上层 workflow 或独立 solver entry point，而不是污染当前 `run_case` 主入口。

## 为什么不应恢复 complex transient_absorption scaffold

当前 TA / pump-probe 主线应分层：

```text
field layer:
  make_gaussian_carrier_envelope_field(...)
  FieldPhySeries(fields=(pump, probe), sub_field_names=("pump", "probe"))

solver layer:
  ordinary full-window lab_exact run_case

spectroscopy layer:
  polarization_C_per_m2
  lab_frame_absorption_response

workflow layer:
  delay scan
  probe-only reference
  phase cycling
  differential spectrum
  figure layout
```

复杂 transient absorption scaffold 若包含 propagation、piecewise active/dark、dark materialization、special result series 或 long-window benchmark，就不属于当前主线。

当前可以保留的 TA 能力是：

```text
pump-probe physical field composition
FieldPhySeries["pump"] / FieldPhySeries["probe"]
delay convention in workflow
ordinary full-window simulation
post-processing absorption response
```

不应宣传为：

```text
完整实验 TA pipeline
传播模型
样品厚度模型
detector response
transmitted intensity model
piecewise TA acceleration
```

## 后续开发者提醒

在修改 README 或 docs 时，先检查是否出现以下词：

```text
PieceDynamicsResultSeries
dark_propagation
piecewise_propagation
ActiveWindow
PropagationPiece
execute_piece_sequence
materialize_full
run_case(piecewise
save_result_case(piecewise
long-window piecewise benchmark
transient_absorption scaffold
CarrierFieldPhysical
GaussianCarrierFieldPhysical
TAField
TwoDESField
make_default_gaussian_carrier_field
make_ta_gaussian_field
make_pump_probe_field_from_templates
make_twodes_gaussian_field
```

如果出现，应判断它是否只是 deprecated boundary。如果不是，应移出主线文档。

在修改 core 或 IO 时，不要为了兼容旧文档而恢复旧架构。正确顺序应是：

```text
1. 先确认当前代码是否支持；
2. 不支持则不要写成已支持；
3. 如果确实需要支持，先写设计文档；
4. 再实现；
5. 最后更新 README 和 current architecture。
```

## 推荐替代表述

旧表述：

```text
QuDPy supports piecewise active/dark propagation and materialization.
```

推荐改为：

```text
QuDPy-clean currently uses full-window lab-frame propagation and returns one ordinary DynamicsResult per run.
```

旧表述：

```text
save_result_case can save PieceDynamicsResultSeries.
```

推荐改为：

```text
save_result_case saves ordinary full-window DynamicsResult outputs, including density.npz, components.csv, metadata, and optional figures.
```

旧表述：

```text
TA scaffold handles long-window active/dark propagation.
```

推荐改为：

```text
TA-related code currently provides carrier-envelope pump-probe field composition and spectroscopy post-processing utilities. Delay scan, phase cycling and response-map construction belong to the workflow layer.
```
