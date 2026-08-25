# Phase Cycling / Readout / Recipe 目标架构

> **Source of truth**
>
> 本文档是 QuDPy 目标 phase-cycling、readout 与 recipe 架构的权威说明。
> 后续实现、测试、example 和其他文档应以本文档为准。若当前源码与本文档
> 不一致，应把差异视为待迁移项，而不是修改本文档以迁就 legacy 行为。

本文档冻结术语、数学 convention、层级职责、目标数据流和已知迁移边界。
它不表示所有目标能力已经实现。当前实现状态与目标状态必须明确区分。

## Current implementation 与 Target architecture

当前源码已经具备可靠的 physical field、pulse sequence 和基础 phase-grid
表达能力，但尚未完成目标 workflow。特别是：

- 当前 Fourier projection 的 public/default behavior 仍包含 legacy
  `sign=-1`；目标 convention 是固定的 `+i` projection；
- 当前 `ReadoutSpec` 和 readout execution 嵌在 `SingleRunPlan` 中；
- 当前 heavy `PhaseCyclingPlan` 会生成 case、执行 solver/readout 并做投影；
- 当前 TA phase-cycling scaffold 在 recipe-specific postprocess 之前投影
  pump-probe readout；
- 当前一个 `PhaseCyclingPlan` 只接受一个 target，多个 target 会导致昂贵
  计算被重复编排；
- 当前存在 `ProjectedReadoutBundle`、`TAPhaseCycledPumpProbeResult` 等目标中
  将废弃的 wrapper；
- 当前部分 API 使用 `absorption` 表示
  `omega * Im[P(omega) / E(omega)]`，它不是 detector intensity。

以上均是已知 mismatch，不是本文档遗漏。Milestone 0 只冻结语言和目标边界，
不改变 runtime behavior。

## 核心原则

1. `System` 是 matter，`Field` 是 perturbation，`SolverParams` 是数值求解配置。
2. physical pulse count 不等于 phase-dimension count。
3. phase semantics 位于 pulse-sequence / recipe 层，不放进 raw physical Field。
4. Recipe 描述对一个给定 System 做什么实验；System scan 由上层脚本管理。
5. readout physics 与 solver execution 分离。
6. Recipe 先把多个 `ReadoutResult` 组合为实验 observable `S(...)`，然后才做
   generic phase projection。
7. generic phase-projection 层只处理预先计算好的 array-valued `S(...)`，不执行
   solver、readout 或 recipe postprocess。
8. 一份 `S(phi)` 必须可以投影多个 target phase-order vectors，而不重复昂贵计算。
9. 数据容器保持最简：NumPy ndarray、axis names、axis values 和必要 metadata。
10. 新架构优先于 backward compatibility；旧 API 在新 workflow 完成验证后再弃用。

## Terminology Table

| Term | Exact meaning | Owning layer | Current code representation | Target status |
| --- | --- | --- | --- | --- |
| System | Hamiltonian、basis、dipole、initial state、relaxation、dephasing 等 matter physics | System | `NLevelSystem`；底层仍会转换为 `NLevelPhysicalParams` | 保留概念；完善 adapter 属于独立任务 |
| SystemScanAxis | 在不同 System 间扫描的上层坐标，例如 PB、EIS、EID、temperature、model member | Upper-level orchestration | 当前主要由 scripts/examples 的循环表达；无必需 runtime class | 不进入 Recipe；不要求新增 class |
| Field | 进入 Hamiltonian 的 physical perturbation field | Field | `FieldPhyRoot` 及实现类 | 保留 |
| FieldPhyRoot | physical field 的抽象接口，使用物理时间与场强单位 | Field | `qudpy_sjh.utils.fields.lab_fields.FieldPhyRoot` | 保留 |
| CarrierEnvelopeField | 单 carrier、单 envelope 的 finite physical optical field | Field | `CarrierEnvelopeField` | 保留 |
| FieldPhySeries | 多个 physical fields 的线性和及 named subfields | Field | `FieldPhySeries` | 保留 |
| PulseSpec | physical field template 加 pulse timing/base phase/phase-tag metadata | PulseSequence | `PulseSpec` | 保留 |
| FieldGroupSpec | 多个 coherent physical pulses 的组合，可共享 group phase tag | PulseSequence | `FieldGroupSpec` | 保留当前名称 |
| PulseSequenceSpec | 一次 acquisition condition 的 pulse/group composition，可构造 concrete physical field | PulseSequence | `PulseSequenceSpec` | 保留 |
| SingleRunFieldPlan | 当前一次 concrete centers/phases 的 field-construction record | PulseSequence / execution boundary | `SingleRunFieldPlan` | 暂缓 rename/removal，后续重新评估 |
| SingleRunPlan | 对一个 concrete physical field 执行一次 dynamics simulation 的计划 | Execution | 当前还包含 `ReadoutSpec` 并在 `execute()` 中做 readout | 目标只负责 solver/dynamics execution |
| SimRes | 一次昂贵 dynamics simulation 的 canonical result | Execution | 当前 canonical class 是 `DynamicsResult` | 概念名可用 `SimRes`；当前实现继续用 `DynamicsResult` |
| SolverParams | time grid、tolerance、integrator/backend、numerical options、checkpoint settings | Execution | 当前高层 `NLevelPhysicalParams` 混合 System/Field/time grid；当前 `SolverParams` 是归一化内部摘要 | 目标概念已冻结；runtime 重构不在 M0 |
| Recipe | 对一个给定 System 定义实验 cases、coordinates、readout、postprocess 和 targets 的轻量 script/module | Recipe | TA 当前由多种 `TA*Plan/Spec` 类和 examples 表达 | 目标保持轻量，不要求 generic Recipe base class |
| Condition | 同一个 System 下不同 acquisition configuration，例如 pump on/off、probe on/off、chopper 或 LO state | Recipe | 当前 TA 使用 pump-probe/probe-only cases | 不代表不同 material/System |
| recipe coordinate | Recipe 自己定义的非 phase 实验坐标，例如 TA 的 `T` 或 2DES 的 `tau, T, t` | Recipe | 当前 TA delay fields/classes | 不创建 generic DelayAxis/ConditionAxis |
| phase dimension | 一个真正被 phase cycling 的、独立可控的 physical/mathematical phase degree of freedom | Recipe / phase domain | 由 pulse/group phase semantics 与 `PhaseGrid` 共同表达 | 正式术语 |
| phase_tag | 在代码中标识一个 phase dimension 的字符串 identifier | PulseSequence / phase domain | `PulseSpec.phase_tag`、`FieldGroupSpec.phase_tag`、`PhaseGrid.tags` | 正式代码术语 |
| PhaseGrid | phase tags 到 phase samples 的映射及其 Cartesian-product domain | Generic phase projection | `PhaseGrid` | 保留；扩展 uniform-grid convenience |
| phase order | observable 对某个 cycled phase dimension 的整数 Fourier/pathway order | Physics / Recipe | 当前 target coefficient | 正式物理术语 |
| phase-order vector | 所有 phase dimensions 的整数 order vector `m=(m_1,...,m_D)` | Physics / Recipe | `dict[str, int]` | 正式物理术语 |
| target_phase_vector | phase_tag 到 physical phase order 的代码映射 | Recipe / generic projection boundary | `target_phase_vector` | 保留名称；未来不暴露 configurable sign semantics |
| ReadoutField | coherent detection 使用的 field object/reference；可来自 interaction field，也可为 external LO | Readout | 当前 `readout_field_name` 只能选择 interaction field/其 subfield | 目标支持 named reference 或 external `FieldPhyRoot` |
| ReadoutPlan | 把 polarization 转换为 detector/readout observable 的 callable plan | Readout | 尚不存在；`ReadoutSpec` 只是配置并由外部函数执行 | 后续 behavioral redesign，不是简单 rename |
| ReadoutResult | 一次 readout 执行得到的 detector/readout data 与 axes/metadata | Readout | 当前近似对应 `SingleRunReadoutResult` | 后续独立于 `SingleRunResult` |
| Recipe.postprocess | 匹配、复用、broadcast、组合多个 `ReadoutResult`，形成每个 phase case 的 `S(...)` | Recipe | 当前 TA subtraction 分散在 TA helper/plan 中 | 目标为普通 Recipe 方法/函数，不建 `PostprocessPlan` |
| S(...) | recipe-specific postprocess 后、phase projection 前的 array-valued experimental observable | Recipe -> projection interface | 当前没有统一接口；可能直接投影 single-run readout | 正式接口；不限定为 polarization/intensity/absorption |
| phase cycling | 实验上改变一个或多个独立 phase variables 并 acquisition 的过程/概念 | Experiment / Recipe | 当前也被用于 heavy runner 名称 | 保留实验术语 |
| phase projection | 对已经计算好的 `S(phi)` 做离散 Fourier projection 得到 `S_m` | Generic phase projection | `fourier_project_phase_cases` 及 heavy `PhaseCyclingPlan` 的一部分 | 目标为 plain mathematical operation |
| payload axis | `S` 中既非 phase dimension、也非 system-scan axis 的普通数据维度，例如 omega 或 detection time | Recipe data / analysis | 当前由 readout arrays 与 `AxisMetadataSpec` 松散表达 | 用 `axis_names`/`axis_values` 明确绑定 |
| axis_names | 与 ndarray 每一维按位置一一对应的名称 tuple | Recipe data / generic projection | 当前没有统一 contract | 目标最小 contract |
| axis_values | axis name 到一维 coordinate values 的 mapping | Recipe data / generic projection | 当前 bundle 中有松散 `axes` dict | 目标最小 contract |
| UFANSYS / user analysis | FFT、plotting、alignment、selection、slicing 与后续 spectroscopy analysis | Analysis | 仓库外/用户脚本 | 不进入 Recipe 或 phase projection core |

## 容易混淆的术语

### Phase dimension 与 phase_tag

`phase dimension` 是物理和数学自由度；`phase_tag` 是标识该自由度的程序字符串。
例如 `phi_pump` 是 phase dimension，代码可以用 `phase_tag="pump"` 表示它。

当前 `PulseSpec` / `FieldGroupSpec` 还包含 `independent_phase`。它可能和
`phase_tag`、`PhaseGrid` 形成第二套 independence source of truth。M0 只记录
该风险；后续应决定是否清理，当前 runtime 不变。

### Physical pulse 与 phase dimension

```text
pulse1 phase = phi_pump
pulse2 phase = phi_pump
probe  phase = phi_probe
```

这是三个 physical pulses、两个 phase dimensions。多个 pulses 可以通过相同
phase tag 或 coherent `FieldGroupSpec` 共享一个被 cycle 的 phase。固定相位 pulse
不需要出现在 `PhaseGrid` 中；数学上它可视为 `N_j=1`，程序无需暴露该维度。

### Phase cycling 与 phase projection

- phase cycling：实验概念，包括设定 phase、进行 acquisition 和得到 phase cases；
- phase projection：对预先计算好的 `S(phi)` 进行数学 Fourier projection。

generic phase-projection 层不执行实验 case，也不执行 solver。

### Recipe coordinate、phase dimension 与 SystemScanAxis

- `T`、`tau`、`t` 是 recipe coordinates；
- `phi_pump`、`phi_probe` 是 phase dimensions；
- PB、EIS、EID、temperature、model member 是 SystemScanAxis coordinates。

Recipe 决定前两类的实验含义。System scan 完全由上层脚本管理。

### ReadoutResult 与 S

`ReadoutResult` 是一个 acquisition case 经 detector/readout physics 得到的结果。
`S` 是 Recipe 将一个或多个 `ReadoutResult` reuse、broadcast、combine 后形成的
实验 observable。只有 `S` 进入 generic phase projection。

## Adopted Fourier Convention

一条 pathway 的 phase 与空间因子定义为：

```text
exp[-i * sum_j(s_j * phi_laser,j)]
*
exp[+i * sum_j(s_j * k_j dot r)]
```

共享一个 pump phase 的多次 pump interaction 满足：

```text
m_pu = sum_j(s_pu,j)
```

一般 phase-order vector 为 `m=(m_1,...,m_D)`。采用的 projection convention 是：

```text
S_m = 1/product_j(N_j)
      * sum_over_phase_cases[
          S(phi_1,...,phi_D)
          * exp(+i * sum_j(m_j * phi_j))
        ]
```

uniform grid 为：

```text
phi_j,n = 2*pi*n/N_j
```

所以：

```text
S(phi) proportional to exp(-i*k*phi)
    -> target phase order m = k
```

`target_phase_vector` 直接表示 physical phase-order vector `m`。未来 public API
不应再把 configurable `sign` 当作 phase-order semantics 的组成部分。

> **Current mismatch:** 当前 implementation 的 projection helpers/specs 默认仍为
> `sign=-1`。这将在 Milestone 1 迁移；在迁移完成前，不能根据本文档假定 runtime
> 已使用 `+i` convention。

## PhaseGrid 的数学边界

目标 `PhaseGrid` 支持：

- 任意 phase-dimension 数量 `D`；
- 每个维度不同的 `N_j`；
- 一个 `N` 应用于全部 dimensions 的 convenience input；
- 标准 uniform grids；
- 用户显式给定的 finite phase values。

“API 接受任意 phase values”不等于“equal-weight DFT 对任意 nonuniform samples
都是正确的 Fourier inversion”。当前及目标中的简单平均 projection 对标准完整
uniform grid 有明确离散正交性；对任意 nonuniform/不完整采样，通常需要 least
squares、quadrature weights 或其他 inversion 方法。未来 API 必须标明所采用的
数学假设，不能把输入 validation 误写成一般 nonuniform reconstruction 保证。

## Target Workflow

```mermaid
flowchart TD
    A[Upper-level script] --> B[System plus Recipe plus SolverParams]
    A --> A1[Optional SystemScanAxis]
    A1 --> B
    B --> C[Recipe-defined simulations]
    C --> D[PulseSequence]
    D --> E[SingleRunPlan]
    E --> F[Solver]
    F --> G[SimRes / DynamicsResult]
    G --> H[Polarization]
    H --> I[ReadoutPlan]
    I --> J[ReadoutResult]
    J --> K[Recipe postprocess]
    K --> L[S with recipe, phase, and payload axes]
    L --> M[Generic phase projection]
    M --> N[Projected phase-order data]
    N --> O[UFANSYS or user analysis]
```

Recipe 可以直接由 script/module 定义。不要为了图中的逻辑节点创建
`ConditionPlan`、`RecipeCase`、`DelayAxis` 或 `PostprocessPlan` runtime classes。

## Layer Responsibility Table

| Layer | Owns | Does not own |
| --- | --- | --- |
| System | matter Hamiltonian、dipoles、initial state、relaxation、dephasing、system makers/adapters | Recipe、phase cycling、SystemScan orchestration |
| Field | carrier、envelope、amplitude、physical phase、field center 和 physical perturbation evaluation | pump/probe/LO experiment role、TA delay semantics |
| PulseSequence | experiment pulse composition、shared phase grouping、concrete pulse centers/phases 到 physical fields 的映射 | TA delay convention、Fourier targets、detector physics |
| Execution | one-run dynamics execution、solver invocation、checkpoint infrastructure、SimRes | detector/readout physics、Recipe postprocess、phase projection |
| Readout | target `ReadoutPlan`、polarization-to-detector transform、coherent detector physics、`ReadoutResult` | experimental condition subtraction、phase projection |
| Recipe | cases、Conditions、recipe coordinates、delay convention、选择/生成 ReadoutPlan、postprocess、requested target phase orders | SystemScanAxis、generic Fourier projection、solver internals |
| Generic phase projection | `PhaseGrid`、target phase orders、mathematical projection、future multi-target projection、minimal projection metadata | solver、readout、conditions、TA subtraction、Recipe reuse/broadcast |
| Upper-level scripts | SystemScanAxis、factorial system scans、对多个 Systems 重复 Recipe execution | Recipe 内部 coordinate semantics |
| UFANSYS / user analysis | FFT、plotting、alignment、selection、slicing、下游 spectroscopy analysis | solver/readout/Recipe execution orchestration |

## Readout Architecture

目标数据流是：

```text
SimRes -> polarization P -> ReadoutPlan.execute(P) -> ReadoutResult I
```

`ReadoutPlan` 是真正 callable 的 generic abstraction，而不是 `ReadoutSpec` 的
简单 rename。它至少配置 readout field、frequency/transform settings、window、
detector mode 和 detector parameters。第一版允许：

1. 引用 `PulseSequence` 中已经进入 Hamiltonian 的 named interaction field；
2. 直接持有一个不进入 Hamiltonian 的 external `FieldPhyRoot` 作为 guiding/LO。

当前 scope 中 readout/guiding field 不参与 phase cycling；暂不设计 phase-cycled LO。

Readout behavior 至少应能够表达 full detector 与 weak-signal approximation：

```text
I_det(omega) = |E_readout(omega)|^2
             + 2 Re[E_readout*(omega) E_sig(omega)]
             + |E_sig(omega)|^2

E_sig proportional to i * omega * P_sig
```

忽略相应高阶项属于 weak-signal readout approximation，不属于 phase cycling。

当前 `absorption` readout 实际计算：

```text
absorption = omega * Im[P(omega) / E(omega)]
```

它是 susceptibility/absorption-like quantity，不是 full detector intensity。目标 public
术语倾向 `absorption_like_response`，最终 runtime rename 留给后续 milestone 决定。

## Recipe、Condition 与 postprocess

Recipe 对一个给定 System 定义：

- 哪些 simulation/acquisition cases 必须执行；
- 每个 Condition 使用哪个 `PulseSequenceSpec`；
- recipe coordinates 如何映射为 concrete pulse centers；
- 选择或生成哪个 `ReadoutPlan`；
- 如何把 `ReadoutResult` 合成为 `S(...)`；
- 请求哪些 target phase-order vectors。

Condition 始终表示相同 System 下不同 acquisition configuration。PB、EIS、EID、
Hamiltonian、relaxation/dephasing、temperature、model member 或 disorder realization
的改变会产生不同 System，不是 Condition。

TA recipe 可利用自己知道的依赖关系避免重复昂贵模拟：

```text
I_on(T, phi_pump, phi_probe)
I_off(phi_probe)

S(T, phi_pump, phi_probe, omega)
  = [I_on(T, phi_pump, phi_probe, omega) - I_off(phi_probe, omega)]
    / I_off(phi_probe, omega)
```

`I_off` 不依赖 `T` 和 `phi_pump`，所以只执行必要 cases，再由
`Recipe.postprocess` broadcast。`I_off` 是 `deltaT/T` 定义中的 denominator；
`|E_readout|^2` 只可能在 weak-signal 推导中作为近似出现。

delay convention 同样属于 Recipe。PulseSequence 只接收 concrete centers，不知道
“TA positive delay”的含义。当前 TA 可以使用 `probe_center=0`、`pump_center=-T`；
其他 Recipe 可以采用不同 origin，只要明确 coordinate-to-center mapping。

## S 与 Named Axes Contract

`S(phi_1,...,phi_D; other coordinates)` 是 Recipe postprocess 后、phase projection
前的正式接口。它可以是 scalar、real/complex ndarray、`S(omega)`、`S(T,omega)`
或任意高维 observable；generic projection 不解释其物理含义。

最低数据 contract 是：

```text
data: ndarray
axis_names: tuple[str, ...]
axis_values: mapping[str, ndarray]

len(axis_names) == data.ndim
len(axis_values[name]) == data.shape[axis_names.index(name)]
```

phase axes、recipe-coordinate axes 和 payload axes 都必须在 `axis_names` 中有明确
位置。projection 移除被投影的 phase axes，并保留剩余 axis names/values。
不引入 xarray，也不预先设计 heavy named-array hierarchy。

一份 `S(phi)` 必须原生支持多个 targets：昂贵的 solver、readout 和 Recipe
postprocess 各执行一次，只重复廉价 Fourier projection。

## Persistence

一次 Recipe execution 应保存：

1. 所有昂贵 `SimRes`；
2. final projected data；
3. recipe/execution metadata。

中间 `S(phi)` 不要求默认持久化，因为它可以从保存的 `SimRes` 经 readout 和
Recipe postprocess 重建。projected output 至少保存：

- target phase-order vector(s)；
- phase convention 及 convention/schema version；
- normalization；
- phase-grid definition；
- remaining `axis_names` 与 `axis_values`；
- 必要 provenance。

不要为了 persistence 建立大型 result hierarchy。

## Current Abstraction Status

| Current abstraction | Target status | Reason |
| --- | --- | --- |
| `PhaseGrid` | keep | 有独立 phase-domain semantics 与 validation |
| heavy `PhaseCyclingPlan` | target-to-remove | 不应执行 case generation、solver、readout、postprocess 或 aggregation |
| `PhaseProjectionSpec` | target-to-remove | plain projection function 可直接接受 data/axes/grid/targets/normalization |
| `PhaseCaseRecord` | move/remove from projection layer | execution provenance 属于 Recipe/execution manifest |
| heavy `PhaseCyclingResult` | simplify/remove if unnecessary | projected ndarray(s) 加最小 metadata 即可 |
| `AxisMetadataSpec` | candidate for simplification/removal | 当前与 ndarray dimension 绑定不足且依赖 SingleRun quantity extraction |
| `ProjectedReadoutBundle` | target-to-remove | projection 输入是 recipe observable，不一定是 readout |
| `TAPhaseCyclingSpec` | split into Recipe inputs | targets 属于 Recipe；projection math 属于 generic layer |
| `TAPhaseCycledPumpProbeResult` | target-to-remove | projected data 应保持 experiment-generic |
| TA projected-bundle builders | target-to-remove | 不再需要 readout-specific projection wrapper |
| `ReadoutSpec` | redesign later | 目标是 behavioral `ReadoutPlan`，不是简单 rename |
| `SingleRunReadoutResult` | redesign later | 目标为独立 `ReadoutResult` |
| `SingleRunFieldPlan` | defer | 是否仍需单独存在要在 execution/readout 重构后评估 |
| `independent_phase` | defer | 可能形成 phase-independence 双 source of truth |
| System adapter initial-state/dephasing mapping | defer/separate issue | 问题真实但与 PC/Readout/Recipe 重构正交 |

Milestone 0 不删除、rename 或改变以上任何 runtime object。

## Planned Migration Boundaries

- **Milestone 1:** 迁移 Fourier implementation 到固定 `+i` convention，增加
  convention/version metadata 与 synthetic harmonic tests。
- **Later readout milestone:** 从 `SingleRunPlan` 抽取 behavioral `ReadoutPlan`，
  支持 interaction-field reference、external field、full/weak detector modes。
- **Later recipe milestone:** 先 readout，再 `Recipe.postprocess`，最后 generic
  multi-target phase projection；实现昂贵 cases 的 reuse/broadcast。
- **Later cleanup milestone:** 数值/物理验证和 example/test migration 后，才标记
  heavy runner、bundle 和 TA projected wrappers deprecated，随后删除。
- **Separate system milestone:** 修复 `NLevelSystem.initial_state` 与 transition
  dephasing 到 solver execution 的完整映射。

## Explicit Non-goals

当前架构不要求：

- generic `ConditionPlan`、`RecipeCase`、`DelayAxis` 或 `PostprocessPlan`；
- xarray 或 heavy named-array class；
- phase-cycled readout/LO field；
- 把 SystemScanAxis 放入 Recipe；
- 在 generic phase projection 中执行 solver、readout、subtraction 或 TA logic；
- 在本次 terminology freeze 中修改任何 runtime behavior。
