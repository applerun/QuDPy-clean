# Pulse Sequence 与 Phase Cycling Architecture

本文记录 M7 清理后的 active architecture。历史接口由 Git history 保存，不属于
当前运行时 API。

## Canonical data flow

```text
System
  -> PulseSpec / PulseSequenceSpec
  -> SingleRunPlan.execute()
  -> SingleRunResult / DynamicsResult
  -> compute_polarization_result(...)
  -> ReadoutPlan.execute(...)
  -> ReadoutResult
  -> TAPrePCRecipe.postprocess(...)
  -> TAPrePCObservable S(...)
  -> project_phase_orders(...)
  -> lightweight projected mapping
  -> save_projected_result / load_projected_result
```

昂贵状态只由 `DynamicsResult` checkpoint 保存。readout、recipe postprocess 与
phase projection 都可从保存的 dynamics 重新计算，不需要重新运行 solver。

## Layer ownership

| Layer | Owns | Does not own |
| --- | --- | --- |
| System | Hamiltonian、dipoles、initial state、relaxation/dephasing | pulse roles、delay、phase projection |
| Field | envelope、carrier、physical carrier phase、field evaluation | pump/probe 语义、Fourier target |
| Pulse sequence | physical pulses、groups、shared phase tags、centers 与 phase override | detector、TA subtraction、solver |
| Single run | 一个 concrete physical field 的 dynamics execution 与 checkpoint | readout、phase cycling |
| Readout | polarization 到 detector/absorption-like observable | acquisition conditions、projection |
| TA recipe | pump-on/off cases、delay convention、reuse/broadcast、TA observable | System scan、Fourier math |
| Phase projection | phase grid、integer target orders、named-axis Fourier sum | solver、readout、TA logic |
| Persistence | projected mapping 的 NPZ arrays 与 strict JSON metadata | runtime result hierarchy |

## Physical phase

`PulseSpec.phase_tag` 把 recipe 提供的 phase value 映射到真实 physical field。
`PulseSequenceSpec.build_field(...)` 对 field template 调用 phase-shift 能力，因此
phase cycling 改变进入 Hamiltonian 的场，而不只是 metadata。

`FieldGroupSpec` 可包含多个 pulses。group-level shared phase 会统一加到组内 field；
多个 pulses 也可共享同一 phase tag。phase tag 名称由调用者定义，不写死为
`pump` 或 `probe`。

## Single-run boundary

`SingleRunPlan` 输入：

- `NLevelPhysicalParams`；
- `SingleRunFieldPlan`；
- normalizer 与可选 checkpoint settings；
- case/provenance metadata。

`execute()` 只返回 dynamics-oriented `SingleRunResult`。该 result 不含 readout
字段。任意 detector 处理必须显式经过：

```python
polarization = compute_polarization_result(single_run.dynamics_result, ...)
readout = ReadoutPlan(...).execute(
    polarization,
    interaction_field=single_run.params.field,
)
```

## Readout

Active modes：

- `polarization`：返回 polarization time trace；
- `absorption_like`：返回 `absorption_like_response`；
- `full`：`|E_readout + E_signal|^2`；
- `weak`：`|E_readout|^2 + 2 Re[E_readout* E_signal]`。

`readout_field` 可以引用 interaction field 中的 named subfield，也可以是一个不进入
Hamiltonian 的 external physical field。readout 不是 phase cycling 的隐式组成部分。

## TA recipe

`TAPrePCRecipe` 定义：

```text
pump_on(T, phase dimensions)
pump_off(probe phase if cycled)
```

delay convention 为：

```text
pump_center_fs = probe_center_fs - T
```

因此正 `T` 表示 pump 先到。pump-off 只按真实依赖的 probe phase 计算一次，再在
postprocess 中沿 `T` 与 pump phase broadcast。

Detector observable：

```text
delta_T_over_T = (I_on - I_off) / I_off
```

兼容物理分析需要的 absorption-like difference 明确命名为：

```text
delta_absorption_like = A_on - A_off
```

`TAPrePCObservable` 保存 data、difference、named axes、denominator mask 与 diagnostics；
它是 recipe postprocess boundary，不是 projected-result wrapper。

## Fourier convention

pathway phase dependence：

```text
S(phi) proportional to exp(-i * sum_j(m_j * phi_j))
```

projection：

```text
S_m = 1 / product_j(N_j)
      * sum_phase_cases S(phi) * exp(+i * sum_j(m_j * phi_j))
```

`target_phase_vector` 直接表示 physical integer phase-order vector。例如：

```python
{"pump": 0, "probe": 1}
```

Fourier sign 已冻结为 `+i`，不再是可配置 runtime 参数。

## Generic projection API

Active public phase API：

```python
PhaseGrid
build_uniform_phase_grid
project_phase_orders
```

`PhaseGrid` 支持任意有限 phase values、任意 N、多个 phase tags，以及每个 tag
不同的 N。uniform helper 构造完整 Cartesian grid。对 nonuniform values 执行的是
equal-weight sum，不宣称一般 nonuniform Fourier inversion。

`project_phase_orders` 输入 precomputed ndarray、named axes、phase grid 与一个或多个
target vectors。所有 phase axes 被移除，其他 axes 保持相对顺序。归一化 divisor
是实际 Cartesian phase-case 数，不写死为 4。

返回普通 mapping：

```text
projected[target_name] -> ndarray
axis_names             -> remaining axes
axis_values            -> remaining coordinates
targets                -> normalized integer vectors
metadata               -> convention, grid, normalization, provenance
```

## Persistence

`save_projected_result` 保存 compressed NPZ 与 strict JSON manifest；
`load_projected_result` 校验 schema、Fourier convention、targets、axes 与 arrays 后返回
同一 lightweight mapping。complex/NaN arrays 在 NPZ 中无损保留，JSON diagnostics
遵循 strict serialization policy。

## Public surface

Phase cycling：`PhaseGrid`, `build_uniform_phase_grid`, `project_phase_orders`。

Readout：`PolarizationResult`, `ReadoutPlan`, `ReadoutResult`,
`compute_polarization_result`。

TA：`TADelayCenters`, `TAPrePCRecipe`, `TAPrePCObservable`。

Persistence：`save_projected_result`, `load_projected_result`。

## Historical architecture removed in M7

M7 删除了 heavy phase runner family、axis/readout projected bundles、TA v1/v2 plans
与 wrappers，以及 embedded single-run readout compatibility path。旧名称和旧用法只在
Git history 中保留；active code 不提供 aliases、adapters 或 deprecation warnings。

System adapter 的 initial-state/dephasing 完整映射仍是独立 milestone，本轮未修改。
