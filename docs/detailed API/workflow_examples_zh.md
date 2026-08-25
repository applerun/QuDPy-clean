# Examples 与 Workflow 边界说明

本文档说明当前 `QuDPy-clean` 中 examples / scripts / workflow 层应承担的职责，避免把扫描、批处理、phase cycling 或 response map 逻辑塞回 core。

## 基本原则

core 层只负责：

```text
参数 dataclass
归一化
Hamiltonian / collapse operator 构造
单次 full-window 求解
单条 DynamicsResult 容器
```

workflow 层负责：

```text
parameter scan
delay scan
phase cycling
power scan
batch run
probe-only reference
结果汇总
绘图排版
CSV / JSON / map 生成
```

## 普通 workflow 模式

推荐写普通上层循环：

```python
results = []

for scan_value in scan_values:
    field = ...
    params = NLevelPhysicalParams(
        energies_eV=...,
        dipole_matrix_D=...,
        t_start_fs=...,
        t_end_fs=...,
        dt_fs=...,
        field=field,
    )
    result = run_case(params)
    results.append((scan_value, result))
```

不要恢复：

```text
ParameterSweep
run_parameter_sweep
PhysicalParameterSweep
run_physical_parameter_sweep
```

除非未来重新设计并明确其与当前 core 的边界。

## field scan helper

`FieldPhySeries` 和 field helper 可以提供轻量 scan helper，例如：

```text
iter_scan_params
```

这些 helper 只负责生成 field 或参数组合，不负责运行 solver、不负责保存文件、不负责画图。

## TA / pump-probe workflow

当前 TA 主线应分层：

```text
field layer:
  probe_field = make_gaussian_carrier_envelope_field(...)
  pump_field = make_gaussian_carrier_envelope_field(...)
  field = FieldPhySeries(
      fields=(pump_field, probe_field),
      sub_field_names=("pump", "probe"),
  )

solver layer:
  ordinary full-window run_case

spectroscopy layer:
  polarization_C_per_m2
  lab_frame_absorption_response

workflow layer:
  delay scan
  pump phase cycling
  pump+probe / probe-only 对照
  differential response
  map / figure / CSV / JSON
```

`CarrierEnvelopeField` 不存放 pump / probe / LO role。role 应由 `FieldPhySeries.sub_field_names`、case metadata 或 workflow 层表达。

## delay convention

当前 TA demo 采用 probe-anchored convention：

```text
probe_center_fs 固定
pump_center_fs = probe_center_fs - delay_fs
positive delay means pump before probe
```

推荐在 workflow 中显式写：

```python
probe_center = config.probe_center_fs
pump_center = probe_center - delay_fs
```

不要把 delay convention 隐式塞进 core field 类。

## 当前 TA workflow

当前 canonical example 是
`bin/examples/ta/ta_three_level_canonical_phase_step_convergence.py`：

```python
recipe = TAPrePCRecipe(
    base_params=base_params,
    pump=pump,
    probe=probe,
    delays_fs=delays_fs,
    phase_grid=phase_grid,
    readout_plan=ReadoutPlan(...),
)
dynamics = recipe.execute_dynamics()
readouts = recipe.apply_readout(dynamics)
observable = recipe.postprocess(readouts)
projected = project_phase_orders(
    observable.data,
    axis_names=observable.axis_names,
    axis_values=observable.axis_values,
    phase_grid=phase_grid,
    targets=targets,
)
```

Pulse templates、phase tags 与 centers 由 recipe 映射为 concrete physical fields。
不在脚本中维护第二套 phase-case runner。

## probe-only reference

在 probe-anchored convention 下：

```text
probe_center_fs 固定
probe phase 固定
probe envelope 固定
pump_center_fs 随 delay 移动
```

因此 probe-only reference 可按真实 probe-phase dependency 运行一次，并在多个 delay
和多个 pump phase case 中复用。

复用前应确保：

```text
time grid 一致
probe field 一致
window / mask 规则一致
E_probe(omega) 一致
```

## phase cycling

phase cycling 是 workflow 层逻辑，不属于 field core。

`PhaseGrid` 可定义任意 N、任意 finite phase values 和多个 phase tags；
`project_phase_orders` 对 recipe 产生的完整 `S(phi)` 执行统一的 `+i` Fourier
projection。不要为 phase cycling 新增特殊 `DynamicsResult` 类型，也不要把 phase
average 写回 solver core。

Detector-level TA 由 recipe postprocess 定义为：

```text
delta_T_over_T = (I_on - I_off) / I_off
```

absorption-like 分析则显式命名为：

```text
delta_absorption_like = A_on - A_off
```

## spectroscopy workflow

推荐 workflow：

```python
rho_t = result.density_array()
P = polarization_C_per_m2(
    rho_t,
    result.physical_params.dipole_matrix_D,
    number_density_m3=...,
)

response = lab_frame_absorption_response(
    time_fs=result.times_fs,
    polarization_C_per_m2=P,
    field=probe_field(result.times_fs),
)
```

注意：TA response 的分母应使用 `E_probe(t)`，不是 pump+probe 总场。

如果要保存 response，应在 workflow 层另存为：

```text
absorption_response.csv
ta_response_map.csv
analysis_components.csv
```

不要把 absorption response 塞入 `DynamicsResult` 或普通 `components.csv`。

## plotting workflow

plotting 层应只生成 matplotlib figure 或 axes，不默认保存文件。

保存 figure 属于 IO 或上层脚本职责：

```python
fig, ax = ...
save_figure(fig, "path/to/figure.png")
```

TA map、phase-cycling comparison、preview trace、rho preview 等都属于 workflow / plotting 层，不应进入 core。

## metadata workflow

推荐在上层 workflow 中设置：

```text
example_name
condition_name
case_name
input_description
input_metadata
field role
delay_fs
pump_phase_rad
probe_phase_rad
response_definition
```

这些字段只用于 metadata，不参与求解。

不要把 workflow-specific 信息塞进 normalizer 或 solver core。

## 不应进入 workflow 主线的旧内容

当前 workflow 文档不应宣传：

```text
piecewise propagation
dark propagation
ActiveWindow
PropagationPiece
PieceDynamicsResultSeries
materialize_full
run_case(piecewise=...)
save_result_case 对 piecewise series 的支持
long-window piecewise benchmark
complex transient_absorption scaffold
TAField
TwoDESField
make_ta_gaussian_field
make_pump_probe_field_from_templates
make_twodes_gaussian_field
```

如果未来需要 long-window acceleration 或完整实验传播模型，应先单独写设计文档，再决定是否引入新的 result object、solver entry point 或 IO schema。
