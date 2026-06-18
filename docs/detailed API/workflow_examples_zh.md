# Examples 与 Workflow 边界说明

本文档说明当前 `QuDPy-clean` 中 examples / scripts / workflow 层应承担的职责，避免把扫描、批处理或分析地图逻辑塞回 core。

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
power scan
batch run
probe-only reference
结果汇总
绘图排版
CSV / map 生成
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
iter_ta_gaussian_fields
iter_twodes_gaussian_fields
```

这些 helper 只负责生成 field 或参数组合，不负责运行 solver、不负责保存文件、不负责画图。

## TA / pump-probe workflow

当前 TA 主线应分层：

```text
fields layer:
  TAField
  make_ta_gaussian_field
  make_pump_probe_field_from_templates

solver layer:
  ordinary full-window run_case

spectroscopy layer:
  lab_frame_absorption_response

workflow layer:
  delay scan
  pump+probe / probe-only 对照
  differential response
  map / figure / CSV
```

## delay convention

template-based pump-probe helper 的约定是：

```text
probe_center_fs 默认 0
pump_center_fs = probe_center_fs - delay_fs
```

该约定适合 probe-anchored workflow。

direct Gaussian helper `make_ta_gaussian_field(...)` 的约定是：

```text
probe_center_fs = pump_center_fs + probe_delay_fs
```

文档和脚本中应明确使用哪一个 helper，避免混淆。

## TA intrinsic response 示例边界

一种简单的 intrinsic response workflow：

```text
for each delay:
    construct pump+probe TAField
    run pump+probe with full TAField
    run probe-only with TAField["probe"]
    compute P_pump_probe(t)
    compute P_probe_only(t)
    compute absorption response using the same E_probe(t)
    subtract spectra
```

概念公式：

```text
S_TA(omega, delay)
=
omega * Im[P_pump_probe(omega, delay) / E_probe(omega)]
-
omega * Im[P_probe_only(omega) / E_probe(omega)]
```

其中：

```text
P(t) = number_density * Tr[rho(t) mu]
E_probe(t) = TAField["probe"](t_fs)
```

该 response 是模拟层的材料本征响应，不包含：

```text
传播效应
样品厚度
transmitted intensity
detector response
实验仪器响应
```

这些实验链路因素不属于当前 QuDPy 主线。

## probe-only reference

如果采用 probe-anchored convention：

```text
probe_center_fs 固定
pump_center_fs 随 delay 移动
```

则 probe-only reference 可以尽量只运行一次，并在多个 delay 中复用。

但复用前应确保：

```text
time grid 一致
probe field 一致
window / mask 规则一致
E_probe(omega) 一致
```

## spectroscopy workflow

推荐 workflow：

```python
rho_t = result.density_array()
E = result.field_MV_per_cm_values()
P = polarization_C_per_m2(
    rho_t,
    result.physical_params.dipole_matrix_D,
    number_density_m3=...,
)

response = lab_frame_absorption_response(
    time_fs=result.times_fs,
    polarization_C_per_m2=P,
    field=E,
)
```

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

`save_result_case(..., output_preview=True)` 可以生成和保存低清 preview，但这仍属于 IO / presentation 层，不改变 `DynamicsResult` 主结构。

## metadata workflow

推荐在上层 workflow 中设置：

```text
example_name
condition_name
case_name
input_description
input_metadata
```

这些字段只用于 metadata，不参与求解。

不要把 workflow-specific 信息塞进 normalizer 或 solver core。

## 不应进入 workflow 主线的旧内容

当前 workflow 文档不应宣传：

```text
piecewise propagation
dark propagation
active/dark PropagationPiece
PieceDynamicsResultSeries
materialize_full
long-window piecewise benchmark
complex transient_absorption scaffold
```

如果未来需要 long-window acceleration，应先单独写设计文档，再决定是否引入新的 result object 或 IO schema。
