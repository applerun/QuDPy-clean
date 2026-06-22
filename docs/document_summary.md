# 当前文档总览与更新摘要

本文档记录 `QuDPy-clean` 当前文档主线、active code path、模块边界和需要替换的旧表述。当前代码包路径统一为：

```text
qudpy_sjh/
```

## 当前主线

当前 active code path 是：

```text
FieldPhyRoot / FieldPhyCustomed / TimeShiftedField / FieldPhySeries
+ carrier_envelope.CarrierEnvelopeField
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
4. `ParaNormalizer` 只负责单位换算和 code-unit scaling，不定义物理模型。
5. 多脉冲叠加只发生在 physical field 层，例如 `FieldPhySeries`。
6. pump / probe / LO 这类 role 不属于单个 `CarrierEnvelopeField`，应由 `FieldPhySeries.sub_field_names`、case metadata 或 workflow 层表达。
7. solver 主路径只接收一个 code-unit callable：`args={"field": field}`。
8. 当前主线是 full-window `lab_exact`。
9. 标准结果对象是 `DynamicsResult`。
10. 一个 `DynamicsResult` 只表示一条 density-matrix trajectory。
11. 参数扫描、delay scan、phase cycling、power scan 和 batch run 属于上层 examples/scripts/workflow，不属于 core。
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
│  │  └─ carrier_envelope/
│  │     ├─ __init__.py
│  │     ├─ carrier_spec.py
│  │     ├─ envelope_spec.py
│  │     ├─ carrier_envelope_field.py
│  │     └─ builders.py
│  ├─ spectroscopy/
│  │  ├─ __init__.py
│  │  ├─ observables.py
│  │  ├─ absorption_spectra.py
│  │  └─ theory.py
│  ├─ plotting.py
│  ├─ io.py
│  └─ checks.py
```

新文档应以 `utils/core/`、`utils/fields/`、`utils/fields/carrier_envelope/` 和 `utils/spectroscopy/` 为准。

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
- `deprecated_boundaries_zh.md` 明确旧架构和旧 field helper 边界；
- `docs/detailed API/` 下的文档记录专题 API 与 workflow 边界。

## fields 层摘要

`utils/fields/` 描述用户侧 lab-frame physical field。所有内置 field 都应继承或符合 `FieldPhyRoot` 接口。

当前结构：

```text
qudpy_sjh/utils/fields/
  __init__.py
  lab_fields.py
  field_series.py
  carrier_envelope/
    __init__.py
    carrier_spec.py
    envelope_spec.py
    carrier_envelope_field.py
    builders.py
```

`qudpy_sjh.utils.fields` 当前 public API：

```text
FieldPhyRoot
FieldPhyCustomed
TimeShiftedField
make_code_field_adapter
FieldPhySeries
iter_scan_params
```

`qudpy_sjh.utils.fields.carrier_envelope` 当前主要 API：

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

`TimeShiftedField` 的约定是：

```text
field.time_shifted(shift_fs)
shift_fs > 0 表示场整体向更晚时间移动
E_shifted(t) = E_original(t - shift_fs)
```

`CarrierEnvelopeField` 的物理约定是：

```text
E(t) = 2 E0 envelope(t) cos[omega * (t - center) + phase]
```

其中 `center` 来自 envelope 的 `center_fs`。`phase_rad` 是相对于 envelope center 定义的 carrier phase，不是全局 lab-frame `omega*t + phase`。

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
- 不根据 Gaussian / CW / pump-probe / TA / 2DES 写特殊分支。

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

主定义：

```text
absorption = omega * Im[P(omega) / E(omega)]
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

## TA workflow 摘要

当前 TA workflow 应保持分层：

```text
field layer:
  make_gaussian_carrier_envelope_field(...)
  FieldPhySeries(fields=(pump, probe), sub_field_names=("pump", "probe"))

solver layer:
  ordinary full-window run_case

spectroscopy layer:
  polarization_C_per_m2(...)
  lab_frame_absorption_response(...)

workflow layer:
  delay scan
  pump phase cycling
  probe-only reference reuse
  differential response map
  plotting / CSV / JSON
```

TA response 在 spectroscopy / workflow 层计算：

```text
S_TA = omega * Im[P_pump_probe(omega)/E_probe(omega)]
       - omega * Im[P_probe_only(omega)/E_probe(omega)]
```

不要把 TA map、phase cycling、plotting 塞回 core。

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

## 本次更新摘要

本次文档口径更新重点：

1. 将文档主线统一到当前 `qudpy_sjh/` 代码路径。
2. 将 field 主线改为 `FieldPhyRoot`、`FieldPhyCustomed`、`TimeShiftedField`、`FieldPhySeries` 和 `carrier_envelope` 子包。
3. 移除旧 `specific/basic_fields.py`、`specific/ta_fields.py`、`specific/twodes_fields.py` 作为推荐 public API 的表述。
4. 移除 `CarrierFieldPhysical`、`GaussianCarrierFieldPhysical`、`TAField`、`TwoDESField` 和相关 old helper 作为当前主线 API 的表述。
5. 明确 carrier-envelope phase 是相对于 envelope center 定义的。
6. 明确 pump / probe / LO role 不属于单个 `CarrierEnvelopeField`。
7. 将 TA workflow 改为当前 demo 的分层方式。
8. 明确 probe-only reference 可以只运行一次并复用。
9. 将 spectroscopy 主入口更新为 `lab_frame_absorption_response(...)`。
10. 将 `lab_frame_fft_response_legacy(...)` 标记为较底层 / 历史兼容 helper。
11. 明确 solver 主线是 full-window `lab_exact`。
12. 明确标准结果是普通 `DynamicsResult`。
13. 明确 parameter scan / delay scan / phase cycling / batch workflow 属于 examples/scripts 层。
14. 明确普通 IO 只保存 full-window `DynamicsResult`。
15. 移除或降级 piecewise / dark / materialization / long-window benchmark / complex transient_absorption scaffold 相关表述。

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
