# sjh_learn

`sjh_learn` 是当前用来学习 two-level optical Bloch dynamics 的小型 QuTiP 项目。现在的主线已经收口到单轨迹架构：一次求解只产生一个 `DynamicsResult`，顶层脚本负责决定要跑哪些 case、怎样拼图、怎样保存最终图像。

## 当前架构

- 标准结果对象是 `DynamicsResult`。
- 一个 `DynamicsResult` 只表示一条 density-matrix trajectory。
- 当前主线是 `lab_exact`。`rwa` 是 legacy mode，默认禁用。
- `rotating_view` 不是独立求解，而是由 `lab_exact` 结果通过幺正变换派生。
- solver 层一次只返回一个 result，不同时运行多个 mode。
- result 层只保存数值轨迹和 metadata，不依赖 matplotlib。
- plotting 层只画到 matplotlib `fig/axes` 上并返回句柄，不默认保存文件。
- 顶层脚本负责组合多个 result、排版、加总标题、保存最终图。
- `io.py` 只负责保存 result 数字数据、metadata 和已经构造好的 figure。

## Field / Drive

`utils/fields/` 按物理含义和单位边界拆分：

- `utils/fields/lab_fields.py`：用户侧 lab-frame physical field，包括 `CarrierFieldPhysical` 和 `GaussianCarrierFieldPhysical`。
- `utils/fields/field_series.py`：physical-layer 多场组合，包括 `FieldPhySeries`、`TAField` 和 `TwoDESField`。
- `utils/fields/__init__.py`：公共 re-export，只导出 physical field API。

用户侧 field class 只使用物理单位。`CarrierFieldPhysical.__call__(t_fs)` 返回单位为 `MV/cm` 的 `E(t)`。普通示例应显式构造 physical field，再传入 `NLevelPhysicalParams(..., field=field)`；solver 内部 code-unit callable 只能由 `ParaNormalizer.make_code_field()` 生成。

已支持：

- `CarrierFieldPhysical`: lab frame 载波场，`E(t) = 2E0 cos(omega t + phase)`。
- `GaussianCarrierFieldPhysical`: lab frame 高斯包络载波场。
- `FieldPhySeries`: physical-layer 多个 field 相加。
- `TAField` / `TwoDESField`: TA / 2DES 常用多脉冲 field。

每个 field/drive 支持：

- `__call__(t)`: 接受 scalar 或 numpy array。
- `to_dict()` / `rebuild(...)`: 用于可靠保存和重建。
- `__repr__()`: 用于人类可读日志。

RWA 当前默认禁用；旧 solver-unit field / drive 类已经从当前 API 中删除。

## 结果和单位

`DynamicsResult` 保存：

- `mode`
- `times`
- `times_fs`
- `states`
- `parameters`
- `physical_params`
- `solver_params`
- `metadata`
- `source_mode`
- input drive/field 的 `to_dict()` 和 `to_expr()`

density matrix、population、coherence 都是无量纲量。CSV 中主时间轴保存为真实 `time_fs`；NPZ 中同时保存 `time_fs` 和内部求解用的 `time_code`，方便回溯归一化。

## Spectroscopy Observables

`utils/analysis/observables.py` 提供第一层谱学 observable，属于 analysis 层：

- `dipole_expectation_D(rho_t, dipole_matrix_D)` 计算 `p(t)=Tr[rho(t) mu]`，单位 Debye。
- `polarization_C_per_m2(rho_t, dipole_matrix_D, number_density_m3)` 计算 `P(t)=N p(t)`，单位 `C/m^2`，其中 `number_density_m3` 的单位是 `m^-3`。
- `chi_two_level_linear(...)` 给出 two-level analytic linear-response susceptibility，用作后续数值谱学结果的参考。

迹的指标约定是 `Tr(rho mu)=sum_ij rho_ij mu_ji`，代码使用 `np.einsum("tij,ji->t", rho_t, mu)`。这里必须使用物理输入 `dipole_matrix_D`，不能使用已经包含光场强度和 code-unit 归一化的 `coupling_matrix_code`。

`DynamicsResult.components_dataframe()` 和 simulation 输出的 `components.csv` 只保存 density matrix、population、coherence 以及输入 drive/field 相关列；dipole expectation、polarization、FFT response 和吸收功相关量只在 analysis 层输出，例如 `analysis_components.csv` 和 `fft_response.csv`。

## 绘图和 IO

`utils/plotting.py` 提供低层绘图函数：

- `plot_drive(result, ax=None, ...)`
- `plot_field(field, times, ax=None, ...)`
- `plot_populations(result, ax=None, ...)`
- `plot_coherences(result, ax=None, ...)`
- `plot_density_components(result, axes=None, include_drive=False, ...)`
- `plot_multilevel_components(result, axes=None, ...)`
- `build_preview_figure(result, ...)`

所有绘图函数只返回 `fig, ax/axes`，不保存 PNG。低清 preview 由 `plotting.py` 构建，由 `io.py` 保存。

`utils/io.py` 提供：

- `save_result_data(result, output_dir, save_npz=True, save_csv=True, save_json=True)`
- `save_figure(fig, output_path, dpi=120)`
- `save_result_case(result, output_dir, output_data=True, output_preview=False, ...)`
- `QuantumResultIO`

默认 case 输出结构：

```text
outdir/
├─ results.csv
└─ <case_name>/
  ├─ data/
  │  ├─ density.npz
  │  ├─ components.csv
  │  └─ populations.csv
  ├─ figs/
  │  ├─ preview.png
  │  └─ full.png
  └─ meta.json
```

完整数据和预览图可以分别开关。`output_preview=True` 时，如果没有传入 `preview_fig`，IO 层会调用 `build_preview_figure(result)` 生成低清预览图；result 对象本身不画图。

## Demo 和 Example

`sjh_learn/bin/multilevel_demo.py` 是当前 N-level explicit-field API demo：脚本显式构造 field 对象，再传入 `NLevelPhysicalParams(..., field=field)`。

`sjh_learn/bin/n2_equivalence_check.py` 是 N=2 regression check，用来确认 N=2 physical mainline 与显式 normalize+solver 路径保持一致。

稳定示例放在 `sjh_learn/examples/absorption/`：

- `cw_pulse_absorption_compare.py`：保留 lab_exact 主线和 legacy RWA 对比分支；默认不运行 RWA。
- `three_level_absorption_lab_exact.py`：two-level 与 three-level lab_exact absorption 对比。
- `absorption_01_chi_analytic.py`：解析 two-level linear susceptibility 参考。

`scratch/` 只保留临时验证脚本，不放稳定 example。

## 运行检查

```powershell
conda --no-plugins run -n quantum python -m compileall sjh_learn scratch
conda --no-plugins run -n quantum python sjh_learn\bin\multilevel_demo.py
conda --no-plugins run -n quantum python sjh_learn\bin\n2_equivalence_check.py
conda --no-plugins run -n quantum python sjh_learn\examples\absorption\cw_pulse_absorption_compare.py
conda --no-plugins run -n quantum python sjh_learn\examples\absorption\three_level_absorption_lab_exact.py
```

当前阶段 QuDPy 主线维护 lab-frame / exact solver；RWA solver-unit drive 路径默认禁用，
只保留 legacy diagnostic 后处理说明。不要把 RWA 作为默认 fallback。

推荐的 field 输入方式是显式构造 physical field：

```python
field = make_default_gaussian_carrier_field(
    E0_MV_per_cm=0.05,
    laser_energy_eV=1.625,
    pulse_center_fs=0.0,
    pulse_sigma_fs=8.0,
)

params = NLevelPhysicalParams(
    energies_eV=...,
    dipole_matrix_D=...,
    t_start_fs=...,
    t_end_fs=...,
    dt_fs=...,
    field=field,
)

result = run_case(params)
```

Each case writes two metadata files. `meta.json` is a short human-readable summary with `example_name`, `condition_name`, `case_name`, physical N-level inputs, physical field/drive information, derived physical rates, trajectory summary, and output-file paths. `debug_meta.json` keeps the full raw `DynamicsResult.metadata_dict()` payload, including full code parameters, `tlist`, `times_fs`, code-unit drive metadata, solver internals, and sanity checks.

## Unit Conventions

- 现在用户侧标准物理系统对象是 `NLevelPhysicalParams`。two-level system 不再是核心层的特殊标量模型，而是 `N=2` 的普通 N-level system；multilevel system 也是普通 `N>2` system。
- `dipole_matrix_D` 是沿选定 optical polarization 投影后的偶极矩矩阵，单位是 Debye。光场由 `field` 对象唯一描述，归一化时 `ParaNormalizer` 读取 `field.reference_MV_per_cm` 生成 coupling matrix。
- population relaxation 由 `relaxation_channels` 定义，每个通道表示 `C_{to <- from} = sqrt(rate) |to><from|`。通道可用 `T1_fs` 或 `rate_fs_inv` 指定速率。
- pure dephasing 由 `pure_dephasing_channels` 定义，每个通道表示 `C_level^phi = sqrt(rate) |level><level|`。通道可用 `Tphi_fs` 或 `rate_fs_inv` 指定速率。
- 标量 `dipole_D`、`T1_fs`、`Tphi_fs`、`T2_fs` 不再是核心物理模型输入；如果某个 N=2 example 需要这些概念，会在 example-local helper 中把它们翻译成 `dipole_matrix_D` 和 channel list。
- `NLevelPhysicalParams` 使用真实物理单位，例如 `energies_eV`、`dipole_matrix_D`、`time_fs` 和 `fs^-1` 速率；光场信息来自 `field` 对象。
- `NLevelSolverParams` 是内部 solver 参数容器，保存 N-level matrices、channel lists 和 code-unit 时间/频率；普通用户示例不直接构造它。
- `ParaNormalizer` 把这些物理单位转换为 solver 内部使用的 code unit，包括时间、频率、drive 和 decay rate。
- `model.py` 构造 N-level 的 `H0`、`H_int(t)` 和 Lindblad `c_ops`；`solvers.py` 每次只返回一条 `DynamicsResult` 轨迹。
- 用户侧 field 类只使用物理单位：`E0_MV_per_cm`、`laser_energy_eV` 或 optional `omega_L_fs_inv`、`phase_rad`、`pulse_center_fs`、`pulse_sigma_fs`。
- Plotting, CSV export, and example summaries default to physical units when available.
- Code-unit outputs are kept only as metadata or clearly labeled diagnostic fields such as `drive_code`.
- Density matrix, populations, and coherences are dimensionless.

## Parameter Scan

参数扫描、delay scan、power scan 和 batch run 属于 examples / workflow 层，不属于
core solver API。推荐写普通上层循环：

```python
results = []
for scan_params, field in iter_ta_gaussian_fields(...):
    params = NLevelPhysicalParams(
        energies_eV=...,
        dipole_matrix_D=...,
        t_start_fs=...,
        t_end_fs=...,
        dt_fs=...,
        field=field,
    )
    results.append((scan_params, run_case(params)))
```

## Dynamics Analysis

- `DynamicsResult.save_ckp(path)` 可以把一次模拟得到的完整 result 保存为 `.ckp` checkpoint；`DynamicsResult.from_ckp(path)` 可以从 checkpoint 重新加载 result。`.ckp` 是内部 checkpoint / 后处理缓存，不保证跨版本长期稳定，不应加载不可信来源文件，也不是替代 `density.npz`、`components.csv`、`meta.json`、`debug_meta.json` 的归档格式。
- `utils/analysis/DynamicsAnalysis` 是 analysis 层对象，可由 `DynamicsAnalysis.from_dynamics_res(result)` 或 `DynamicsAnalysis.from_ckp(path)` 创建。
- analysis 层只读取 `DynamicsResult` 的公开 API，不调用 solver，也不使用 solver-internal code-unit 输入。
- analysis 默认使用通用 N-level polarization：`P(t)=number_density_m3 * Tr[rho(t) mu_D] * DEBYE_TO_C_M`，其中 `mu_D` 来自 `physical_params.dipole_matrix_D`，`number_density_m3` 必须显式传入。
- `Tr[rho(t) mu_D]` 的实现使用 `np.einsum("tij,ji->t", rho_t, dipole_matrix_D)`；若物理偶极矩期望值的虚部超过容差，analysis 会直接报错，不会静默丢弃。
- 常见 two-level 0-1 近似公式 `P(t)=2 n mu_01 Re[rho_01(t)]` 只作为文档说明保留；analysis API 不提供 two-level 专用 polarization 函数，避免误用于 N-level 体系。
- `rho_over_E = fft_rho12 / fft_E` 是 coherence response-like quantity，不是 `chi` 或 absorption。`P_over_E = P_fft / fft_E` 和 `omega_Im_P_over_E` 更接近 polarization response / 吸收功方向的分析量，但仍依赖 Fourier convention 和线性响应条件。
- 频域 CSV 同时保存 `frequency_fs_inv`、`angular_frequency_fs_inv` 和 `energy_eV`；时域 CSV 保存为 `analysis_components.csv`，避免和 simulation output 的 `components.csv` 混淆。
- 默认分析输出目录是 `outputs/analysis/<example_name>/<case_name>/`，包含 `analysis_components.csv`、`fft_response.csv`、`figs/polarization_time.png`、`figs/fft_response.png` 和 `analysis_metadata.json`。

## Multi-Level Result Export

- `components.csv` is dimension-aware. It saves all diagonal density-matrix elements as populations (`rho_00`, `rho_11`, ...).
- Upper-triangular off-diagonal elements are saved as coherences with `Re_rho_ij`, `Im_rho_ij`, `abs_rho_ij`, `phase_rho_ij`, and `phase_rho_ij_unwrapped` columns, using zero-based indices.
- `populations.csv` saves every diagonal population, and `density.npz` always contains the full density-matrix trajectory.
- `DynamicsResult` is a dimension-aware result object and does not provide a two-level-only `components()` helper.
- Two-level demos or legacy comparison scripts may still use two-level-specific `rho_11` / `rho_01` helper extraction; that logic lives in example-level helpers, not in generic result export.
- 官方 multilevel 路线已经统一为 `NLevelPhysicalParams`：two-level system 是普通 `N=2`，multilevel system 是普通 `N>2`。旧的 solver-ready multilevel 路径已移除，普通示例不再直接构造 code-unit multilevel 输入。
