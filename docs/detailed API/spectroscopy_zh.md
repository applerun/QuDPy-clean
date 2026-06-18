# Spectroscopy 与 Absorption Response 说明

本文档说明当前 `qudpy_sjh.utils.spectroscopy` 的谱学后处理接口。

spectroscopy 层只处理已经完成的 dynamics 结果，不构造 Hamiltonian，不调用 solver。

## 当前位置

```text
qudpy_sjh/utils/spectroscopy/
  __init__.py
  observables.py
  absorption_spectra.py
  theory.py
```

推荐导入：

```python
from qudpy_sjh.utils.spectroscopy import (
    dipole_expectation_D,
    polarization_C_per_m2,
    diagnose_uniform_time_axis,
    apply_time_window,
    lab_frame_absorption_response,
    lab_frame_fft_response_legacy,
)
```

## 职责边界

spectroscopy 层处理：

```text
rho(t)
dipole expectation
polarization
E(t)
FFT response
absorption response
```

spectroscopy 层不处理：

```text
Hamiltonian 构造
collapse operator 构造
mesolve 调用
parameter scan
case 输出目录管理
```

## observables

### dipole_expectation_D

```python
dipole_expectation_D(rho_t, dipole_matrix_D)
```

计算单个量子体系的偶极矩期望值，单位 Debye：

```text
p(t) = Tr[rho(t) mu] = sum_ij rho_ij(t) mu_ji
```

实现约定等价于：

```python
np.einsum("tij,ji->t", rho_t, dipole_matrix_D)
```

注意：这里必须使用用户侧物理偶极矩 `dipole_matrix_D`，不能使用已经包含光场强度和 code-unit 归一化的 `coupling_matrix_code`。

### polarization_C_per_m2

```python
polarization_C_per_m2(rho_t, dipole_matrix_D, number_density_m3)
```

计算宏观 polarization，单位 `C/m^2`：

```text
P(t) = number_density_m3 * p_single(t) * DEBYE_TO_C_M
```

`number_density_m3` 必须显式传入，单位 `m^-3`。

## 推荐主入口：lab_frame_absorption_response

当前推荐的 absorption-only 主入口是：

```python
lab_frame_absorption_response(...)
```

调用形式：

```python
response = lab_frame_absorption_response(
    time_fs=time_fs,
    polarization_C_per_m2=P,
    field=E,
    window="hann",
    subtract_mean=True,
)
```

主输出字段：

```text
energy_eV
omega_fs_inv
absorption
omega_im_P_over_E
```

定义：

```text
absorption = omega * Im[P(omega) / E(omega)]
omega_im_P_over_E = absorption
```

可选兼容项：

```python
include_legacy_alias=True
```

会额外返回：

```text
neg_omega_im_P_over_E
```

该字段只用于旧符号约定兼容，新代码应优先使用 `absorption`。

## 参数别名

推荐使用新参数名：

```text
time_fs
polarization_C_per_m2
field
```

兼容旧参数名：

```text
t_fs
P_C_per_m2
E_MV_per_cm
```

新文档和新脚本应优先使用推荐参数名。

## time axis diagnostics

```python
diagnose_uniform_time_axis(time_fs)
```

返回：

```text
n
t0_fs
t1_fs
min_dt_fs
max_dt_fs
median_dt_fs
is_uniform
```

`lab_frame_absorption_response(...)` 要求输入时间轴 uniform sampled。非均匀时间轴会直接报错。

## window

当前公共 helper：

```python
apply_time_window(values, window)
```

支持：

```text
None
"none"
"hann"
```

若需要 `"hamming"` 或其它 window，应先扩展公共 helper，不应在单个脚本里手写第二套 window 逻辑。

## mask 与 safe complex ratio

`spectroscopy` 使用：

```python
safe_complex_ratio(numerator, denominator, rel_threshold=...)
```

避免在 `E(omega)` 很小的位置做不稳定除法。

有效 mask 近似为：

```text
abs(E_omega) > rel_threshold * max(abs(E_omega))
```

最终返回的是：

```text
positive-frequency mask
AND
valid_E mask
```

筛选后的频率轴。

因此不同输入 field、window、zero padding 或 `rel_threshold` 可能导致返回的 `energy_eV` 长度不同。

## lab_frame_fft_response_legacy

当前保留较底层 / 历史兼容 helper：

```python
lab_frame_fft_response_legacy(...)
```

它返回更多中间量：

```text
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

其中：

```text
neg_omega_im_P_over_E = -omega * Im[P(omega) / E(omega)]
```

它不应作为当前 absorption-only 主入口宣传。

当前未在代码中确认存在精确名称：

```python
lab_frame_fft_response(...)
```

旧文档如使用该名称，应更新为 `lab_frame_fft_response_legacy(...)` 或改用 `lab_frame_absorption_response(...)`。

## 示例

```python
from qudpy_sjh.utils.spectroscopy import (
    polarization_C_per_m2,
    lab_frame_absorption_response,
)

rho_t = result.density_array()
time_fs = result.times_fs
E = result.field_MV_per_cm_values()

P = polarization_C_per_m2(
    rho_t,
    result.physical_params.dipole_matrix_D,
    number_density_m3=1.0e24,
)

response = lab_frame_absorption_response(
    time_fs=time_fs,
    polarization_C_per_m2=P,
    field=E,
    window="hann",
    subtract_mean=True,
)

energy_eV = response["energy_eV"]
absorption = response["absorption"]
```

## 与 IO 的边界

`spectroscopy` 只返回数组和 metadata dict，不负责 case 输出目录管理。

如果需要保存响应谱，建议在 workflow 层另存：

```text
absorption_response.csv
fft_response.csv
analysis_components.csv
```

不要把 absorption response 混入普通 `DynamicsResult.components_dataframe()`。
