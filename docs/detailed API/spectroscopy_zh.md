# Spectroscopy API 说明

本文档说明 `sjh_learn.utils.spectroscopy` 中的 active API。该包只处理已完成模拟的 `rho(t)`、输入场 `E(t)` / drive、polarization 和频域响应；不负责构造 Hamiltonian，也不引入实验数据分析依赖。

当前定位：

```text
DynamicsResult
-> density_array()
-> dipole / polarization observable
-> FFT response
-> absorption / TA intrinsic response
```

这些属于 analysis / example 层，不属于 core solver。

## 推荐导入

```python
from qudpy_sjh.utils.spectroscopy import (
    dipole_expectation_D,
    polarization_C_per_m2,
    apply_time_window,
    safe_complex_ratio,
    lab_frame_fft_response_legacy,
)
```

当前 `sjh_learn.utils.spectroscopy.__init__` 还 re-export 了：

```text
DEBYE_TO_C_M
EPSILON0_F_PER_M
chi_two_level_linear
gamma2_fs_inv_from_T1_Tphi
rwa_fft_response
```

`rwa_fft_response()` 只应作为 legacy RWA-like trajectory 的诊断性后处理，不是 current lab_exact 主线依赖。

## dipole_expectation_D

```python
dipole_expectation_D(rho_t, dipole_matrix_D) -> np.ndarray
```

### 作用

计算单个量子体系的偶极矩期望值，单位 Debye：

```text
p(t) = Tr[rho(t) mu] = sum_ij rho_ij(t) mu_ji
```

实现约定等价于：

```python
P_t = np.einsum("tij,ji->t", rho_t, dipole_matrix_D)
```

### 输入

| 参数 | 含义 |
|---|---|
| `rho_t` | shape `(T, N, N)` 的 density-matrix trajectory |
| `dipole_matrix_D` | shape `(N, N)` 的用户侧物理偶极矩矩阵，单位 Debye |

### 注意

必须使用用户侧物理偶极矩 `dipole_matrix_D`，不能使用 `coupling_matrix_code`。后者已经包含 field reference 和 code-unit scaling，不是 observable 计算所需的物理偶极矩。

## polarization_C_per_m2

```python
polarization_C_per_m2(
    rho_t,
    dipole_matrix_D,
    number_density_m3: float,
) -> np.ndarray
```

### 作用

计算宏观 polarization，单位 `C/m^2`：

```text
P_macro(t) = number_density_m3 * p_single(t) * DEBYE_TO_C_M
```

其中 `number_density_m3` 的单位是 `m^-3`，`DEBYE_TO_C_M` 把 Debye 转为 `C*m`。

### 输入

| 参数 | 含义 |
|---|---|
| `rho_t` | shape `(T, N, N)` 的 density-matrix trajectory |
| `dipole_matrix_D` | shape `(N, N)` 的用户侧物理偶极矩矩阵，单位 Debye |
| `number_density_m3` | number density，单位 `m^-3` |

### 注意

`number_density_m3` 不能为负。若只关心 arbitrary-unit intrinsic response，可以在 example 中固定一个 reference number density，但要在 metadata 中说明。

## apply_time_window

```python
apply_time_window(values: np.ndarray, window: str | None) -> np.ndarray
```

### 作用

在 FFT 前对时间域信号加 window。

当前支持：

```text
None
"none"
"hann"
```

行为：

```python
if window is None or window == "none":
    return values

if window == "hann":
    return values * np.hanning(values.size)
```

其它 window 会报错：

```text
ValueError: window must be None, 'none', or 'hann'.
```

### 重要约束

如果需要 `"hamming"` 或其它 window，应扩展 `apply_time_window()` 这个公共 helper，而不是在 example / scratch 脚本里手写 `np.hamming()`。这样可以避免出现多套 FFT / window 逻辑。

## safe_complex_ratio

```python
safe_complex_ratio(
    numerator: np.ndarray,
    denominator: np.ndarray,
    *,
    rel_threshold: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]
```

### 作用

安全计算 complex ratio：

```text
ratio = numerator / denominator
```

但只在 denominator 足够大时计算。有效点定义为：

```text
abs(denominator) > rel_threshold * max(abs(denominator))
```

返回：

```text
ratio, valid
```

其中 `ratio` 在无效点填入 `nan + 1j * nan`，`valid` 是 bool mask。

### 用途

主要用于避免 `P(omega) / E(omega)` 或 `rho(omega) / E(omega)` 中的 weak-field 小分母问题。

### 注意

如果 denominator spectrum 完全为零，会抛出：

```text
ValueError: The input spectrum is identically zero.
```

## lab_frame_fft_response

```python
lab_frame_fft_response(
    *,
    t_fs: np.ndarray,
    E_MV_per_cm: np.ndarray,
    P_C_per_m2: np.ndarray,
    rhoij: np.ndarray | None = None,
    rho12: np.ndarray | None = None,
    window: str | None = "hann",
    subtract_mean: bool = True,
    rel_threshold: float = 1e-6,
    zero_padding_factor: int = 4,
) -> dict[str, np.ndarray]
```

### 作用

从 lab-frame time-domain signals 计算 pulse response spectrum。它已经负责：

```text
uniform dt 检查
可选 subtract mean
window
zero padding
FFT
frequency axis
omega axis
energy axis
P_fft / E_fft
rhoij_fft / E_fft
positive-frequency selection
small-denominator mask
```

因此 example / scratch 脚本不应重新实现 FFT、frequency axis、positive-frequency selection 或 small-denominator mask。应调用本函数。

### 输入

| 参数 | 含义 |
|---|---|
| `t_fs` | 均匀采样 physical 时间轴，单位 fs |
| `E_MV_per_cm` | lab-frame physical electric field，单位 MV/cm |
| `P_C_per_m2` | polarization，单位 C/m^2 |
| `rhoij` | 可选 density matrix coherence-like signal |
| `rho12` | `rhoij` 的旧兼容命名；如果 `rhoij=None` 且给出 `rho12`，会使用 `rho12` |
| `window` | `None` / `"none"` / `"hann"` |
| `subtract_mean` | FFT 前是否减去均值 |
| `rel_threshold` | `E_fft` 小分母 mask 相对阈值 |
| `zero_padding_factor` | zero padding 倍数，内部会取 2 的幂 FFT size |

如果 `rhoij` 和 `rho12` 都为 `None`，会报错：

```text
ValueError: rhoij is required.
```

`t_fs` 必须是至少两个点的均匀采样时间轴。

### 输出

返回 dict，常用键包括：

```text
omega_fs_inv
energy_eV
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

兼容旧 `rho12` 命名的键包括：

```text
rho12_fft
rho12_over_E
abs_rho12_over_E
im_rho12_over_E
omega_im_rho12_over_E
```

### absorption-like spectrum

当前函数保留的 polarization-based absorption-like 输出键是：

```python
response["neg_omega_im_P_over_E"]
```

其定义是：

```text
- omega * Im[P(omega) / E(omega)]
```

这是为了保留当前数值行为。

如果某个 example 需要使用：

```text
omega * Im[P(omega) / E(omega)]
```

可以在 example 中显式写：

```python
spectrum = response["omega_fs_inv"] * np.imag(response["P_over_E"])
```

但应在 example metadata 中说明符号约定。

### 为什么不同 response 的 energy axis 长度可能不同

`lab_frame_fft_response()` 返回的是经过 mask 后的有效正频率轴：

```text
mask = positive_frequency & valid_E
```

而 `valid_E` 来自 `safe_complex_ratio()` 对 `E_fft` 的小分母筛选。因此，即使原始 FFT frequency grid 一样，不同输入场、不同时间窗位置、不同 window 或不同 `rel_threshold` 都可能导致返回的 `energy_eV` 长度不同。

对 delay-resolved TA map，更稳定的做法是：

```text
固定 probe
所有 delay 共用同一 time grid
probe-only reference 只跑一次
E_probe(omega) 和 valid_E mask 只定义一次
所有 delay 使用同一 energy axis / mask
```

不要让每个 delay 独立决定 mask 后再强行要求 energy axis 一致。

## TA intrinsic response 用法

TA intrinsic response example 可按以下流程组织：

```python
rho_pp = pump_probe_result.density_array()
rho_pr = probe_only_result.density_array()

P_pp = polarization_C_per_m2(
    rho_pp,
    pump_probe_result.physical_params.dipole_matrix_D,
    number_density_m3=1.0e24,
)

P_pr = polarization_C_per_m2(
    rho_pr,
    probe_only_result.physical_params.dipole_matrix_D,
    number_density_m3=1.0e24,
)

E_probe_t = probe_field(t_fs)

response_pp = lab_frame_fft_response(
    t_fs=t_fs,
    E_MV_per_cm=E_probe_t,
    P_C_per_m2=P_pp,
    rhoij=pump_probe_result.matrix_element(0, 1),
)

response_pr = lab_frame_fft_response(
    t_fs=t_fs,
    E_MV_per_cm=E_probe_t,
    P_C_per_m2=P_pr,
    rhoij=probe_only_result.matrix_element(0, 1),
)

S_pp = response_pp["omega_fs_inv"] * np.imag(response_pp["P_over_E"])
S_pr = response_pr["omega_fs_inv"] * np.imag(response_pr["P_over_E"])
S_TA = S_pp - S_pr
```

该定义是材料本征响应，不包含传播效应、样品厚度、transmitted intensity 或 detector response。

## window diagnostic 用法

比较 window 影响时，应该复用同一个 `lab_frame_fft_response()`：

```python
for window in (None, "hann"):
    response = lab_frame_fft_response(
        t_fs=t_fs,
        E_MV_per_cm=E_t,
        P_C_per_m2=P_t,
        rhoij=rhoij_t,
        window=window,
    )
```

如果要加入 `"hamming"`，先扩展 `apply_time_window()`，再让所有脚本调用同一套 helper。
