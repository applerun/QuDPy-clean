# Field 物理电场接口说明

`FieldPhyRoot` 是 QuDPy 用户侧 physical field 的基类。它描述真实 lab-frame
电场，时间单位固定为 `fs`，电场单位固定为 `MV/cm`。当前正式入口是：

```python
field = make_default_gaussian_carrier_field(
    E0_MV_per_cm=0.1,
    laser_energy_eV=1.625,
    pulse_center_fs=0.0,
    pulse_sigma_fs=5.0,
)

params = NLevelPhysicalParams(
    energies_eV=...,
    dipole_matrix_D=...,
    t_start_fs=...,
    t_end_fs=...,
    dt_fs=...,
    field=field,
)
```

`NLevelPhysicalParams` 不再保存 `field_MV_per_cm`、`laser_energy_eV`、
`pulse_center_fs` 或 `pulse_sigma_fs` 这类顶层光场标量。field metadata 由
field 对象自身导出。

## FieldPhyRoot 的要求

自定义 field 至少需要实现：

```python
def physical_E_MV_per_cm(self, t_fs: np.ndarray) -> np.ndarray:
    ...
```

要求：

- 输入 `t_fs` 是 numpy array，单位 `fs`。
- 返回值必须和 `t_fs` shape 相同。
- 返回值是真实 lab-frame 电场，单位 `MV/cm`。
- 不要返回 code-unit field。
- 不要在 field 中构造 Hamiltonian、Lindblad channel 或 density matrix 后处理。

`FieldPhyRoot.__call__(t_fs)` 会调用 `physical_E_MV_per_cm(t_fs)`，并检查 shape。

## reference_MV_per_cm

`reference_MV_per_cm` 是 field 提供给 core 的正式数值接口，单位 `MV/cm`。
它不是 metadata，也不应从 `to_dict()` 中猜测。

`ParaNormalizer` 使用同一个 reference 做两件事：

```text
coupling_matrix_fs_inv = mu_D * reference_MV_per_cm * constant
E_code(t) = E_phys(t) / reference_MV_per_cm
```

这两处必须使用同一个 reference。否则 Hamiltonian 中的 `mu * E(t)` 会被重复缩放
或漏缩放。

默认 `FieldPhyRoot.reference_MV_per_cm` 返回 `None`。能进入 solver 主线的自定义
field 必须覆盖该 property 并返回非零值。

内置 field 约定：

- `CarrierFieldPhysical.reference_MV_per_cm = E0_MV_per_cm`
- `GaussianCarrierFieldPhysical.reference_MV_per_cm = E0_MV_per_cm`
- `FieldPhySeries.reference_MV_per_cm = sum(abs(subfield.reference_MV_per_cm))`

若 `FieldPhySeries` 中任一 subfield 没有 reference，则整个 series 的 reference
为 `None`，normalizer 会 fail-fast。

## normalization_rate_candidates_fs_inv

`normalization_rate_candidates_fs_inv` 是 field 给 `ParaNormalizer` 的 auto-scale
建议，单位 `fs^-1`。它只帮助选择 code-unit 时间尺度，不改变物理模型。

该 property 可以返回多个 candidates，例如：

- Gaussian envelope bandwidth：`1 / sigma_fs`
- envelope modulation angular frequency
- pulse repetition rate
- 其它会影响 `E(t)` 数值变化速度的 field-specific 时间尺度

默认实现返回空 tuple：

```python
()
```

`ParaNormalizer` 只读取这个通用 property，不会根据 Gaussian、CW、TAField、
TwoDESField 或 FieldPhySeries 等具体类型分支。

## 内置 Field

### CarrierFieldPhysical

CW lab-frame carrier：

```text
E(t_fs) = 2 E0 cos(omega_L t_fs + phase)
```

`E0_MV_per_cm` 是表达式中的 `E0`，不是峰峰值。`phase_rad` 是 carrier /
optical phase。

### GaussianCarrierFieldPhysical

Gaussian envelope lab-frame carrier：

```text
E(t_fs) = 2 E0 exp[-(t_fs-center)^2/(2 sigma^2)] cos(omega_L t_fs + phase)
```

`phase_rad` 仍然是 carrier / optical phase。当前类不定义独立 envelope phase。
如果后续需要 complex envelope，应新增专门 field class，而不是把特化语义塞进
real lab-frame Gaussian field。

## FieldPhySeries、TAField 和 TwoDESField

多脉冲组合只在 physical field 层完成：

```python
field = FieldPhySeries(fields=(pump, probe))
field = make_ta_gaussian_field(...)
field = make_twodes_gaussian_field(...)
```

`FieldPhySeries` 本身仍然是 `FieldPhyRoot`，可直接传入
`NLevelPhysicalParams(..., field=field)`。solver/model 层只看到一个 code-unit
callable，不知道也不处理 subfields。

`TAField` 表示 pump-probe / transient absorption 常用的 physical multi-pulse
field。`TwoDESField` 表示 2DES 常用的 three-pulse physical field。它们的 delay
和相位语义属于 field / workflow 层，不属于 solver core。

## 自定义 Field 指南

最小自定义 field 示例：

```python
class MyField(FieldPhyCustomed):
    def __init__(self, amplitude_MV_per_cm, modulation_fs_inv):
        self.amplitude_MV_per_cm = amplitude_MV_per_cm
        self.modulation_fs_inv = modulation_fs_inv

    @property
    def reference_MV_per_cm(self):
        return abs(self.amplitude_MV_per_cm)

    @property
    def normalization_rate_candidates_fs_inv(self):
        return (abs(self.modulation_fs_inv),)

    def physical_E_MV_per_cm(self, t_fs):
        return self.amplitude_MV_per_cm * np.sin(self.modulation_fs_inv * t_fs)

    def __repr__(self):
        return (
            "MyField("
            f"amplitude_MV_per_cm={self.amplitude_MV_per_cm!r}, "
            f"modulation_fs_inv={self.modulation_fs_inv!r})"
        )
```

什么时候需要覆盖 `reference_MV_per_cm`：

- field 要进入 `run_case()` 主线；
- field 有明确的非零参考幅度；
- field 是多个子场的组合，需要定义整体归一化参考幅度。

什么时候需要覆盖 `normalization_rate_candidates_fs_inv`：

- field 包含 pulse width、delay modulation、重复频率或其它快时间尺度；
- 默认 coupling / relaxation / dephasing candidates 不足以代表 field 的变化速度。

`to_dict()` 只用于 metadata、debug metadata 和可选 rebuild 信息。它可以包含
`omega_L_fs_inv`、`laser_energy_eV`、`pulse_center_fs` 等 field-specific metadata，
但 core 数值接口应来自 property，而不是来自 `to_dict()`。

## 设计边界

Field 只负责描述外加物理电场。它不负责：

- Hamiltonian 构造；
- Lindblad channel 构造；
- density matrix 后处理；
- spectroscopy pathway 管理；
- absorption / 2DES 数据分析；
- plotting style。
