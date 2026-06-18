# Field 物理电场接口说明

本文档说明当前 `qudpy_sjh.utils.fields` 的 physical field API。field 层负责描述用户侧 lab-frame electric field，是 normalizer 和 solver 的物理输入边界。

## 当前结构

```text
qudpy_sjh/utils/fields/
  __init__.py
  lab_fields.py
  field_series.py
  specific/
    __init__.py
    basic_fields.py
    ta_fields.py
    twodes_fields.py
```

职责边界：

- `lab_fields.py`：基础 lab-frame field 抽象和 wrapper；
- `field_series.py`：多个 physical field 的线性叠加与 scan helper；
- `specific/basic_fields.py`：基础载波场和高斯载波场；
- `specific/ta_fields.py`：TA / pump-probe field helper；
- `specific/twodes_fields.py`：2DES field helper；
- `fields/__init__.py`：用户侧 public API re-export。

## 单位约定

用户侧 field 使用真实物理单位：

```text
time: fs
electric field: MV/cm
laser energy: eV
angular frequency: fs^-1
phase: rad
```

所有 field 应满足：

```python
E = field(t_fs)
```

其中 `E` 的单位是 `MV/cm`。

## FieldPhyRoot

`FieldPhyRoot` 是 lab-frame physical electric field 的基类。

核心接口：

```python
physical_E_MV_per_cm(t_fs)
reference_MV_per_cm
normalization_rate_candidates_fs_inv
to_dict()
time_shifted(shift_fs)
```

含义：

- `physical_E_MV_per_cm(t_fs)` 返回真实 lab-frame 电场，单位 `MV/cm`；
- `reference_MV_per_cm` 是 normalizer 用于构造 code-unit field 的参考电场；
- `normalization_rate_candidates_fs_inv` 为 auto-scale 提供候选速率；
- `to_dict()` 用于 metadata / debug / rebuild；
- `time_shifted(shift_fs)` 返回非原地修改的时间平移 wrapper。

`to_dict()` 不是 core 数值接口。normalizer 不应从 `field.to_dict()` 读取核心缩放参数。

## FieldPhyCustomed

`FieldPhyCustomed` 是用户自定义 physical field 的推荐基类。

自定义 field 至少应实现：

```python
physical_E_MV_per_cm(t_fs)
__repr__()
reference_MV_per_cm
```

如果自定义场包含已知快速时间尺度，建议同时提供：

```python
normalization_rate_candidates_fs_inv
```

否则 normalizer 的 auto-scale 只会根据系统能量、coupling 和 dissipative rates 选择时间尺度。

## 基础场

### CarrierFieldPhysical

`CarrierFieldPhysical` 表示 lab-frame continuous carrier field：

```text
E(t_fs) = 2 E0 cos(omega_L t_fs + phase)
```

主要字段：

```text
E0_MV_per_cm
omega_L_fs_inv
phase_rad
name
metadata
```

推荐 helper：

```python
make_default_carrier_field(
    E0_MV_per_cm=...,
    laser_energy_eV=...,
    phase_rad=0.0,
)
```

该 helper 根据 `laser_energy_eV` 生成 `omega_L_fs_inv`。

### GaussianCarrierFieldPhysical

`GaussianCarrierFieldPhysical` 表示高斯包络载波场：

```text
E(t_fs)
=
2 E0 exp[-(t_fs - center_fs)^2 / (2 sigma_fs^2)]
cos(omega_L t_fs + phase)
```

主要字段：

```text
E0_MV_per_cm
omega_L_fs_inv
center_fs
sigma_fs
phase_rad
name
metadata
```

推荐 helper：

```python
make_default_gaussian_carrier_field(
    E0_MV_per_cm=...,
    laser_energy_eV=...,
    pulse_center_fs=...,
    pulse_sigma_fs=...,
    phase_rad=0.0,
)
```

## TimeShiftedField

`TimeShiftedField` 是 non-mutating time-shift wrapper。

推荐调用：

```python
shifted = field.time_shifted(shift_fs)
```

约定：

```text
shift_fs > 0 表示场整体向更晚时间移动
E_shifted(t) = E_original(t - shift_fs)
```

例子：

```python
pulse_0 = make_default_gaussian_carrier_field(
    E0_MV_per_cm=0.05,
    laser_energy_eV=1.5,
    pulse_center_fs=0.0,
    pulse_sigma_fs=10.0,
)

pulse_50 = pulse_0.time_shifted(50.0)
```

此时 `pulse_50` 的中心移动到 `50 fs`。

连续平移会合并：

```python
field.time_shifted(20.0).time_shifted(30.0)
```

等价于：

```python
field.time_shifted(50.0)
```

## FieldPhySeries

`FieldPhySeries` 表示多个 physical field 的线性叠加：

```text
E_total(t_fs) = sum_k E_k(t_fs)
```

示例：

```python
from qudpy_sjh.utils.fields import FieldPhySeries

field = FieldPhySeries(
    fields=(pump, probe),
    sub_field_names=("pump", "probe"),
)

E_total = field(t_fs)
E_probe = field["probe"](t_fs)
```

`FieldPhySeries` 本身仍是 `FieldPhyRoot`，可直接传入：

```python
NLevelPhysicalParams(..., field=field)
```

solver 不需要知道内部有几个 subfield。多场叠加只发生在 physical field 层。

## TA / pump-probe helper

主要对象：

```text
TAField
```

主要 helper：

```text
make_ta_gaussian_field
make_pump_probe_field_from_templates
make_ta_field_from_templates
iter_ta_gaussian_fields
```

### TAField

`TAField` 是 `FieldPhySeries` 的子类，用于 pump-probe / transient absorption 场。它通常包含：

```text
sub_field_names = ("pump", "probe")
```

可通过：

```python
pump = field["pump"]
probe = field["probe"]
```

提取子场。

### make_ta_gaussian_field

`make_ta_gaussian_field(...)` 用于直接构造 Gaussian pump 和 Gaussian probe。

当前语义是：

```text
probe_center_fs = pump_center_fs + probe_delay_fs
```

因此它默认固定 pump center，然后根据 delay 移动 probe。

### make_pump_probe_field_from_templates

`make_pump_probe_field_from_templates(...)` 用 zero-centered templates 构造 pump-probe field。

当前 pump-probe convention 是：

```text
probe_center_fs 默认 0
pump_center_fs = probe_center_fs - delay_fs
```

因此它默认固定 probe，然后根据 delay 移动 pump。

示例：

```python
field = make_pump_probe_field_from_templates(
    pump_template=pump_template,
    probe_template=probe_template,
    delay_fs=100.0,
    probe_center_fs=0.0,
)
```

含义：

```text
probe center = 0 fs
pump center = -100 fs
```

### make_ta_field_from_templates

`make_ta_field_from_templates(...)` 是 template-based helper 的兼容入口。可以传：

```text
probe_delay_fs
```

或：

```text
delay_fs
```

二者都传时必须一致。

## 2DES helper

主要对象：

```text
TwoDESField
```

主要 helper：

```text
make_twodes_gaussian_field
iter_twodes_gaussian_fields
```

`TwoDESField` 是三脉冲 field series：

```text
sub_field_names = ("pump1", "pump2", "probe")
```

当前 Gaussian helper 的中心时间约定：

```text
pump2_center_fs = pump1_center_fs + pump_tau_fs
probe_center_fs = pump2_center_fs + probe_delay_fs
```

## 推荐导入路径

推荐从 public API 导入：

```python
from qudpy_sjh.utils.fields import (
    FieldPhyRoot,
    FieldPhyCustomed,
    TimeShiftedField,
    FieldPhySeries,
    CarrierFieldPhysical,
    GaussianCarrierFieldPhysical,
    TAField,
    TwoDESField,
    make_default_carrier_field,
    make_default_gaussian_carrier_field,
    make_pump_probe_field_from_templates,
    make_ta_field_from_templates,
    make_ta_gaussian_field,
    make_twodes_gaussian_field,
)
```

普通用户不应直接构造 `_CodeFieldAdapter`。它是 `ParaNormalizer` 和 solver 内部使用的 code-unit adapter。
