# Field 物理电场接口说明

本文档说明当前 `qudpy_sjh.utils.fields` 的 physical field API。field 层负责描述用户侧 lab-frame electric field，是 normalizer 和 solver 的物理输入边界。

## 当前结构

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

职责边界：

- `lab_fields.py`：基础 lab-frame field 抽象、time-shift wrapper 和 code-unit adapter；
- `field_series.py`：多个 physical field 的线性叠加与 scan helper；
- `carrier_envelope/`：显式 carrier + envelope 语义的有限脉冲 field；
- `fields/__init__.py`：基础 field public API re-export。

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

其中 `E` 的单位是 `MV/cm`，返回数组 shape 必须与 `t_fs` 一致。

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
pulse_0 = make_gaussian_carrier_envelope_field(
    E0_MV_per_cm=0.05,
    laser_energy_eV=1.5,
    center_fs=0.0,
    sigma_fs=10.0,
)

pulse_50 = pulse_0.time_shifted(50.0)
```

在普通 `TimeShiftedField` wrapper 中，这等价于 `pulse_50(t)=pulse_0(t-50 fs)`。在 `CarrierEnvelopeField.time_shifted(...)` 中，当前实现采用 envelope center shift：`center_new = center_old + shift_fs`，carrier phase 仍相对于新的 envelope center 定义。

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

solver 不需要知道内部有几个 subfield。多场叠加只发生在 physical field 层。`sub_field_names` 可用于表达 pump / probe / LO 等 workflow role。

## carrier_envelope 子包

推荐从子包导入：

```python
from qudpy_sjh.utils.fields.carrier_envelope import (
    CarrierSpec,
    EnvelopeSpec,
    GaussianEnvelopeSpec,
    SechEnvelopeSpec,
    ConstantEnvelopeSpec,
    CarrierEnvelopeField,
    make_gaussian_carrier_envelope_field,
    make_pump_probe_field_series,
)
```

当前核心对象：

```text
CarrierSpec
EnvelopeSpec
GaussianEnvelopeSpec
SechEnvelopeSpec
ConstantEnvelopeSpec
CarrierEnvelopeField
```

当前推荐 helper：

```text
make_gaussian_carrier_envelope_field
make_pump_probe_field_series
```

`carrier_envelope.__init__` 还导出若干通用 builder；文档主线优先使用直接构造 `CarrierEnvelopeField` 或 `make_gaussian_carrier_envelope_field(...)`。

## CarrierSpec

`CarrierSpec` 表示单个准单色 carrier。

核心字段：

```text
omega_fs_inv
phase_rad
label
metadata
```

也可由能量构造：

```python
carrier = CarrierSpec.from_energy_eV(
    laser_energy_eV=1.55,
    phase_rad=0.0,
)
```

phase 约定：

```text
carrier phase = omega_fs_inv * (t_fs - center_fs) + phase_rad
```

因此 `phase_rad` 是相对于 envelope center 定义的 carrier phase，不是全局 lab-frame `omega*t + phase`。

## EnvelopeSpec

`EnvelopeSpec` 是 dimensionless envelope 的抽象基类。具体 envelope 应提供：

```python
value(t_fs)
shifted(shift_fs)
to_dict()
```

当前主要 concrete spec：

```text
GaussianEnvelopeSpec
SechEnvelopeSpec
ConstantEnvelopeSpec
```

`GaussianEnvelopeSpec`：

```text
envelope(t) = amplitude * exp[-(t-center)^2/(2*sigma^2)]
```

`SechEnvelopeSpec`：

```text
envelope(t) = amplitude / cosh[(t-center)/width]
```

`ConstantEnvelopeSpec`：

```text
envelope(t) = amplitude
```

## CarrierEnvelopeField

`CarrierEnvelopeField` 是当前推荐的 finite optical pulse field。它组合一个 `CarrierSpec` 和一个 `EnvelopeSpec`。

物理约定：

```text
E(t) = 2 E0 envelope(t) cos[omega * (t - center) + phase]
```

其中：

```text
E0 = E0_MV_per_cm
envelope(t) = self.envelope.value(t)
center = self.envelope.center_fs
omega = self.carrier.omega_fs_inv
phase = self.carrier.phase_rad
```

注意：

1. `peak_E_MV_per_cm` 在 metadata 中是 `2*E0_MV_per_cm`；
2. carrier phase 相对于 envelope center 定义；
3. `CarrierEnvelopeField` 不包含 pump / probe / LO role；
4. role 应由 `FieldPhySeries.sub_field_names`、case metadata 或 workflow 层表达。

直接构造示例：

```python
from qudpy_sjh.utils.fields.carrier_envelope import (
    CarrierEnvelopeField,
    CarrierSpec,
    GaussianEnvelopeSpec,
)

field = CarrierEnvelopeField(
    E0_MV_per_cm=0.05,
    carrier=CarrierSpec.from_energy_eV(1.50, phase_rad=0.0),
    envelope=GaussianEnvelopeSpec(center_fs=0.0, sigma_fs=10.0),
    name="probe",
)
```

helper 构造示例：

```python
from qudpy_sjh.utils.fields.carrier_envelope import (
    make_gaussian_carrier_envelope_field,
)

field = make_gaussian_carrier_envelope_field(
    E0_MV_per_cm=0.05,
    laser_energy_eV=1.50,
    center_fs=0.0,
    sigma_fs=10.0,
    phase_rad=0.0,
    name="probe",
)
```

## pump-probe 组合

当前不再推荐把 TA role 写成单独 field subclass。推荐方式是：

```python
from qudpy_sjh.utils.fields import FieldPhySeries
from qudpy_sjh.utils.fields.carrier_envelope import (
    make_gaussian_carrier_envelope_field,
)

probe = make_gaussian_carrier_envelope_field(
    E0_MV_per_cm=0.008,
    laser_energy_eV=1.62,
    center_fs=0.0,
    sigma_fs=7.0,
    phase_rad=0.0,
    name="probe",
)

pump = make_gaussian_carrier_envelope_field(
    E0_MV_per_cm=0.30,
    laser_energy_eV=1.55,
    center_fs=-100.0,
    sigma_fs=12.0,
    phase_rad=0.0,
    name="pump",
)

field = FieldPhySeries(
    fields=(pump, probe),
    sub_field_names=("pump", "probe"),
    name="pump_probe",
)
```

等价地，可用便利 helper：

```python
from qudpy_sjh.utils.fields.carrier_envelope import make_pump_probe_field_series

field = make_pump_probe_field_series(
    pump_field=pump,
    probe_field=probe,
)
```

delay convention 建议放在 workflow 中显式写清楚。例如当前 TA demo 使用：

```text
probe_center_fs 固定
pump_center_fs = probe_center_fs - delay_fs
positive delay means pump before probe
```

## 推荐导入路径

基础 field API：

```python
from qudpy_sjh.utils.fields import (
    FieldPhyRoot,
    FieldPhyCustomed,
    TimeShiftedField,
    FieldPhySeries,
    iter_scan_params,
)
```

carrier-envelope API：

```python
from qudpy_sjh.utils.fields.carrier_envelope import (
    CarrierSpec,
    EnvelopeSpec,
    GaussianEnvelopeSpec,
    SechEnvelopeSpec,
    ConstantEnvelopeSpec,
    CarrierEnvelopeField,
    make_gaussian_carrier_envelope_field,
    make_pump_probe_field_series,
)
```

普通用户不应直接构造 `_CodeFieldAdapter`。它是 `ParaNormalizer` 和 solver 内部使用的 code-unit adapter。

## 不再作为当前主线宣传的旧接口

以下旧接口若在旧文档或旧脚本中出现，应视为历史残留或 archived code，不应作为当前推荐 public API：

```text
specific/basic_fields.py
specific/ta_fields.py
specific/twodes_fields.py
CarrierFieldPhysical
GaussianCarrierFieldPhysical
TAField
TwoDESField
make_default_carrier_field
make_default_gaussian_carrier_field
make_pump_probe_field_from_templates
make_ta_field_from_templates
make_ta_gaussian_field
make_twodes_gaussian_field
```
