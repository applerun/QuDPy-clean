# ParaNormalizer 说明

本文档说明当前 `qudpy_sjh.utils.core.normalization.ParaNormalizer` 的职责和边界。

`ParaNormalizer` 只负责真实物理单位到 solver code unit 的转换。它不定义物理模型，不构造 Hamiltonian，也不生成 collapse operator。

## 当前位置

```text
qudpy_sjh/utils/core/normalization.py
```

推荐导入：

```python
from qudpy_sjh.utils.core import ParaNormalizer
```

## 主职责

`ParaNormalizer` 的主流程是：

```text
NLevelPhysicalParams
-> validate physical units and shapes
-> derive fs^-1 quantities
-> choose time_scale_fs
-> build SolverParams
-> make one _CodeFieldAdapter
```

它接收用户侧 `NLevelPhysicalParams`，输出 solver 可用的 `SolverParams`。

## 单位转换

用户侧单位：

```text
energy: eV
dipole: Debye
time: fs
field: MV/cm
rate: fs^-1 或由时间常数得到
```

中间物理单位：

```text
energy -> fs^-1
coupling -> fs^-1
relaxation / dephasing rate -> fs^-1
```

solver 内部：

```text
time -> code unit
frequency / rate -> code unit
field -> dimensionless code signal
```

## field 缩放

当前 field 缩放的核心关系是：

```text
coupling_matrix_fs_inv = dipole_matrix_D * reference_MV_per_cm * constant
E_code(t) = E_phys(t) / reference_MV_per_cm
```

其中 `reference_MV_per_cm` 来自：

```python
field.reference_MV_per_cm
```

不要从：

```python
field.to_dict()
```

读取 core 数值参数。

## reference_MV_per_cm

`field.reference_MV_per_cm` 是 normalizer 的正式数值接口。

如果 `reference_MV_per_cm` 是 `None` 或 0，normalizer 应直接报错，而不是默默构造无意义的 code-unit field。

对 `FieldPhySeries`，当前 reference 通常为各子场 reference 绝对值之和。

## normalization_rate_candidates_fs_inv

`field.normalization_rate_candidates_fs_inv` 为 auto-scale 提供 field 相关候选速率。

例如 Gaussian pulse 可以提供：

```text
1 / sigma_fs
```

`FieldPhySeries` 会汇总所有 subfield 的候选速率。

normalizer 不应根据具体 field class 写分支，例如：

```text
不要写 Gaussian 特例
不要写 TAField 特例
不要写 TwoDESField 特例
```

如果 field 需要提供时间尺度提示，应由 field 自身通过统一接口暴露。

## time_scale_fs

`ParaNormalizer` 可由用户显式指定：

```python
ParaNormalizer(time_scale_fs=..., auto_scale=True)
```

若未指定且 `auto_scale=True`，normalizer 会根据候选速率选择 code-unit time scale。

若 `auto_scale=False`，通常使用 `time_scale_fs = 1.0`。

## normalize()

主入口：

```python
solver = normalizer.normalize(physical_params)
```

`normalize()` 会：

1. 校验 `NLevelPhysicalParams`；
2. 转换 `energies_eV` 到 `energies_fs_inv`；
3. 从 `field.reference_MV_per_cm` 读取参考场强；
4. 生成 `coupling_matrix_fs_inv`；
5. 将 relaxation / dephasing channel 转为 rate dict；
6. 选择 `time_scale_fs`；
7. 构造 code-unit `tlist`；
8. 返回 `SolverParams`。

## make_code_field()

`make_code_field(...)` 把用户侧 physical field 转换成 solver 内部 callable。

调用形式：

```python
code_field = normalizer.make_code_field(field_phy, solver)
```

内部完成：

```text
t_code -> t_fs
E_MV_per_cm -> E_code
```

生成的对象是 `_CodeFieldAdapter`。它不是用户侧 API。

## _CodeFieldAdapter

`_CodeFieldAdapter` 的作用是把 solver 看到的 code time 和 code field 连接到真实 physical field。

概念关系：

```text
solver 调用 field(t_code)
adapter 将 t_code 转回 t_fs
adapter 调用 field_phy(t_fs)
adapter 将 E_MV_per_cm 除以 reference_MV_per_cm
返回 dimensionless E_code
```

它还提供：

```python
physical(t_fs)
```

用于在 result / IO 中复用 solver 实际使用的 physical field，避免展示层重新拼写场函数。

## summary_dict()

`summary_dict(...)` 用于生成 physical / solver 参数摘要，适合 debug metadata。

该输出可以包含 code-unit 数组和内部参数，不应作为面向普通用户的唯一 metadata。

普通人类可读 metadata 应由 IO 层生成 `meta.json`。

## 不应承担的职责

`ParaNormalizer` 不应：

```text
构造 Hamiltonian
构造 collapse operators
调用 qutip.mesolve
决定 parameter scan
决定 delay scan
保存文件
绘图
计算 FFT response
计算 absorption response
根据具体 field class 写 workflow 特例
```

这些职责分别属于 core model、solver、workflow、IO、plotting 和 spectroscopy。

## 示例

```python
from qudpy_sjh.utils.core import ParaNormalizer

normalizer = ParaNormalizer()
solver_params = normalizer.normalize(params)
code_field = normalizer.make_code_field(params.field, solver_params)
```

普通用户通常不需要手动调用 `make_code_field()`，因为 `run_case(params)` 会在内部完成该步骤。
