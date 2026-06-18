# Parameters API 说明

本文档说明当前 `qudpy_sjh.utils.core.parameters` 的 active API。该模块只定义数据容器，不做单位换算、不构造 Hamiltonian，也不调用 QuTiP。

当前主线是：

```text
FieldPhyRoot / FieldPhySeries / TAField / TwoDESField
-> NLevelPhysicalParams
-> ParaNormalizer
-> NLevelSolverParams
-> run_case()
-> DynamicsResult
```

## 推荐导入

推荐从 `qudpy_sjh.utils.core` 导入：

```python
from qudpy_sjh.utils.core import (
    NLevelPhysicalParams,
    RelaxationChannel,
    PureDephasingChannel,
    SolverParams,
    NLevelSolverParams,
)
```

## NLevelPhysicalParams

`NLevelPhysicalParams` 是用户侧 N-level 物理系统输入。

它描述：

```text
能级能量
偶极矩矩阵
时间网格
外加 physical field
population relaxation
pure dephasing
solver mode
用户备注 metadata
```

正式入口是：

```python
NLevelPhysicalParams(..., field=field)
```

`field` 必须是 `FieldPhyRoot` 或其子类 / wrapper。

核心字段：

```python
NLevelPhysicalParams(
    energies_eV=(...),
    dipole_matrix_D=((...), ...),
    t_start_fs=...,
    t_end_fs=...,
    dt_fs=...,
    field=field,
    basis=(...) or None,
    relaxation_channels=(...),
    pure_dephasing_channels=(...),
    solver_mode="lab_exact",
    input_description=...,
    input_metadata={...},
)
```

单位：

| 字段 | 单位 / 类型 |
|---|---|
| `energies_eV` | eV |
| `dipole_matrix_D` | Debye |
| `t_start_fs`, `t_end_fs`, `dt_fs` | fs |
| `field` | `FieldPhyRoot` |
| `basis` | optional state labels |
| `relaxation_channels` | tuple of `RelaxationChannel` |
| `pure_dephasing_channels` | tuple of `PureDephasingChannel` |
| `solver_mode` | 当前主线为 `"lab_exact"` |

## two-level 和 multilevel

two-level system 只是普通 `N=2` system。核心层不再把二能级作为特殊标量模型处理。

multilevel system 是普通 `N>2` system。

`dimension` 返回能级数量 `N`。

`energy_gap_eV` 是 0->1 兼容属性；对 `N>2`，它只表示第 0 和第 1 个能级之间的能量差，不代表体系唯一能隙。

## dipole_matrix_D

`dipole_matrix_D` 表示沿选定 optical polarization 投影后的跃迁偶极矩矩阵，单位 Debye。

约束：

```text
shape = (N, N)
应与 energies_eV 的长度一致
应为 Hermitian
```

允许 complex transition dipole，但必须满足：

```text
mu_ij = conj(mu_ji)
```

## RelaxationChannel

`RelaxationChannel` 描述 population relaxation：

```text
C_{to <- from} = sqrt(rate) |to><from|
```

字段：

```python
RelaxationChannel(
    name="relaxation_1_to_0",
    from_level=1,
    to_level=0,
    T1_fs=1000.0,
)
```

或：

```python
RelaxationChannel(
    name="relaxation_1_to_0",
    from_level=1,
    to_level=0,
    rate_fs_inv=1.0e-3,
)
```

`T1_fs` 和 `rate_fs_inv` 都可指定。二者都给出时，normalizer 优先使用 `rate_fs_inv`。

错误写法：

```python
relaxation_channels=(
    {"from_level": 1, "to_level": 0, "T1_fs": 1000.0},
)
```

dict 形式适合 JSON metadata，不是 `NLevelPhysicalParams` 的 active input。

## PureDephasingChannel

`PureDephasingChannel` 描述 level projector pure dephasing：

```text
C_level^phi = sqrt(rate) |level><level|
```

字段：

```python
PureDephasingChannel(
    name="pure_dephasing_level_1",
    level=1,
    Tphi_fs=120.0,
)
```

或：

```python
PureDephasingChannel(
    name="pure_dephasing_level_1",
    level=1,
    rate_fs_inv=1.0 / 120.0,
)
```

`Tphi_fs` 和 `rate_fs_inv` 都可指定。二者都给出时，normalizer 优先使用 `rate_fs_inv`。

## SolverParams

`SolverParams` 是 `ParaNormalizer.normalize()` 输出的 solver 参数摘要。

它包含：

```text
time_scale_fs
energies_fs_inv
energies_code
dipole_matrix_D
coupling_matrix_fs_inv
coupling_matrix_code
relaxation_channels_fs_inv
pure_dephasing_channels_fs_inv
relaxation_channels_code
pure_dephasing_channels_code
t_start
t_end
dt
tlist
```

`SolverParams` 同时保留 fs^-1 和 code-unit 信息，主要用于 solver 构造、debug metadata 和 sanity check。

普通用户不应手动构造 `SolverParams`。

## NLevelSolverParams

`NLevelSolverParams` 是 solver 内部 code-unit 参数容器。普通 example 应构造 `NLevelPhysicalParams`，再经 `ParaNormalizer` 转换，不应直接构造 `NLevelSolverParams` 作为用户入口。

它包含：

```text
code-unit energies
code-unit coupling matrix
code-unit relaxation / dephasing channels
code-unit time grid
internal field callable
basis metadata
```

## solver_mode

当前主线是：

```text
solver_mode="lab_exact"
```

`solver_mode="rwa"` 当前应视为 legacy guard。新示例和新文档不应依赖 RWA solver-unit drive path。

## 最小示例

```python
from qudpy_sjh.utils.fields import make_default_gaussian_carrier_field
from qudpy_sjh.utils.core import (
    NLevelPhysicalParams,
    RelaxationChannel,
    PureDephasingChannel,
)

field = make_default_gaussian_carrier_field(
    E0_MV_per_cm=0.05,
    laser_energy_eV=1.5,
    pulse_center_fs=0.0,
    pulse_sigma_fs=10.0,
)

params = NLevelPhysicalParams(
    energies_eV=(0.0, 1.5),
    dipole_matrix_D=((0.0, 1.0), (1.0, 0.0)),
    t_start_fs=-100.0,
    t_end_fs=200.0,
    dt_fs=0.2,
    field=field,
    relaxation_channels=(
        RelaxationChannel(
            name="relaxation_1_to_0",
            from_level=1,
            to_level=0,
            T1_fs=1000.0,
        ),
    ),
    pure_dephasing_channels=(
        PureDephasingChannel(
            name="pure_dephasing_level_1",
            level=1,
            Tphi_fs=200.0,
        ),
    ),
)
```
