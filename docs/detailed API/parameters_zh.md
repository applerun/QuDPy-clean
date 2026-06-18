# Parameters API 说明

本文档说明 `sjh_learn.utils.core.parameters` 中的 active API。该模块只定义数据容器，不做单位换算、不构造 Hamiltonian，也不调用 QuTiP。

当前主线是：

```text
FieldPhyRoot / FieldPhySeries / TAField
-> NLevelPhysicalParams
-> ParaNormalizer
-> NLevelSolverParams
-> run_case()
-> DynamicsResult
```

## 公开入口

推荐从 `sjh_learn.utils.core` 导入：

```python
from qudpy_sjh.utils.core import (
    NLevelPhysicalParams,
    RelaxationChannel,
    PureDephasingChannel,
    SolverParams,
    NLevelSolverParams,
)
```

`RelaxationChannel` 和 `PureDephasingChannel` 是 dataclass 对象，不是 dict。不能把 metadata JSON 中的 channel dict 直接传给 `NLevelPhysicalParams`。

## NLevelPhysicalParams

`NLevelPhysicalParams` 是当前用户侧标准物理系统输入。two-level system 只是普通 `N=2` system，multilevel system 是普通 `N>2` system。

### 字段

```python
NLevelPhysicalParams(
    energies_eV: tuple[float, ...],
    dipole_matrix_D: tuple[tuple[complex, ...], ...],
    t_start_fs: float,
    t_end_fs: float,
    dt_fs: float,
    field: FieldPhyRoot,
    basis: tuple[str, ...] | None = None,
    relaxation_channels: tuple[RelaxationChannel, ...] = (),
    pure_dephasing_channels: tuple[PureDephasingChannel, ...] = (),
    solver_mode: str = "lab_exact",
    input_description: str | None = None,
    input_metadata: dict[str, Any] | None = None,
)
```

### 单位约定

| 字段 | 含义 | 单位 |
|---|---|---|
| `energies_eV` | N 个能级能量 | eV |
| `dipole_matrix_D` | 沿 selected optical polarization 投影后的 N x N 偶极矩矩阵 | Debye |
| `t_start_fs` | 模拟起始时间 | fs |
| `t_end_fs` | 模拟结束时间 | fs |
| `dt_fs` | 输出时间步长 | fs |
| `field` | 唯一输入 physical field | field 自身约定为 fs / MV/cm |
| `basis` | basis state 名称 | dimensionless |
| `relaxation_channels` | population relaxation channel 列表 | fs 或 fs^-1 |
| `pure_dephasing_channels` | pure dephasing channel 列表 | fs 或 fs^-1 |
| `solver_mode` | solver 模式 | 当前主线为 `"lab_exact"` |
| `input_description` | 用户描述 | metadata-only |
| `input_metadata` | 用户备注 | metadata-only |

### field 边界

`NLevelPhysicalParams` 不再保存这些旧 top-level optical scalar：

```text
field_MV_per_cm
laser_energy_eV
pulse_center_fs
pulse_sigma_fs
```

这些信息属于 field 对象本身，例如 `GaussianCarrierFieldPhysical` 或 `TAField`。用户侧应显式构造 field，然后传入：

```python
field = make_default_gaussian_carrier_field(...)
params = NLevelPhysicalParams(
    energies_eV=(0.0, 1.55),
    dipole_matrix_D=((0.0, 5.0), (5.0, 0.0)),
    t_start_fs=-200.0,
    t_end_fs=500.0,
    dt_fs=0.2,
    field=field,
)
```

### basis

`basis` 可以为 `None`。如果为 `None`，metadata / 展示层可默认使用 `"0"`, `"1"`, ..., `"N-1"`。如果给出，应与 `energies_eV` 长度一致。

### properties

```python
params.dimension
```

返回能级数量 N，即 `len(params.energies_eV)`。

```python
params.energy_gap_eV
```

返回 `energies_eV[1] - energies_eV[0]`。这是 N=2 教学示例中常用的兼容属性。对于 N>2，它只表示第 0 和第 1 个能级之间的能量差，不代表体系唯一能隙。

## RelaxationChannel

`RelaxationChannel` 描述 population relaxation：

```text
C_{to <- from} = sqrt(rate) |to><from|
```

### 构造

```python
RelaxationChannel(
    name: str,
    from_level: int,
    to_level: int,
    T1_fs: float | None = None,
    rate_fs_inv: float | None = None,
)
```

### 字段

| 字段 | 含义 | 单位 |
|---|---|---|
| `name` | channel 名称 | metadata/debug |
| `from_level` | population 来源能级 | zero-based index |
| `to_level` | population 去向能级 | zero-based index |
| `T1_fs` | relaxation 时间常数 | fs |
| `rate_fs_inv` | relaxation 速率 | fs^-1 |

可以用 `T1_fs` 或 `rate_fs_inv` 指定速率。二者都给出时，normalizer 优先使用 `rate_fs_inv`。

### 示例

```python
relaxation_channels=(
    RelaxationChannel(
        name="relaxation_1_to_0",
        from_level=1,
        to_level=0,
        T1_fs=1000.0,
    ),
)
```

或：

```python
relaxation_channels=(
    RelaxationChannel(
        name="relaxation_1_to_0",
        from_level=1,
        to_level=0,
        rate_fs_inv=1.0e-3,
    ),
)
```

### 常见错误

错误写法：

```python
relaxation_channels=(
    {"from_level": 1, "to_level": 0, "T1_fs": 1000.0},
)
```

原因：core API 需要 `RelaxationChannel` 对象。dict 形式适合 JSON metadata，不是 `NLevelPhysicalParams` 的 active input。

## PureDephasingChannel

`PureDephasingChannel` 描述 level projector pure dephasing：

```text
C_level^phi = sqrt(rate) |level><level|
```

### 构造

```python
PureDephasingChannel(
    name: str,
    level: int,
    Tphi_fs: float | None = None,
    rate_fs_inv: float | None = None,
)
```

### 字段

| 字段 | 含义 | 单位 |
|---|---|---|
| `name` | channel 名称 | metadata/debug |
| `level` | dephasing 作用能级 | zero-based index |
| `Tphi_fs` | pure dephasing 时间常数 | fs |
| `rate_fs_inv` | pure dephasing 速率 | fs^-1 |

可以用 `Tphi_fs` 或 `rate_fs_inv` 指定速率。二者都给出时，normalizer 优先使用 `rate_fs_inv`。

### 示例

```python
pure_dephasing_channels=(
    PureDephasingChannel(
        name="pure_dephasing_level_1",
        level=1,
        Tphi_fs=120.0,
    ),
)
```

或：

```python
pure_dephasing_channels=(
    PureDephasingChannel(
        name="pure_dephasing_level_1",
        level=1,
        rate_fs_inv=1.0 / 120.0,
    ),
)
```

## 二能级示例

```python
from qudpy_sjh.utils.core import (
    NLevelPhysicalParams,
    RelaxationChannel,
    PureDephasingChannel,
)
from qudpy_sjh.utils.fields import make_default_gaussian_carrier_field

field = make_default_gaussian_carrier_field(
    E0_MV_per_cm = 0.005,
    laser_energy_eV = 1.55,
    pulse_center_fs = 0.0,
    pulse_sigma_fs = 8.0,
    phase_rad = 0.0,
    name = "probe",
)

params = NLevelPhysicalParams(
    energies_eV = (0.0, 1.55),
    dipole_matrix_D = ((0.0, 5.0), (5.0, 0.0)),
    t_start_fs = -200.0,
    t_end_fs = 500.0,
    dt_fs = 0.2,
    field = field,
    basis = ("g", "e"),
    relaxation_channels = (
        RelaxationChannel(
            name = "relaxation_1_to_0",
            from_level = 1,
            to_level = 0,
            T1_fs = 1000.0,
        ),
    ),
    pure_dephasing_channels = (
        PureDephasingChannel(
            name = "pure_dephasing_level_1",
            level = 1,
            Tphi_fs = 120.0,
        ),
    ),
    solver_mode = "lab_exact",
)
```

## 三能级示例

```python
params = NLevelPhysicalParams(
    energies_eV=(0.0, 1.48, 1.72),
    dipole_matrix_D=(
        (0.0, 4.0, 3.0),
        (4.0, 0.0, 0.0),
        (3.0, 0.0, 0.0),
    ),
    t_start_fs=-200.0,
    t_end_fs=500.0,
    dt_fs=0.2,
    field=field,
    basis=("g", "e1", "e2"),
    relaxation_channels=(
        RelaxationChannel(
            name="relaxation_1_to_0",
            from_level=1,
            to_level=0,
            T1_fs=1000.0,
        ),
        RelaxationChannel(
            name="relaxation_2_to_0",
            from_level=2,
            to_level=0,
            T1_fs=900.0,
        ),
    ),
    pure_dephasing_channels=(
        PureDephasingChannel(
            name="pure_dephasing_level_1",
            level=1,
            Tphi_fs=120.0,
        ),
        PureDephasingChannel(
            name="pure_dephasing_level_2",
            level=2,
            Tphi_fs=100.0,
        ),
    ),
    solver_mode="lab_exact",
)
```

## SolverParams

`SolverParams` 是 `ParaNormalizer.normalize()` 之后的内部参数摘要，包含 fs^-1 和 code-unit 两套量。普通用户和 example 通常不直接构造它。

常见字段包括：

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
omega_L_fs_inv
omega_L
t_start
t_end
dt
tlist
```

兼容属性包括：

```text
omega_eg_fs_inv
omega_eg
detuning_fs_inv
detuning
rabi_fs_inv
rabi
gamma1_fs_inv
gamma_phi_fs_inv
gamma2_fs_inv
gamma1
gamma_phi
gamma2
```

这些属性主要用于 debug metadata、sanity check 和旧 two-level 教学输出。不要把单一 `detuning` 或单一 `rabi` 当成 N-level 体系的完整物理描述。

## NLevelSolverParams

`NLevelSolverParams` 是 solver 内部使用的 code-unit 参数容器。普通用户侧示例应构造 `NLevelPhysicalParams`，再由 `ParaNormalizer` 生成内部参数。

常见字段包括：

```text
t_start
t_final
t_end
dt
hbar
energies
dipole_matrix
coupling_matrix
omega_drive
relaxation_channels
pure_dephasing_channels
field
tlist
times_fs
pulse_center
pulse_sigma
basis
detuning
```

`NLevelSolverParams` 不应作为用户侧主接口。它是 `run_case()` 内部路径的一部分。
