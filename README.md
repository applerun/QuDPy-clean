# QuDPy-clean

## 项目定位

`QuDPy-clean` 是从旧 QuDPy 仓库迁移出的干净版本，用于学习、验证和组织 N-level optical Bloch dynamics、lab-frame physical field 输入、full-window quantum dynamics solver 以及基础谱学后处理。

当前文档以 `main` 分支代码为准。代码包路径统一为：

```text
qudpy_sjh/
```

## 当前主线

当前主线可以概括为：

```text
lab-frame physical field
-> NLevelPhysicalParams(..., field=field)
-> ParaNormalizer
-> internal code-unit field adapter
-> full-window lab_exact solver
-> DynamicsResult
-> spectroscopy / IO / plotting / workflow
```

推荐使用的 field 主线是：

```text
qudpy_sjh.utils.fields
  FieldPhyRoot
  FieldPhyCustomed
  TimeShiftedField
  FieldPhySeries

qudpy_sjh.utils.fields.carrier_envelope
  CarrierSpec
  EnvelopeSpec
  GaussianEnvelopeSpec
  SechEnvelopeSpec
  ConstantEnvelopeSpec
  CarrierEnvelopeField
  make_gaussian_carrier_envelope_field
  make_pump_probe_field_series
```

当前主线不宣传 piecewise propagation、dark propagation、`ActiveWindow`、`PropagationPiece`、`PieceDynamicsResultSeries`、`materialize_full`、`run_case(piecewise=...)` 或复杂 transient_absorption scaffold。

## 安装与环境

当前仓库根目录未确认存在正式 packaging 配置。建议先在已有 Python 环境中从仓库根目录运行，并确保环境中已安装项目依赖，例如：

```text
numpy
pandas
qutip
matplotlib
```

在仓库根目录运行脚本时，通常可以直接导入：

```python
import qudpy_sjh
```

## 最小工作流

下面示例展示当前主线的最小调用边界。参数只用于说明接口结构，不代表具体物理系统建议。

```python
from qudpy_sjh.utils.core import (
    NLevelPhysicalParams,
    ParaNormalizer,
    PureDephasingChannel,
    RelaxationChannel,
    run_case,
)
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
    solver_mode="lab_exact",
)

result = run_case(params, normalizer=ParaNormalizer(auto_scale=True))
```

`result` 是普通 `DynamicsResult`，可用于：

```python
density = result.density_array()
populations = result.populations()
field_values = result.field_MV_per_cm_values()
metadata = result.metadata_dict()
```

## carrier-envelope field 语义

`CarrierEnvelopeField` 使用有限脉冲的 carrier-envelope 约定：

```text
E(t) = 2 E0 envelope(t) cos[omega * (t - center) + phase]
```

其中 `center` 来自 envelope 的 `center_fs`。`phase_rad` 是相对于 envelope center 定义的 carrier phase，不是全局 lab-frame `omega*t + phase`。

`CarrierEnvelopeField` 不包含 pump / probe / LO 这类 role。role 应由 `FieldPhySeries.sub_field_names`、case metadata 或 workflow 层表达。

## pump-probe field 组合

当前推荐用普通 `CarrierEnvelopeField` 构造单个脉冲，再用 `FieldPhySeries` 表达 pump / probe 组合：

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

`make_pump_probe_field_series(pump_field=..., probe_field=...)` 是同一组合方式的便利 helper。delay scan、phase cycling、probe-only reference 和 TA map 生成属于 workflow 层。

## 目录结构

当前建议按以下结构理解主线代码：

```text
qudpy_sjh/
  utils/
    core/
      __init__.py
      parameters.py
      normalization.py
      model.py
      solvers.py
      results.py
      config.py

    fields/
      __init__.py
      lab_fields.py
      field_series.py
      carrier_envelope/
        __init__.py
        carrier_spec.py
        envelope_spec.py
        carrier_envelope_field.py
        builders.py

    spectroscopy/
      __init__.py
      observables.py
      absorption_spectra.py
      theory.py

    io.py
    plotting.py
    checks.py
```

`docs/detailed API/` 可作为字段级或函数级 API 细节文档目录。README 只做主线导览，不重复所有 API 细节。

## 文档导航

建议阅读顺序：

```text
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

## 测试

当前未确认存在独立稳定的 `tests/` 目录。建议先做编译检查：

```bash
python -m compileall qudpy_sjh
```

然后运行当前维护的 example，例如：

```bash
python bin/examples/ta/ta_three_level_intrinsic_response_phase_cycling_demo.py --quick
```

## 当前不在主线的内容

以下旧接口或旧架构不应作为当前推荐 API 宣传：

```text
specific/basic_fields.py
specific/ta_fields.py
specific/twodes_fields.py
CarrierFieldPhysical
GaussianCarrierFieldPhysical
TAField
TwoDESField
make_default_gaussian_carrier_field
make_ta_gaussian_field
make_pump_probe_field_from_templates
make_twodes_gaussian_field
piecewise propagation
dark propagation
ActiveWindow
PropagationPiece
PieceDynamicsResultSeries
materialize_full
run_case(piecewise=...)
save_result_case 对 piecewise series 的支持
complex transient_absorption scaffold
```

如果未来需要重新引入其中某些能力，应先作为独立设计 proposal 讨论，不应隐式恢复到当前 core / IO / README 主线中。
