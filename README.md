# QuDPy-clean

## 项目定位

`QuDPy-clean` 是从旧 QuDPy 仓库迁移出的干净版本，用于学习、验证和组织 N-level optical Bloch dynamics、lab-frame physical field 输入、full-window quantum dynamics solver 以及基础谱学后处理。

当前仓库仍处于重构阶段。文档和示例应以当前 `main` 分支代码为准，避免继续沿用旧仓库中的实验性架构表述。

当前代码包路径为：

```text
qudpy_sjh/
```

旧文档中如果仍出现 `sjh_learn/`，应视为历史命名残留。后续文档建议统一改为 `qudpy_sjh/`。

## 当前主线

当前主线可以概括为：

```text
physical lab-frame field
-> NLevelPhysicalParams(..., field=field)
-> ParaNormalizer
-> internal code-unit field adapter
-> full-window lab_exact solver
-> DynamicsResult
-> spectroscopy / IO / plotting / workflow
```

当前推荐强调的能力包括：

- lab-frame physical field；
- field time shift；
- pump-probe / TA / 2DES field helper；
- physical-to-code normalizer；
- full-window lab-frame exact solver；
- ordinary `DynamicsResult`；
- spectroscopy absorption response；
- 普通 CSV / NPZ / JSON metadata 输出。

当前主线不宣传 piecewise propagation、dark propagation、active/dark `PropagationPiece`、`PieceDynamicsResultSeries`、`materialize_full` 或 `run_case(piecewise=...)`。

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

如果后续补充 `pyproject.toml`、`setup.py` 或其它安装配置，应同步更新本节。

## 最小工作流

下面示例展示当前主线的最小调用边界。参数只用于说明接口结构，不代表具体物理系统建议。

```python
from qudpy_sjh.utils.fields import make_default_gaussian_carrier_field
from qudpy_sjh.utils.core import (
    NLevelPhysicalParams,
    RelaxationChannel,
    PureDephasingChannel,
    run_case,
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

result = run_case(params)
```

`result` 是普通 `DynamicsResult`，可用于：

```python
density = result.density_array()
populations = result.populations()
field_values = result.field_MV_per_cm_values()
metadata = result.metadata_dict()
```

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
      specific/
        __init__.py
        basic_fields.py
        ta_fields.py
        twodes_fields.py

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

`docs/document_summary.md` 建议替代旧的 `docs/document_updated.md` 和 `docs/document_update_summary.md`。

## 测试

当前未确认存在独立稳定的 `tests/` 目录。建议先做编译检查：

```bash
python -m compileall qudpy_sjh
```

然后运行仓库中当前维护的 demo、example 或 scratch validation scripts。具体脚本名称以后续维护的 `bin/`、`experiments/` 或 `scratch/` 文档为准。

## 当前不在主线的内容

以下旧架构或实验性方向不应在 README 和主线文档中宣传：

```text
piecewise propagation
dark propagation
active/dark PropagationPiece
PieceDynamicsResultSeries
materialize_full
run_case(piecewise=...)
save_result_case 对 piecewise series 的支持
long-window piecewise benchmark
复杂 transient_absorption scaffold 主线
```

如果未来需要重新引入其中某些能力，应先作为独立设计 proposal 讨论，不应隐式恢复到当前 core / IO / README 主线中。
