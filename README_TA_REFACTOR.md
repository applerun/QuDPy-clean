# TA refactor draft

建议复制到仓库相对路径：

```text
qudpy_sjh/experiments/ta/__init__.py
qudpy_sjh/experiments/ta/ta_settings.py
qudpy_sjh/experiments/ta/ta_case_plan.py
qudpy_sjh/experiments/ta/ta_result.py
bin/examples/ta/ta_three_level_intrinsic_response_plan_demo.py
```

主流程：

```text
TASettings -> TAPlan.execute() -> TAResult -> TAResultIO / analysis
```

计算阶段保存 checkpoint，并默认保存 TA 标准谱学输出。raw `DynamicsResult`
的 preview/export 通过 `TAPlan.save_preview_from_checkpoints()` 从 `.ckp`
后处理生成。
