# document.md 更新摘要

本次更新重点：

1. 把目录和模块描述从旧的 `utils/model.py`、`utils/solvers.py`、`utils/analysis/observables.py` 口径，更新为当前 `utils/core/` 与 `utils/spectroscopy/` 口径。
2. 补充 `parameters.py` 的 active API：`NLevelPhysicalParams`、`RelaxationChannel`、`PureDephasingChannel`、`SolverParams`、`NLevelSolverParams`。
3. 明确 `relaxation_channels` 和 `pure_dephasing_channels` 必须传 dataclass object，不应直接传 metadata dict。
4. 补充 `run_case()` 的 checkpoint 参数和 RWA legacy guard。
5. 补充 `DynamicsResult` 的常用字段和方法，包括 `field_MV_per_cm_values()`、checkpoint 和 dataframe exporters。
6. 把 spectroscopy observable 从旧的 analysis 口径改为 `utils/spectroscopy/`，并记录 `lab_frame_fft_response()` 的 FFT、window、mask、energy axis 行为。
7. 明确当前 `apply_time_window()` 只支持 `None / "none" / "hann"`；如果需要 `"hamming"`，应扩展公共 helper。
8. 增加 TA intrinsic response workflow 边界：TA example 可放 QuDPy，后续实验数据后处理应留给 UFANSYS。
9. 保留 RWA / ParameterSweep / solver-unit field 的删除边界。
