"""`sjh_learn.utils` 顶层轻量入口。

核心求解对象请从 `sjh_learn.utils.core` 导入；IO、plotting、fields、analysis
也应从各自子模块显式导入。顶层只保留常用 analysis 对象，避免重新形成
大杂烩式导出。
"""

from .analysis import DynamicsAnalysis

__all__ = ["DynamicsAnalysis"]
