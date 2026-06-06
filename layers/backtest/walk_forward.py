"""
Walk-Forward 回测

支持:
    - expanding window: train 窗口不断扩大
    - rolling window: train 窗口固定长度, 整体滚动
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class WalkForwardResult:
    """单个 walk-forward fold 的结果"""
    fold: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    metrics: dict = field(default_factory=dict)
    model_path: str = ""


class WalkForwardBacktest:
    """Walk-Forward 回测引擎"""

    def __init__(self, n_folds: int = 5,
                 train_window_years: int = 3,
                 test_window_years: int = 1,
                 mode: str = "expanding"):
        """
        Args:
            n_folds: fold 数量
            train_window_years: 训练窗口 (年)
            test_window_years: 测试窗口 (年)
            mode: expanding | rolling
        """
        self.n_folds = n_folds
        self.train_window_years = train_window_years
        self.test_window_years = test_window_years
        self.mode = mode
        self.results: list[WalkForwardResult] = []

    def run(self, data: pd.DataFrame,
            train_fn: Callable,
            test_fn: Callable) -> list[WalkForwardResult]:
        """
        Args:
            data: 完整数据 (按时间排序)
            train_fn: (train_data) → model_path
            test_fn: (test_data, model_path) → metrics dict

        Returns:
            每个 fold 的结果列表
        """
        trading_days_per_year = 252
        train_size = self.train_window_years * trading_days_per_year
        test_size = self.test_window_years * trading_days_per_year

        total_len = len(data)
        self.results = []

        for fold in range(self.n_folds):
            # 计算窗口边界
            test_end = total_len - fold * test_size
            test_start = max(0, test_end - test_size)

            if self.mode == "expanding":
                train_start = 0
            else:
                train_start = max(0, test_start - train_size)

            train_end = test_start

            if train_end - train_start < 100 or test_end - test_start < 20:
                break

            # 切分数据
            train_data = data.iloc[train_start:train_end]
            test_data = data.iloc[test_start:test_end]

            # 训练
            model_path = train_fn(train_data)

            # 测试
            metrics = test_fn(test_data, model_path)

            result = WalkForwardResult(
                fold=fold,
                train_start=str(data.index[train_start]),
                train_end=str(data.index[train_end - 1]),
                test_start=str(data.index[test_start]),
                test_end=str(data.index[test_end - 1]),
                metrics=metrics,
                model_path=model_path,
            )
            self.results.append(result)

            if test_start <= 0:
                break

        return self.results

    def summary(self) -> dict:
        """汇总所有 fold 的指标"""
        if not self.results:
            return {}

        all_metrics = {}
        for r in self.results:
            for k, v in r.metrics.items():
                if isinstance(v, (int, float)):
                    all_metrics.setdefault(k, []).append(v)

        return {
            "n_folds_completed": len(self.results),
            "avg_metrics": {k: float(np.mean(v)) for k, v in all_metrics.items()},
            "std_metrics": {k: float(np.std(v)) for k, v in all_metrics.items()},
        }
