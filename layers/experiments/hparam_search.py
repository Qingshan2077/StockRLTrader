"""
超参数搜索 — 网格搜索 / 随机搜索
"""
import numpy as np
from itertools import product
from typing import Callable


class HyperParameterSearch:
    """超参数搜索"""

    def __init__(self, objective_fn: Callable,
                 param_grid: dict[str, list],
                 method: str = "grid"):
        """
        Args:
            objective_fn: (params_dict) → metrics_dict
            param_grid: {param_name: [values]}
            method: grid | random
        """
        self.objective_fn = objective_fn
        self.param_grid = param_grid
        self.method = method
        self.results: list[dict] = []

    def run(self, n_iter: int = None) -> list[dict]:
        """执行搜索"""
        if self.method == "grid":
            self._grid_search()
        elif self.method == "random":
            self._random_search(n_iter or 20)
        return self.results

    def _grid_search(self) -> None:
        keys = list(self.param_grid.keys())
        values = list(self.param_grid.values())
        for combo in product(*values):
            params = dict(zip(keys, combo))
            metrics = self.objective_fn(params)
            self.results.append({"params": params, "metrics": metrics})

    def _random_search(self, n_iter: int) -> None:
        keys = list(self.param_grid.keys())
        for _ in range(n_iter):
            params = {}
            for key in keys:
                params[key] = np.random.choice(self.param_grid[key])
            metrics = self.objective_fn(params)
            self.results.append({"params": params, "metrics": metrics})

    def best(self, metric: str = "sharpe_ratio",
             maximize: bool = True) -> dict:
        """返回最优参数组合"""
        if not self.results:
            return {}
        best = self.results[0]
        best_val = best["metrics"].get(metric, -float("inf") if maximize else float("inf"))
        for r in self.results[1:]:
            val = r["metrics"].get(metric, -float("inf") if maximize else float("inf"))
            if (maximize and val > best_val) or (not maximize and val < best_val):
                best = r
                best_val = val
        return best
