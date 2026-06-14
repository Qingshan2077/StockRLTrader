"""
策略对比分析
"""
import numpy as np
from scipy import stats


class StrategyComparison:
    """多策略对比分析"""

    def __init__(self, results: dict[str, dict]):
        """
        Args:
            results: {strategy_name: metrics_dict}
        """
        self.results = results

    def compare_table(self) -> list[dict]:
        """生成对比表"""
        rows = []
        metric_keys = [
            ("annualized_return", "年化收益率", True),
            ("sharpe_ratio", "夏普比率", False),
            ("max_drawdown", "最大回撤", True),
            ("sortino_ratio", "Sortino", False),
            ("calmar_ratio", "Calmar", False),
            ("win_rate", "胜率", True),
            ("turnover_rate", "换手率", True),
            ("total_trades", "交易次数", False),
        ]

        for key, name, is_pct in metric_keys:
            row = {"metric": name}
            for strategy, metrics in self.results.items():
                val = metrics.get(key, 0)
                if is_pct and key not in ("sharpe_ratio", "sortino_ratio", "calmar_ratio"):
                    row[strategy] = f"{float(val) * 100:.2f}%"
                else:
                    row[strategy] = f"{float(val):.3f}"
            rows.append(row)
        return rows

    def best_strategy(self, metric: str = "sharpe_ratio") -> str:
        """返回指定指标最优的策略名"""
        best = None
        best_val = -float("inf")
        for name, m in self.results.items():
            val = m.get(metric, -float("inf"))
            if val > best_val:
                best_val = val
                best = name
        return best or ""

    def significance_test(self, metric: str = "sharpe_ratio") -> dict:
        """策略间统计显著性 (简化版)"""
        values = {k: v.get(metric, 0) for k, v in self.results.items()}
        if len(values) < 2:
            return {}

        names = list(values.keys())
        t_stat, p_value = stats.ttest_1samp(
            list(values.values()),
            np.mean(list(values.values()))
        )
        return {"t_stat": float(t_stat), "p_value": float(p_value)}

    def __repr__(self) -> str:
        return f"StrategyComparison({len(self.results)} strategies)"
