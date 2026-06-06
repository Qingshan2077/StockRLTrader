"""
多实验对比工具
"""
import json
from pathlib import Path


class ExperimentComparison:
    """实验对比"""

    def __init__(self, storage):
        self.storage = storage

    def compare(self, exp_ids: list[str],
                metric_names: list[str] = None) -> list[dict]:
        """生成对比表"""
        if metric_names is None:
            metric_names = [
                "sharpe_ratio", "annualized_return", "max_drawdown",
                "sortino_ratio", "calmar_ratio", "win_rate", "turnover_rate",
            ]

        exps = self.storage.compare_experiments(exp_ids, metric_names)
        rows = []

        for exp in exps:
            row = {
                "experiment": exp.get("name", exp["exp_id"]),
                "status": exp.get("status", "unknown"),
            }
            fm = exp.get("final_metrics", {})
            for key in metric_names:
                val = fm.get(key)
                if val is not None:
                    if "rate" in key or "drawdown" in key:
                        row[key] = f"{float(val) * 100:.2f}%"
                    elif "ratio" in key and "sharpe" not in key.lower() and "sortino" not in key.lower():
                        row[key] = f"{float(val) * 100:.2f}%"
                    else:
                        row[key] = f"{float(val):.3f}"
                else:
                    row[key] = "N/A"
            rows.append(row)
        return rows
