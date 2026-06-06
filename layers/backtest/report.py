"""
回测报告生成器
"""
import json
import numpy as np
from pathlib import Path
from datetime import datetime


class BacktestReport:
    """回测报告 — 汇总输出"""

    def __init__(self, output_dir: str = "results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate(self, results: dict, ticker: str = "",
                 save: bool = True) -> dict:
        """
        Args:
            results: {"signal_only": metrics, "signal_risk": metrics, "full": metrics}
            ticker: 股票代码
            save: 是否保存到文件

        Returns:
            报告 dict
        """
        report = {
            "timestamp": datetime.now().isoformat(),
            "ticker": ticker,
            "modes": list(results.keys()),
            "comparison": self._make_comparison(results),
            "raw_metrics": {k: {kk: float(vv) if isinstance(vv, (np.floating, np.integer)) else vv
                                 for kk, vv in v.items()}
                            for k, v in results.items()},
        }

        if save:
            filename = f"{ticker}_report_{datetime.now():%Y%m%d_%H%M%S}.json"
            path = self.output_dir / filename
            with open(path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False, default=str)

        return report

    def _make_comparison(self, results: dict) -> list[dict]:
        """生成对比表"""
        rows = []
        metric_names = [
            ("annualized_return", "年化收益率"),
            ("sharpe_ratio", "夏普比率"),
            ("max_drawdown", "最大回撤"),
            ("calmar_ratio", "Calmar比率"),
            ("sortino_ratio", "Sortino比率"),
            ("win_rate", "胜率"),
            ("turnover_rate", "换手率"),
            ("total_trades", "交易次数"),
        ]

        for key, name in metric_names:
            row = {"指标": name}
            for mode, metrics in results.items():
                row[mode] = metrics.get(key, 0)
            rows.append(row)

        return rows

    def __repr__(self) -> str:
        return f"BacktestReport(output_dir={self.output_dir})"
