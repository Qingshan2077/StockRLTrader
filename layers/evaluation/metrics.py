"""
回测指标计算器 — 从 NAV 序列和交易记录计算全套评估指标
"""
import numpy as np
from utils.helpers import (compute_drawdown, annualized_sharpe,
                            annualized_return, annualized_volatility,
                            sortino_ratio, max_drawdown_value)


class MetricsCalculator:
    """从回测结果计算全套评估指标"""

    @staticmethod
    def compute_all(nav_series, trades, benchmark_returns=None) -> dict:
        """
        Args:
            nav_series: list[np.ndarray] 净值序列
            trades: list[dict] 交易记录
            benchmark_returns: 基准日收益率序列 (可选)

        Returns:
            dict: 完整指标
        """
        nav = np.array(nav_series, dtype=np.float64)
        if len(nav) < 2:
            return {}

        # 日收益率
        daily_returns = np.diff(nav) / nav[:-1]

        # 基础指标
        ann_ret = annualized_return(nav)
        ann_vol = annualized_volatility(daily_returns)
        sharpe = annualized_sharpe(daily_returns)
        sortino = sortino_ratio(daily_returns)
        max_dd = max_drawdown_value(nav)
        calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0.0

        # 交易指标: 只统计卖出 (已实现盈亏)
        sell_trades = [t for t in trades if t.get("type") in ("sell", "sell_partial", "force_close")]
        total_sells = len(sell_trades)
        profitable_sells = sum(1 for t in sell_trades if t.get("profit", 0) > 0)
        win_rate = profitable_sells / total_sells if total_sells > 0 else 0.0

        # 换手率 (日均)
        total_executions = len(trades)
        if total_executions > 0 and len(nav) > 1:
            total_turnover = sum(abs(t.get("delta", 0)) for t in trades)
            turnover_rate = total_turnover / len(nav)
        else:
            turnover_rate = 0.0

        # 买入持有基准
        if benchmark_returns is not None and len(benchmark_returns) == len(daily_returns):
            bench_ann_ret = annualized_return(
                np.cumprod(1 + np.array(benchmark_returns)))
            alpha = ann_ret - bench_ann_ret
            excess = daily_returns - np.array(benchmark_returns)
            info_ratio = (np.mean(excess) / np.std(excess) * np.sqrt(252)
                          if np.std(excess) > 0 else 0.0)
        else:
            bench_ann_ret = 0.0
            alpha = 0.0
            info_ratio = 0.0

        # 平均交易收益 (卖出交易的已实现盈亏)
        avg_trade_return = (
            np.mean([t.get("profit_pct", 0) for t in sell_trades])
            if total_sells > 0 else 0.0
        )

        return {
            "total_return": float(nav[-1] / nav[0] - 1),
            "annualized_return": float(ann_ret),
            "annualized_volatility": float(ann_vol),
            "sharpe_ratio": float(sharpe),
            "sortino_ratio": float(sortino),
            "max_drawdown": float(max_dd),
            "calmar_ratio": float(calmar),
            "win_rate": float(win_rate),             # 卖出交易胜率
            "turnover_rate": float(turnover_rate),
            "total_trades": total_executions,         # 全部执行次数 (买入+卖出)
            "total_sells": total_sells,               # 卖出次数 (已实现盈亏)
            "profitable_sells": profitable_sells,     # 盈利卖出次数
            "avg_trade_return": float(avg_trade_return),
            "benchmark_return": float(bench_ann_ret),
            "alpha": float(alpha),
            "information_ratio": float(info_ratio),
            "final_value": float(nav[-1]),
            "n_days": len(nav),
        }
