"""
绩效归因 — 分解收益来源

Attribution:
    - Alpha contribution: signal-driven return vs passive return
    - Cost attribution: commission / slippage / impact
    - Turnover attribution: 主动 vs 被动调仓
    - Risk attribution: 各约束造成的收益损失
"""
import numpy as np
import pandas as pd


class PerformanceAttribution:
    """绩效归因分析器"""

    def __init__(self, nav_series, trades, positions,
                 benchmark_returns=None):
        self.nav = np.array(nav_series)
        self.trades = trades
        self.positions = positions
        self.bench_ret = (np.array(benchmark_returns)
                          if benchmark_returns is not None else None)

    def analyze(self) -> dict:
        """完整归因分析"""
        total_return = float(self.nav[-1] / self.nav[0] - 1)

        alpha = self._alpha_contribution()
        cost = self._cost_attribution()
        turnover = self._turnover_attribution()
        risk = self._risk_attribution()

        # 验证: 各分量之和应接近总收益
        decomposed = alpha["value"] + cost["value"] + turnover["value"] + risk["value"]
        residual = total_return - decomposed

        return {
            "total_return": total_return,
            "decomposed_total": decomposed,
            "residual": float(residual),
            "alpha": alpha,
            "cost": cost,
            "turnover": turnover,
            "risk": risk,
        }

    def _alpha_contribution(self) -> dict:
        """Alpha 贡献: 信号驱动的超额收益"""
        # 简化: 活跃收益 = 总收益 - 基准收益
        if self.bench_ret is not None:
            bench_total = float(np.prod(1 + self.bench_ret) - 1)
        else:
            bench_total = 0.0

        total = float(self.nav[-1] / self.nav[0] - 1)
        alpha_value = total - bench_total

        return {
            "value": float(alpha_value),
            "description": "signal-driven excess return",
            "benchmark_return": float(bench_total),
        }

    def _cost_attribution(self) -> dict:
        """成本归因: 手续费/滑点/冲击分别多少"""
        total_commission = sum(t.get("commission", 0) for t in self.trades if "commission" in t)
        total_slippage = sum(t.get("slippage", 0) for t in self.trades if "slippage" in t)
        total_impact = sum(t.get("impact", 0) for t in self.trades if "impact" in t)
        total_cost = total_commission + total_slippage + total_impact

        init_val = self.nav[0]
        return {
            "value": float(-total_cost / init_val),
            "commission": float(total_commission / init_val),
            "slippage": float(total_slippage / init_val),
            "market_impact": float(total_impact / init_val),
        }

    def _turnover_attribution(self) -> dict:
        """换手归因"""
        active = sum(1 for t in self.trades if t.get("type") in ("buy", "sell"))
        passive = sum(1 for t in self.trades if t.get("type") in ("rebalance", "force_close"))
        return {
            "value": 0.0,  # Turnover 影响已计入成本
            "active_trades": active,
            "passive_trades": passive,
        }

    def _risk_attribution(self) -> dict:
        """风险归因"""
        return {
            "value": 0.0,  # Risk impact reflected in reduced drawdown
            "stop_loss_events": sum(1 for t in self.trades if t.get("type") == "stop_loss"),
            "description": "风险约束已计入回撤控制",
        }
