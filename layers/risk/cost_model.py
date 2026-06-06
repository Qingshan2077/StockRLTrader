"""
交易成本模型 — 显式成本 + 隐式成本 (市场冲击)

Explicit: commission + tax + spread
Implicit: slippage + market impact
Market impact: impact = k * sqrt(order_size / ADV)
"""
import numpy as np


class CostModel:
    """交易成本估算器"""

    def __init__(self, commission: float = 0.001, slippage: float = 0.0005,
                 tax: float = 0.0, impact_model: str = "sqrt", k: float = 0.1):
        """
        Args:
            commission: 手续费率
            slippage: 固定滑点率
            tax: 印花税率
            impact_model: sqrt | linear | none
            k: 冲击系数 (sqrt 模型)
        """
        self.commission = commission
        self.slippage = slippage
        self.tax = tax
        self.impact_model = impact_model
        self.k = k

    def estimate(self, order_value: float, adv: float = None,
                 price: float = 1.0) -> dict:
        """
        估算交易成本

        Args:
            order_value: 订单金额 (绝对值)
            adv: 日均成交额
            price: 当前价格

        Returns:
            dict: 各项成本明细
        """
        if order_value <= 0 or price <= 0:
            return {"total": 0.0, "commission": 0.0, "slippage": 0.0,
                    "tax": 0.0, "market_impact": 0.0}

        # 显式成本
        commission_cost = order_value * self.commission
        slippage_cost = order_value * self.slippage
        tax_cost = order_value * self.tax

        # 隐式成本 (市场冲击)
        impact_cost = 0.0
        if self.impact_model == "sqrt" and adv is not None and adv > 0:
            order_ratio = order_value / adv
            impact_cost = order_value * self.k * np.sqrt(order_ratio)
        elif self.impact_model == "linear" and adv is not None and adv > 0:
            impact_cost = order_value * self.k * (order_value / adv)

        total = commission_cost + slippage_cost + tax_cost + impact_cost

        return {
            "total": float(total),
            "commission": float(commission_cost),
            "slippage": float(slippage_cost),
            "tax": float(tax_cost),
            "market_impact": float(impact_cost),
            "total_bps": float(total / order_value * 10000),
        }

    def estimate_turnover_cost(self, turnover_value: float,
                                adv: float = None) -> float:
        """快捷方法: 估算换仓总成本"""
        return self.estimate(turnover_value, adv)["total"]

    def __repr__(self) -> str:
        return (f"CostModel(comm={self.commission:.3%}, "
                f"slip={self.slippage:.3%}, impact={self.impact_model})")
