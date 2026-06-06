"""
可配置多目标奖励函数

reward = pnl - λ1*cost - λ2*turnover - λ3*drawdown - λ4*volatility

支持 reward ablation: 每次去掉一项, 对比效果
"""
import numpy as np
from typing import Optional


class RewardFactory:
    """多目标奖励工厂"""

    def __init__(self, pnl_weight: float = 1.0,
                 cost_penalty: float = 0.1,
                 turnover_penalty: float = 0.05,
                 drawdown_penalty: float = 0.1,
                 volatility_penalty: float = 0.05,
                 drawdown_threshold: float = 0.02,
                 disabled_components: list[str] = None):
        """
        Args:
            disabled_components: 用于 ablation 的禁用列表,
                可选: "pnl", "cost", "turnover", "drawdown", "volatility"
        """
        self.pnl_weight = pnl_weight
        self.cost_penalty = cost_penalty
        self.turnover_penalty = turnover_penalty
        self.drawdown_penalty = drawdown_penalty
        self.volatility_penalty = volatility_penalty
        self.drawdown_threshold = drawdown_threshold
        self.disabled = set(disabled_components or [])

        # 记录每步的奖励分量 (用于分析)
        self.components_log: list[dict] = []

    def compute(self, portfolio_return: float, cost: float,
                turnover: float, drawdown: float,
                volatility: float = 0.0) -> dict:
        """
        计算多目标奖励

        Returns:
            dict: reward + 各分量值
        """
        pnl_term = self.pnl_weight * portfolio_return
        cost_term = -self.cost_penalty * cost
        turnover_term = -self.turnover_penalty * abs(turnover)
        dd_excess = max(0, abs(drawdown) - self.drawdown_threshold)
        dd_term = -self.drawdown_penalty * dd_excess
        vol_term = -self.volatility_penalty * volatility

        # Ablation: 禁用指定分量
        if "pnl" in self.disabled:
            pnl_term = 0.0
        if "cost" in self.disabled:
            cost_term = 0.0
        if "turnover" in self.disabled:
            turnover_term = 0.0
        if "drawdown" in self.disabled:
            dd_term = 0.0
        if "volatility" in self.disabled:
            vol_term = 0.0

        reward = pnl_term + cost_term + turnover_term + dd_term + vol_term

        components = {
            "reward": float(np.clip(reward, -10.0, 10.0)),
            "pnl_term": float(pnl_term),
            "cost_term": float(cost_term),
            "turnover_term": float(turnover_term),
            "drawdown_term": float(dd_term),
            "volatility_term": float(vol_term),
        }
        self.components_log.append(components)
        return components

    def get_ablation_variants(self) -> list["RewardFactory"]:
        """生成所有 ablation 变体 (去掉一项)"""
        variants = []
        for comp in ["pnl", "cost", "turnover", "drawdown", "volatility"]:
            disabled = list(self.disabled) + [comp]
            variant = RewardFactory(
                pnl_weight=self.pnl_weight,
                cost_penalty=self.cost_penalty,
                turnover_penalty=self.turnover_penalty,
                drawdown_penalty=self.drawdown_penalty,
                volatility_penalty=self.volatility_penalty,
                drawdown_threshold=self.drawdown_threshold,
                disabled_components=disabled,
            )
            variant.name = f"ablation_no_{comp}"
            variants.append(variant)

        # Full reward
        full = RewardFactory(
            pnl_weight=self.pnl_weight,
            cost_penalty=self.cost_penalty,
            turnover_penalty=self.turnover_penalty,
            drawdown_penalty=self.drawdown_penalty,
            volatility_penalty=self.volatility_penalty,
            drawdown_threshold=self.drawdown_threshold,
        )
        full.name = "full_reward"
        variants.append(full)
        return variants

    def reset_log(self) -> None:
        self.components_log.clear()

    def __repr__(self) -> str:
        disabled_str = f" disabled={self.disabled}" if self.disabled else ""
        return (f"RewardFactory(pnl={self.pnl_weight}, cost={self.cost_penalty}, "
                f"turn={self.turnover_penalty}, dd={self.drawdown_penalty}){disabled_str}")
