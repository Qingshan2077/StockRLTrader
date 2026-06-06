"""
Rebalancer — 组合再平衡策略
"""
import numpy as np
from enum import Enum


class RebalanceMethod(Enum):
    CALENDAR = "calendar"        # 固定周期再平衡
    THRESHOLD = "threshold"      # 阈值触发再平衡


class Rebalancer:
    """组合再平衡执行器"""

    def __init__(self, method: str = "threshold",
                 frequency: int = 5,
                 threshold: float = 0.1):
        """
        Args:
            method: calendar | threshold
            frequency: calendar 模式下的再平衡周期(天)
            threshold: threshold 模式下的触发阈值(权重偏离)
        """
        self.method = method
        self.frequency = frequency
        self.threshold = threshold
        self._days_since = 0

    def should_rebalance(self, current_weight: float,
                         target_weight: float) -> bool:
        """判断是否应该再平衡"""
        if self.method == "calendar":
            self._days_since += 1
            if self._days_since >= self.frequency:
                self._days_since = 0
                return True
            return False

        elif self.method == "threshold":
            if abs(target_weight - current_weight) >= self.threshold:
                return True
            return False

        return False

    def compute_rebalance_amount(self, current_weight: float,
                                  target_weight: float,
                                  portfolio_value: float) -> float:
        """计算需要调仓的金额"""
        delta_weight = target_weight - current_weight
        return delta_weight * portfolio_value

    def reset(self) -> None:
        self._days_since = 0

    def __repr__(self) -> str:
        return (f"Rebalancer(method={self.method}, "
                f"freq={self.frequency}, threshold={self.threshold})")
