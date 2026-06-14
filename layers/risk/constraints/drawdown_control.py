"""
回撤控制: dynamic deleverage / circuit breaker
"""
import numpy as np


class DrawdownControl:
    """回撤管理"""

    def __init__(self, circuit_breaker: float = -0.20,
                 deleverage_factor: float = 0.5,
                 enabled: bool = True):
        self.circuit_breaker = circuit_breaker
        self.deleverage_factor = deleverage_factor
        self.enabled = enabled
        self._breached = False

    def apply(self, target_pos: float, current_drawdown: float,
              current_position: float) -> tuple[float, dict]:
        """
        Args:
            target_pos: 目标仓位
            current_drawdown: 当前回撤 (负值)
            current_position: 当前仓位

        Returns:
            (adjusted_position, log)
        """
        log = {}
        if not self.enabled:
            return target_pos, log

        original = target_pos

        # Circuit breaker: 回撤超过阈值, 强制清仓
        if current_drawdown <= self.circuit_breaker:
            self._breached = True
            log["circuit_breaker"] = f"dd={current_drawdown:.3f} ≤ {self.circuit_breaker}, force close"
            return 0.0, log

        # Dynamic deleverage: 回撤越深, 仓位越小
        dd_ratio = abs(current_drawdown) / abs(self.circuit_breaker)
        if dd_ratio > 0.3:  # 回撤超过熔断阈值的 30% 开始降仓
            scale = 1.0 - dd_ratio * self.deleverage_factor
            scale = max(0.1, scale)
            target_pos *= scale
            log["deleverage"] = f"dd={current_drawdown:.3f}, scale={scale:.2f}, pos {original:.3f}→{target_pos:.3f}"

        return float(target_pos), log

    def reset(self) -> None:
        self._breached = False

    @property
    def is_breached(self) -> bool:
        return self._breached
