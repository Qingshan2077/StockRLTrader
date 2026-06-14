"""
流动性约束: ADV limit / minimum volume
"""
import numpy as np


class LiquidityConstraints:
    """流动性过滤"""

    def __init__(self, adv_limit_ratio: float = 0.01,
                 minimum_volume: float = 10000,
                 enabled: bool = True):
        """
        Args:
            adv_limit_ratio: 仓位上限 = adv_limit_ratio * ADV
            minimum_volume: 最低成交量 (低于此值的资产不交易)
        """
        self.adv_limit_ratio = adv_limit_ratio
        self.minimum_volume = minimum_volume
        self.enabled = enabled

    def apply(self, target_pos: float, avg_daily_volume: float,
              current_price: float, portfolio_value: float) -> tuple[float, dict]:
        """
        Args:
            target_pos: 目标仓位比例
            avg_daily_volume: 日均成交量 (股)
            current_price: 当前价格
            portfolio_value: 组合总价值

        Returns:
            (constrained_position, log)
        """
        log = {}
        if not self.enabled:
            return target_pos, log

        original = target_pos

        # 最低成交量过滤
        if avg_daily_volume < self.minimum_volume:
            log["volume_filter"] = f"ADV={avg_daily_volume:.0f} < {self.minimum_volume}, no trade"
            return 0.0, log

        # ADV 限制: 持仓市值不超过日均成交额的 adv_limit_ratio
        max_position_value = avg_daily_volume * current_price * self.adv_limit_ratio
        max_pos_ratio = max_position_value / max(portfolio_value, 1.0)
        target_pos = np.clip(target_pos, -max_pos_ratio, max_pos_ratio)

        if abs(target_pos - original) > 1e-6:
            log["liquidity_limit"] = f"max_pos={max_pos_ratio:.4f}, pos {original:.3f}→{target_pos:.3f}"

        return float(target_pos), log
