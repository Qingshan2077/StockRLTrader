"""
波动率约束: target vol / vol scaling
"""
import numpy as np


class VolatilityConstraints:
    """波动率控制"""

    def __init__(self, target_vol: float = 0.25, vol_window: int = 20,
                 vol_threshold: float = 1.5, enabled: bool = True):
        self.target_vol = target_vol
        self.vol_window = vol_window
        self.vol_threshold = vol_threshold
        self.enabled = enabled

    def apply(self, target_pos: float, current_vol: float) -> tuple[float, dict]:
        """
        波动率缩放: 实际仓位 = 目标仓位 * (target_vol / current_vol)
        """
        log = {}
        if not self.enabled or current_vol <= 0:
            return target_pos, log

        original = target_pos

        # Target volatility scaling
        vol_scale = min(1.0, self.target_vol / max(current_vol, self.target_vol))
        target_pos *= vol_scale

        # 波动率过高额外降仓
        if current_vol > self.vol_threshold * 0.2:  # 0.2 ≈ 中等波动基准
            target_pos *= 0.7

        if abs(target_pos - original) > 1e-6:
            log["volatility_constraint"] = f"vol={current_vol:.3f}, pos {original:.3f}→{target_pos:.3f}"

        return float(target_pos), log
