"""
仓位约束: max gross exposure / max net exposure / max single asset weight
"""
import numpy as np


class PositionConstraints:
    """仓位限制"""

    def __init__(self, max_gross: float = 1.0, max_net: float = 1.0,
                 max_single: float = 1.0):
        self.max_gross = max_gross
        self.max_net = max_net
        self.max_single = max_single

    def apply(self, target_pos: float) -> tuple[float, dict]:
        """
        应用仓位约束, 返回 (constrained_position, log)
        """
        log = {}
        original = target_pos

        # 单资产上限
        target_pos = np.clip(target_pos, -self.max_single, self.max_single)

        # 净仓位上限
        target_pos = np.clip(target_pos, -self.max_net, self.max_net)

        if abs(target_pos) > self.max_gross:
            target_pos = np.sign(target_pos) * self.max_gross

        if target_pos != original:
            log["position_constraint"] = f"{original:.3f} → {target_pos:.3f}"

        return float(target_pos), log
