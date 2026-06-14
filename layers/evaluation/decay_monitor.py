"""信号衰减监控器

跟踪信号质量随时间变化，检测 IC 衰减并触发警报/自动重训练。

用法:
    monitor = SignalDecayMonitor(window=60, alert_threshold=0.0)
    monitor.update(ic_value=0.05, date=today)
    if monitor.should_retrain():
        # 触发重训练
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional


@dataclass
class DecayStatus:
    """衰减状态报告"""
    recent_ic_mean: float          # 近期 IC 均值
    recent_ic_std: float           # 近期 IC 标准差
    baseline_ic_mean: float        # 基线 IC 均值
    ic_drop: float                 # IC 衰减量 (baseline - recent)
    ic_drop_ratio: float           # IC 衰减比例
    should_retrain: bool           # 是否应该重训练
    reason: str = ""               # 触发原因


class SignalDecayMonitor:
    """信号衰减监控器

    Args:
        window: 近期 IC 计算窗口（交易日天数）
        baseline_window: 基线窗口（默认 252 个交易日 ≈ 1 年）
        alert_threshold: 警示阈值
            regression 模式: 近期 IC 均值低于此值时触发
            rank 模式: RankIC 低于此值时触发
        min_samples: 最小样本数，低于此值不判断
        mode: 'regression' | 'rank' (与标签类型匹配)
    """

    def __init__(
        self,
        window: int = 60,
        baseline_window: int = 252,
        alert_threshold: float = 0.0,
        min_samples: int = 20,
        mode: str = "regression",
    ):
        self.window = window
        self.baseline_window = baseline_window
        self.alert_threshold = alert_threshold
        self.min_samples = min_samples
        self.mode = mode

        # 内部存储
        self._ic_history: list[float] = []
        self._dates: list[pd.Timestamp] = []
        self._baseline_computed = False
        self._baseline_mean = 0.0
        self._n_decay_triggers = 0  # 连续触发次数

    def update(self, ic_value: float, date: Optional[pd.Timestamp] = None) -> DecayStatus:
        """添加一个 IC 观测值，返回当前衰减状态

        Args:
            ic_value: 当日 IC（或 RankIC）
            date: 日期

        Returns:
            DecayStatus 状态报告
        """
        self._ic_history.append(float(ic_value))
        if date is not None:
            self._dates.append(pd.Timestamp(date))

        # 计算基线（首次满 baseline_window 时固定）
        if not self._baseline_computed and len(self._ic_history) >= self.baseline_window:
            self._baseline_mean = float(np.mean(self._ic_history[:self.baseline_window]))
            self._baseline_computed = True

        # 样本不足
        if len(self._ic_history) < max(self.window, self.min_samples):
            return DecayStatus(
                recent_ic_mean=0.0,
                recent_ic_std=0.0,
                baseline_ic_mean=0.0,
                ic_drop=0.0,
                ic_drop_ratio=0.0,
                should_retrain=False,
                reason=f"样本不足 ({len(self._ic_history)}/{self.min_samples})",
            )

        # 近期 IC
        recent = self._ic_history[-self.window:]
        recent_mean = float(np.mean(recent))
        recent_std = float(np.std(recent))

        # 基线
        baseline_mean = self._baseline_mean if self._baseline_computed else float(np.mean(self._ic_history))

        ic_drop = baseline_mean - recent_mean
        ic_drop_ratio = ic_drop / abs(baseline_mean) if abs(baseline_mean) > 1e-8 else 0.0

        # 判断是否应重训练
        should_retrain = False
        reason = ""

        if recent_mean < self.alert_threshold:
            self._n_decay_triggers += 1
            if self._n_decay_triggers >= 5:  # 连续 5 次触发才重训练
                should_retrain = True
                reason = f"IC={recent_mean:.4f} 低于阈值 {self.alert_threshold}，连续 {self._n_decay_triggers} 次"
        elif self._baseline_computed and ic_drop_ratio > 0.5:
            # IC 较基线衰减超过 50%
            self._n_decay_triggers += 1
            if self._n_decay_triggers >= 5:
                should_retrain = True
                reason = f"IC 衰减 {ic_drop_ratio:.0%} (基线 {baseline_mean:.4f} → 近期 {recent_mean:.4f})"
        else:
            self._n_decay_triggers = max(0, self._n_decay_triggers - 1)  # 衰减计数

        return DecayStatus(
            recent_ic_mean=recent_mean,
            recent_ic_std=recent_std,
            baseline_ic_mean=baseline_mean,
            ic_drop=ic_drop,
            ic_drop_ratio=ic_drop_ratio,
            should_retrain=should_retrain,
            reason=reason,
        )

    @property
    def n_samples(self) -> int:
        return len(self._ic_history)

    def summary(self) -> str:
        """返回可读的状态摘要"""
        if not self._ic_history:
            return "衰减监控器: 无数据"
        recent = np.mean(self._ic_history[-self.window:]) if len(self._ic_history) >= self.window else 0
        baseline = self._baseline_mean if self._baseline_computed else np.mean(self._ic_history)
        return (
            f"衰减监控器: {len(self._ic_history)} 样本, "
            f"近期 IC={recent:.4f}, 基线 IC={baseline:.4f}, "
            f"触发次数={self._n_decay_triggers}"
        )

    def reset(self):
        """重置监控器（重训练后调用）"""
        self._ic_history.clear()
        self._dates.clear()
        self._baseline_computed = False
        self._baseline_mean = 0.0
        self._n_decay_triggers = 0
