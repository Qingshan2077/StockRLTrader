"""
信号监控 — signal drift / feature drift 检测
"""
import numpy as np
from collections import deque
from datetime import datetime


class SignalMonitor:
    """信号质量监控器"""

    def __init__(self, window_size: int = 100,
                 drift_threshold: float = 3.0):
        self.window_size = window_size
        self.drift_threshold = drift_threshold
        self.signal_history = deque(maxlen=window_size)
        self.feature_means = {}
        self.feature_stds = {}
        self.alerts: list[dict] = []

    def update(self, signal_score: float,
               features: dict[str, float] = None) -> None:
        """更新监控状态"""
        self.signal_history.append(signal_score)

        if features:
            for name, value in features.items():
                if name not in self.feature_means:
                    self.feature_means[name] = value
                    self.feature_stds[name] = 0.1
                else:
                    # 指数移动平均更新
                    alpha = 0.01
                    self.feature_means[name] = (alpha * value +
                        (1 - alpha) * self.feature_means[name])
                    self.feature_stds[name] = (alpha * (value - self.feature_means[name]) ** 2 +
                        (1 - alpha) * self.feature_stds[name])

    def check(self, signal_score: float,
              features: dict[str, float] = None) -> list[str]:
        """检查是否有异常, 返回告警列表"""
        warnings = []

        if len(self.signal_history) >= 10:
            hist_mean = np.mean(self.signal_history)
            hist_std = np.std(self.signal_history) or 1.0
            z = (signal_score - hist_mean) / hist_std
            if abs(z) > self.drift_threshold:
                msg = f"signal_drift: z={z:.2f}"
                warnings.append(msg)
                self.alerts.append({"type": "signal_drift", "z": z,
                                    "timestamp": datetime.now().isoformat()})

        if features and self.feature_means:
            for name, value in features.items():
                if name in self.feature_means:
                    z = abs(value - self.feature_means[name]) / max(self.feature_stds[name], 0.01)
                    if z > self.drift_threshold:
                        msg = f"feature_drift[{name}]: z={z:.2f}"
                        warnings.append(msg)

        return warnings

    def get_status(self) -> dict:
        return {
            "n_signals": len(self.signal_history),
            "signal_mean": float(np.mean(self.signal_history)) if self.signal_history else 0.0,
            "signal_std": float(np.std(self.signal_history)) if self.signal_history else 0.0,
            "n_alerts": len(self.alerts),
        }
