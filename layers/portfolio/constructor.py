"""
Portfolio Constructor — signal_score → portfolio weight

支持:
    - sigmoid mapping
    - z-score normalization
    - rank-based allocation
"""
import numpy as np
from scipy.special import expit  # sigmoid


class PortfolioConstructor:
    """信号到权重的映射"""

    def __init__(self, method: str = "sigmoid", alpha: float = 3.0,
                 max_weight: float = 1.0, min_weight: float = -1.0):
        """
        Args:
            method: sigmoid | zscore | rank
            alpha: sigmoid 陡峭度参数
            max_weight: 最大权重
            min_weight: 最小权重 (负值表示允许做空)
        """
        self.method = method
        self.alpha = alpha
        self.max_weight = max_weight
        self.min_weight = min_weight
        self._train_scores: np.ndarray | None = None  # 用于 zscore 标准化

    def fit(self, train_signals: np.ndarray) -> "PortfolioConstructor":
        """在训练集上计算标准化参数"""
        self._train_scores = train_signals
        return self

    def transform(self, signals: np.ndarray) -> np.ndarray:
        """信号 → 原始权重"""
        if self.method == "sigmoid":
            weights = 2 * expit(self.alpha * signals) - 1  # 映射到 (-1, 1)
        elif self.method == "zscore":
            if self._train_scores is None:
                self._train_scores = signals
            mean = np.mean(self._train_scores)
            std = np.std(self._train_scores) or 1.0
            z = (signals - mean) / std
            weights = np.clip(z / 3.0, -1.0, 1.0)  # 3-sigma → ±1
        elif self.method == "rank":
            # 按信号排序分配权重: top 20% long, bottom 20% short
            ranks = np.argsort(np.argsort(signals)) / len(signals)
            weights = (ranks - 0.5) * 2  # 映射到 (-1, 1)
        else:
            raise ValueError(f"未知方法: {self.method}")

        return np.clip(weights, self.min_weight, self.max_weight)

    def __repr__(self) -> str:
        return (f"PortfolioConstructor(method={self.method}, "
                f"range=[{self.min_weight}, {self.max_weight}])")
