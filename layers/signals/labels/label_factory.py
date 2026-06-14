"""
标签工厂 — 统一构造回归/分类/排序标签
"""
import pandas as pd
import numpy as np
from typing import Literal


class LabelFactory:
    """多类型标签构造器"""

    def __init__(self, label_type: str = "regression",
                 horizons: list[int] = None,
                 classification_threshold: float = 0.005):
        """
        Args:
            label_type: regression | classification | ranking
            horizons: 预测天数列表, 如 [1, 5, 10, 20]
            classification_threshold: 分类阈值 (涨/跌分界)
        """
        self.label_type = label_type
        self.horizons = horizons or [5]
        self.classification_threshold = classification_threshold

    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        """为所有 horizon 构造标签, 返回 DataFrame"""
        labels = pd.DataFrame(index=df.index)
        close = df['Close']

        for h in self.horizons:
            fwd_ret = close.shift(-h) / close - 1

            if self.label_type == "regression":
                labels[f'label_ret_{h}d'] = fwd_ret
            elif self.label_type == "classification":
                labels[f'label_up_{h}d'] = (fwd_ret > self.classification_threshold).astype(int)
                labels[f'label_down_{h}d'] = (fwd_ret < -self.classification_threshold).astype(int)
            elif self.label_type == "ranking":
                labels[f'label_rank_{h}d'] = fwd_ret.rank(pct=True)

        return labels

    def get_target(self, labels: pd.DataFrame, horizon: int = 5) -> pd.Series:
        """获取指定 horizon 的有效标签 (去除 NaN)"""
        if self.label_type == "regression":
            col = f'label_ret_{horizon}d'
        elif self.label_type == "classification":
            col = f'label_up_{horizon}d'
        elif self.label_type == "ranking":
            col = f'label_rank_{horizon}d'
        else:
            raise ValueError(f"未知标签类型: {self.label_type}")
        return labels[col].dropna()

    def __repr__(self) -> str:
        return (f"LabelFactory(type={self.label_type}, "
                f"horizons={self.horizons})")
