"""
标签构造器 — 构建未来 N 日收益率标签，严格防泄露

支持:
    - 回归: fwd_return_{N}d
    - 分类: label_{N}d ∈ {-1, 0, 1}
    - 多 horizon: [1, 3, 5, 10, 20]
"""
import pandas as pd
import numpy as np


class LabelBuilder:
    """构造未来收益率标签"""

    def __init__(self, horizon: int = 5, label_type: str = "regression",
                 classification_threshold: float = 0.005):
        if horizon < 1:
            raise ValueError(f"horizon 必须 >= 1, 当前: {horizon}")
        self.horizon = horizon
        self.label_type = label_type
        self.classification_threshold = classification_threshold

    def build_labels(self, df: pd.DataFrame) -> pd.Series:
        """
        构造标签。

        回归模式: fwd_return_{horizon}d = Close[t+horizon] / Close[t] - 1
        分类模式: label = +1 (涨>threshold), -1 (跌<-threshold), 0 (震荡)
        """
        close = df['Close']
        fwd_return = close.shift(-self.horizon) / close - 1

        if self.label_type == "classification":
            threshold = self.classification_threshold
            label = pd.Series(0, index=fwd_return.index, dtype=int)
            label[fwd_return > threshold] = 1
            label[fwd_return < -threshold] = -1
            label.name = self.label_name
            return label

        fwd_return.name = self.label_name
        return fwd_return

    @property
    def label_name(self) -> str:
        if self.label_type == "classification":
            return f"label_{self.horizon}d"
        return f"fwd_return_{self.horizon}d"

    def align(self, features: pd.DataFrame, labels: pd.Series
              ) -> tuple[pd.DataFrame, pd.Series]:
        """对齐特征和标签，去除无法计算标签的尾部行"""
        valid_labels = labels.dropna()
        aligned_features = features.loc[valid_labels.index]
        return aligned_features, valid_labels

    def __repr__(self) -> str:
        return (f"LabelBuilder(horizon={self.horizon}, "
                f"type={self.label_type})")
