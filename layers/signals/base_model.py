"""
Alpha 模型抽象基类 — 所有信号模型必须实现的统一接口
"""
from abc import ABC, abstractmethod
from pathlib import Path
import pickle
import numpy as np


class BaseAlphaModel(ABC):
    """Alpha 信号模型基类"""

    def __init__(self, name: str = "base", config: dict = None):
        self.name = name
        self.config = config or {}
        self.model = None
        self._fitted = False

    @abstractmethod
    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray = None, y_val: np.ndarray = None) -> dict:
        """训练模型, 返回训练指标 dict"""
        ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """输出原始预测值"""
        ...

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """输出概率 (分类模型需覆盖)"""
        return self.predict(X)

    def get_importance(self) -> dict | None:
        """特征重要性 (可选覆盖)"""
        return None

    def save(self, path: str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"model": self.model, "config": self.config, "name": self.name}, f)

    def load(self, path: str) -> None:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"模型文件不存在: {path}")
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.model = data["model"]
        self.config = data.get("config", {})
        self.name = data.get("name", "unknown")
        self._fitted = True

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, fitted={self._fitted})"
