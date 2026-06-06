"""
Ensemble 抽象基类
"""
from abc import ABC, abstractmethod
import numpy as np


class BaseEnsemble(ABC):
    """信号融合基类"""

    def __init__(self, name: str = "ensemble"):
        self.name = name
        self.models: list = []
        self.weights: list[float] = []
        self._fitted = False

    def add_model(self, model, weight: float = 1.0) -> None:
        """添加子模型"""
        self.models.append(model)
        self.weights.append(weight)

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray,
            X_val: np.ndarray = None, y_val: np.ndarray = None) -> dict:
        ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """输出 ensemble_score"""
        ...

    def predict_with_confidence(self, X: np.ndarray) -> dict:
        """输出 ensemble_score + ensemble_confidence"""
        preds = self.predict(X)
        confidence = self._estimate_confidence(X, preds)
        return {"score": preds, "confidence": confidence}

    def _estimate_confidence(self, X: np.ndarray, preds: np.ndarray) -> np.ndarray:
        """基于模型间预测的一致性估计置信度"""
        if len(self.models) < 2:
            return np.ones_like(preds)
        all_preds = np.column_stack([m.predict(X) for m in self.models])
        # 1 - 变异系数
        std = np.std(all_preds, axis=1)
        mean = np.abs(np.mean(all_preds, axis=1)) + 1e-8
        cv = std / mean
        return 1.0 / (1.0 + cv)

    def get_model_contributions(self, X: np.ndarray) -> dict:
        """各模型贡献比例"""
        if len(self.models) < 2:
            return {self.models[0].name if hasattr(self.models[0], 'name') else "model_0": 1.0}
        all_preds = np.column_stack([m.predict(X) for m in self.models])
        ensemble = np.average(all_preds, axis=1, weights=self.weights)
        contributions = {}
        for i, m in enumerate(self.models):
            name = getattr(m, 'name', f'model_{i}')
            contributions[name] = float(np.corrcoef(all_preds[:, i], ensemble)[0, 1])
        return contributions

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    @property
    def n_models(self) -> int:
        return len(self.models)
