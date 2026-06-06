"""
Ensemble Manager — 统一管理多模型融合
"""
import numpy as np
from typing import Optional


class EnsembleManager:
    """统一 Ensemble 管理器"""

    def __init__(self, method: str = "weighted"):
        self.method = method
        self.ensemble = None
        self._init_ensemble()

    def _init_ensemble(self):
        from layers.ensemble import (
            WeightedEnsemble, StackingEnsemble,
            VotingEnsemble, RegimeEnsemble,
        )
        mapping = {
            "weighted": WeightedEnsemble,
            "stacking": StackingEnsemble,
            "voting": VotingEnsemble,
            "regime": RegimeEnsemble,
        }
        cls = mapping.get(self.method)
        if cls is None:
            raise ValueError(f"未知 ensemble 方法: {self.method}")
        self.ensemble = cls()

    def add_models(self, models: list) -> None:
        for m in models:
            self.ensemble.add_model(m)

    def fit(self, X, y, X_val=None, y_val=None) -> dict:
        return self.ensemble.fit(X, y, X_val, y_val)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.ensemble.predict(X)

    def predict_with_confidence(self, X: np.ndarray) -> dict:
        return self.ensemble.predict_with_confidence(X)

    def get_contributions(self, X: np.ndarray) -> dict:
        return self.ensemble.get_model_contributions(X)

    @property
    def n_models(self) -> int:
        return self.ensemble.n_models

    def __repr__(self) -> str:
        return f"EnsembleManager(method={self.method}, models={self.n_models})"
