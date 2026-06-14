"""
加权平均 Ensemble
"""
import numpy as np
from sklearn.metrics import r2_score
from layers.ensemble.base_ensemble import BaseEnsemble


class WeightedEnsemble(BaseEnsemble):
    """加权平均融合 — 权重按验证集性能分配"""

    def __init__(self, name: str = "weighted"):
        super().__init__(name=name)

    def fit(self, X, y, X_val=None, y_val=None) -> dict:
        metrics = {}

        # 如果提供了验证集，根据验证集 R² 动态分配权重
        if X_val is not None and y_val is not None and len(self.models) > 1:
            self.weights = []
            for m in self.models:
                preds = m.predict(X_val)
                r2 = max(r2_score(y_val, preds), 0.0)
                self.weights.append(r2)

            total = sum(self.weights)
            if total > 0:
                self.weights = [w / total for w in self.weights]
            else:
                self.weights = [1.0 / len(self.models)] * len(self.models)

            importances = dict(
                (getattr(m, 'name', f'model_{i}'), w)
                for i, (m, w) in enumerate(zip(self.models, self.weights))
            )
            metrics["weights"] = importances

        self._fitted = True
        return metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.models:
            return np.zeros(len(X))
        preds = np.column_stack([m.predict(X) for m in self.models])
        weights = np.array(self.weights or [1.0 / len(self.models)] * len(self.models))
        return np.average(preds, axis=1, weights=weights)
