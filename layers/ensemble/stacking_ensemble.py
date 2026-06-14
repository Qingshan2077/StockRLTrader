"""
Stacking Ensemble — 用 meta-learner 融合子模型输出
"""
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from layers.ensemble.base_ensemble import BaseEnsemble


class StackingEnsemble(BaseEnsemble):
    """Stacking: 子模型输出作为 meta-learner 的输入"""

    def __init__(self, name: str = "stacking", meta_model=None):
        super().__init__(name=name)
        self.meta_model = meta_model or Ridge(alpha=1.0)

    def fit(self, X, y, X_val=None, y_val=None) -> dict:
        if len(self.models) < 2:
            self._fitted = True
            return {"error": "stacking needs at least 2 models"}

        # 用验证集训练 meta-learner (避免过拟合)
        if X_val is not None and y_val is not None:
            meta_X = self._get_meta_features(X_val)
            self.meta_model.fit(meta_X, np.ravel(y_val))
        else:
            meta_X = self._get_meta_features(X)
            self.meta_model.fit(meta_X, np.ravel(y))

        self._fitted = True

        metrics = {}
        if X_val is not None and y_val is not None:
            preds = self.predict(X_val)
            metrics["val_r2"] = float(r2_score(y_val, preds))

        return metrics

    def _get_meta_features(self, X: np.ndarray) -> np.ndarray:
        return np.column_stack([m.predict(X) for m in self.models])

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.models:
            return np.zeros(len(X))
        meta_X = self._get_meta_features(X)
        return self.meta_model.predict(meta_X)
