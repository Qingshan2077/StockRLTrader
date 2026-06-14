"""
Voting Ensemble — 分类模式投票
"""
import numpy as np
from scipy.stats import mode
from layers.ensemble.base_ensemble import BaseEnsemble


class VotingEnsemble(BaseEnsemble):
    """分类投票融合 (hard/soft voting)"""

    def __init__(self, name: str = "voting", voting: str = "soft"):
        super().__init__(name=name)
        self.voting = voting  # hard | soft
        self.classes: list | None = None

    def fit(self, X, y, X_val=None, y_val=None) -> dict:
        self.classes = sorted(set(np.ravel(y).astype(int)))
        self._fitted = True
        return {"n_classes": len(self.classes)}

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.models:
            return np.zeros(len(X))

        if self.voting == "hard":
            # 多数投票
            all_preds = np.column_stack([
                np.round(m.predict(X)).astype(int) for m in self.models
            ])
            result = mode(all_preds, axis=1, keepdims=False)[0]
        else:
            # 软投票: 平均概率
            all_probs = np.stack([m.predict_proba(X) for m in self.models])
            avg_probs = np.mean(all_probs, axis=0)
            result = np.argmax(avg_probs, axis=1)
        return result

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.models:
            return np.zeros((len(X), 1))
        all_probs = np.stack([m.predict_proba(X) for m in self.models])
        return np.mean(all_probs, axis=0)
