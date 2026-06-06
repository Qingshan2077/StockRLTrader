"""
Regime Switching Ensemble — 不同行情用不同模型
"""
import numpy as np
from layers.ensemble.base_ensemble import BaseEnsemble


class RegimeEnsemble(BaseEnsemble):
    """
    Regime Switching: 根据市场状态选择模型

    Regime 分类 (简单三维):
        - trend:   +1 上升 / -1 下降 / 0 震荡
        - vol:     +1 高波动 / -1 低波动
        - volume:  +1 活跃 / -1 冷清
    """

    def __init__(self, name: str = "regime"):
        super().__init__(name=name)
        self.regime_map: dict[tuple, int] = {}  # regime → model_index

    def fit(self, X, y, X_val=None, y_val=None) -> dict:
        """自动学习 regime → best model 映射"""
        if len(self.models) < 2:
            self._fitted = True
            return {}

        # 用验证集找到每个 regime 的最佳模型
        if X_val is not None and y_val is not None and X_val.shape[1] >= 3:
            regimes = self._detect_regime(X_val)
            unique_regimes = set(regimes)
            for r in unique_regimes:
                mask = np.array(regimes) == r
                if mask.sum() < 10:
                    continue
                best_idx = 0
                best_score = -float("inf")
                for i, m in enumerate(self.models):
                    preds = m.predict(X_val[mask])
                    corr = np.corrcoef(preds, np.ravel(y_val)[mask])[0, 1]
                    if corr > best_score:
                        best_score = corr
                        best_idx = i
                self.regime_map[r] = best_idx

        self._fitted = True
        return {"regime_map": {str(k): v for k, v in self.regime_map.items()}}

    def _detect_regime(self, X: np.ndarray) -> list[tuple]:
        """检测每行的 market regime"""
        regimes = []
        for row in X:
            trend = 0
            # 尝试用均线距离判断 (假设前几维是 trend 特征)
            if abs(row[0]) > 0.05:
                trend = 1 if row[0] > 0 else -1

            vol = 0
            if len(row) > 5 and abs(row[5]) > 0.5:
                vol = 1 if row[5] > 0 else -1

            volume = 0
            if len(row) > 10 and abs(row[10]) > 0.5:
                volume = 1 if row[10] > 0 else -1

            regimes.append((trend, vol, volume))
        return regimes

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.models:
            return np.zeros(len(X))

        if not self.regime_map:
            # 未学习到 regime 映射, fallback 到简单平均
            preds = np.column_stack([m.predict(X) for m in self.models])
            return np.mean(preds, axis=1)

        regimes = self._detect_regime(X)
        result = np.zeros(len(X))
        default_model = 0
        for i, r in enumerate(regimes):
            model_idx = self.regime_map.get(r, default_model)
            result[i] = self.models[model_idx].predict(X[i:i + 1])[0]

        return result
