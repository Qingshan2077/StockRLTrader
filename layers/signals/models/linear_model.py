"""
线性模型: Linear / Ridge / Lasso
"""
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score, mean_squared_error
from layers.signals.base_model import BaseAlphaModel


class LinearModel(BaseAlphaModel):
    """Linear / Ridge / Lasso 模型"""

    def __init__(self, name: str = "ridge", config: dict = None):
        super().__init__(name=name, config=config)
        self._method = name

    def fit(self, X_train, y_train, X_val=None, y_val=None) -> dict:
        y_train = np.ravel(y_train)

        if self._method == "ridge":
            alpha = self.config.get("alpha", 1.0)
            self.model = Ridge(alpha=alpha)
        elif self._method == "lasso":
            alpha = self.config.get("alpha", 0.001)
            self.model = Lasso(alpha=alpha, max_iter=5000)
        else:
            self.model = LinearRegression()

        self.model.fit(X_train, y_train)
        self._fitted = True

        metrics = {}
        if X_val is not None and y_val is not None:
            y_val = np.ravel(y_val)
            preds = self.model.predict(X_val)
            metrics["val_r2"] = float(r2_score(y_val, preds))
            metrics["val_rmse"] = float(np.sqrt(mean_squared_error(y_val, preds)))

        return metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def get_importance(self) -> dict | None:
        if self._method == "ridge":
            return {"coef": self.model.coef_.tolist()}
        return None
