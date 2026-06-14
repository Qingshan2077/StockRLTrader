"""
LightGBM 模型
"""
import numpy as np
import lightgbm as lgb
from sklearn.metrics import r2_score, mean_squared_error
from layers.signals.base_model import BaseAlphaModel


class LightGBMModel(BaseAlphaModel):
    """LightGBM Alpha 模型"""

    def __init__(self, name: str = "lightgbm", config: dict = None):
        super().__init__(name=name, config=config)

    def fit(self, X_train, y_train, X_val=None, y_val=None) -> dict:
        params = {
            "objective": "regression",
            "metric": "rmse",
            "boosting_type": "gbdt",
            "verbosity": -1,
            "n_estimators": self.config.get("n_estimators", 200),
            "learning_rate": self.config.get("learning_rate", 0.05),
            "max_depth": self.config.get("max_depth", 6),
            "num_leaves": self.config.get("num_leaves", 31),
            "min_child_samples": self.config.get("min_child_samples", 20),
            "subsample": self.config.get("subsample", 0.8),
            "colsample_bytree": self.config.get("colsample_bytree", 0.8),
            "reg_alpha": self.config.get("reg_alpha", 0.1),
            "reg_lambda": self.config.get("reg_lambda", 0.1),
        }

        y_train = np.ravel(y_train)
        train_data = lgb.Dataset(X_train, label=y_train)

        valid_sets = []
        if X_val is not None and y_val is not None:
            y_val = np.ravel(y_val)
            valid_sets = [lgb.Dataset(X_val, label=y_val)]

        self.model = lgb.train(
            params, train_data,
            valid_sets=valid_sets or None,
            num_boost_round=params["n_estimators"],
        )
        self._fitted = True

        metrics = {}
        if X_val is not None and y_val is not None:
            preds = self.model.predict(X_val)
            metrics["val_r2"] = float(r2_score(y_val, preds))
            metrics["val_rmse"] = float(np.sqrt(mean_squared_error(y_val, preds)))

        return metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def get_importance(self) -> dict | None:
        if self.model is None:
            return None
        imp = self.model.feature_importance(importance_type="gain")
        return {
            "importance_gain": imp.tolist(),
            "importance_split": self.model.feature_importance(
                importance_type="split").tolist(),
        }
