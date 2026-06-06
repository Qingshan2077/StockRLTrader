"""
XGBoost 模型
"""
import numpy as np
import xgboost as xgb
from sklearn.metrics import r2_score, mean_squared_error
from layers.signals.base_model import BaseAlphaModel


class XGBoostModel(BaseAlphaModel):
    """XGBoost Alpha 模型"""

    def __init__(self, name: str = "xgboost", config: dict = None):
        super().__init__(name=name, config=config)

    def fit(self, X_train, y_train, X_val=None, y_val=None) -> dict:
        params = {
            "objective": "reg:squarederror",
            "learning_rate": self.config.get("learning_rate", 0.05),
            "max_depth": self.config.get("max_depth", 5),
            "subsample": self.config.get("subsample", 0.8),
            "colsample_bytree": self.config.get("colsample_bytree", 0.8),
            "reg_alpha": self.config.get("reg_alpha", 0.1),
            "reg_lambda": self.config.get("reg_lambda", 0.1),
            "verbosity": 0,
        }

        y_train = np.ravel(y_train)
        dtrain = xgb.DMatrix(X_train, label=y_train)

        eval_list = []
        if X_val is not None and y_val is not None:
            y_val = np.ravel(y_val)
            dval = xgb.DMatrix(X_val, label=y_val)
            eval_list = [(dtrain, "train"), (dval, "val")]

        self.model = xgb.train(
            params, dtrain,
            num_boost_round=self.config.get("n_estimators", 200),
            evals=eval_list,
            verbose_eval=False,
        )
        self._fitted = True

        metrics = {}
        if X_val is not None and y_val is not None:
            preds = self.model.predict(xgb.DMatrix(X_val))
            metrics["val_r2"] = float(r2_score(y_val, preds))
            metrics["val_rmse"] = float(np.sqrt(mean_squared_error(y_val, preds)))

        return metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(xgb.DMatrix(X))

    def get_importance(self) -> dict | None:
        if self.model is None:
            return None
        importance = self.model.get_score(importance_type="gain")
        return {"importance_gain": importance}
