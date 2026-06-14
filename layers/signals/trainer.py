"""
统一模型训练器 — 训练循环 + 指标记录
"""
import numpy as np
import time
from typing import Optional


class ModelTrainer:
    """统一 Alpha 模型训练器"""

    def __init__(self, model: "BaseAlphaModel", early_stopping_rounds: int = 20,
                 verbose: bool = True):
        self.model = model
        self.early_stopping_rounds = early_stopping_rounds
        self.verbose = verbose
        self.train_history: dict[str, list] = {}
        self.best_metric: Optional[float] = None
        self.training_time: float = 0.0

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray = None, y_val: np.ndarray = None) -> dict:
        """
        训练模型并返回评估指标

        Returns:
            dict: metrics dict + training_time + model_name
        """
        t0 = time.time()

        metrics = self.model.fit(X_train, y_train, X_val, y_val)
        self.training_time = time.time() - t0

        result = {
            "model_name": self.model.name,
            "training_time": self.training_time,
            "fitted": self.model.is_fitted,
            **metrics,
        }

        if self.verbose:
            from sklearn.metrics import r2_score
            preds = self.model.predict(X_val if X_val is not None else X_train[-100:])
            y_true = y_val if y_val is not None else y_train[-100:]
            r2 = r2_score(y_true, preds)
            print(f"[{self.model.name}] trained in {self.training_time:.1f}s, "
                  f"R²={r2:.4f}")

        return result

    def cross_validate(self, X: np.ndarray, y: np.ndarray,
                       n_splits: int = 5) -> list[dict]:
        """时序交叉验证 (无 shuffle)"""
        results = []
        n = len(X)
        fold_size = n // (n_splits + 1)

        for i in range(n_splits):
            # 时序切分: 前面 folds 做 train, 当前 fold 做 val
            train_end = (i + 1) * fold_size
            val_end = train_end + fold_size

            X_tr = X[:train_end]
            y_tr = y[:train_end]
            X_v = X[train_end:val_end]
            y_v = y[train_end:val_end]

            metrics = self.model.fit(X_tr, y_tr, X_v, y_v)
            results.append(metrics)

            if self.verbose:
                print(f"  Fold {i+1}/{n_splits}: {metrics}")

        return results
