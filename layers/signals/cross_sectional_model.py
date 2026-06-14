from __future__ import annotations

"""
截面多因子选股模型 — LightGBM Ranker & Regressor

核心设计:
  - 输入: 截面因子面板 (date, code) × 因子
  - 标签: 未来 N 日截面排名百分位 [0, 1]
  - 模型: LightGBM / Ridge 回归
  - 训练: 时间序列切分，不 shuffle

与单股票模型的核心区别:
  - 训练数据是所有股票 × 所有日期的 panel
  - 标签是横截面相对强弱，而非单股票涨跌幅
  - 推理时输出每只股票的预测排名
"""

from typing import Optional

import numpy as np
import pandas as pd


class CrossSectionalRanker:
    """截面多因子选股模型。

    输入为 MultiIndex(date, code) 的截面因子面板，训练目标通常是未来收益在
    每个交易日横截面上的排名分位数。模型只负责预测相对强弱，不直接处理交易执行。
    """

    SUPPORTED_MODELS = ("lightgbm", "ridge")

    def __init__(
        self,
        model_type: str = "lightgbm",
        label_horizon: int = 5,
        model_params: Optional[dict] = None,
        verbose: bool = True,
    ):
        if model_type not in self.SUPPORTED_MODELS:
            raise ValueError(f"Unsupported cross-sectional model: {model_type}")
        self.model_type = model_type
        self.label_horizon = label_horizon
        self.verbose = verbose
        self._model = None
        self._feature_names: list[str] = []
        self._scaler = None
        self._train_dates = 0
        self.model_params = model_params or self._default_params(model_type)

    def _default_params(self, model_type: str) -> dict:
        """按模型类型返回默认参数。"""
        if model_type == "lightgbm":
            return {
                "n_estimators": 300,
                "max_depth": 5,
                "num_leaves": 31,
                "learning_rate": 0.05,
                "reg_alpha": 0.5,
                "reg_lambda": 0.5,
                "min_child_samples": 50,
                "subsample": 0.8,
                "colsample_bytree": 0.6,
                "random_state": 42,
                "n_jobs": -1,
                "verbose": -1,
            }
        if model_type == "ridge":
            return {"alpha": 1.0, "random_state": 42}
        return {}

    def fit(
        self,
        panel: pd.DataFrame,
        factor_cols: list[str],
        label_col: str = "label_rank",
        val_panel: Optional[pd.DataFrame] = None,
    ) -> dict:
        """训练截面排名模型。

        Args:
            panel: 训练面板，索引为 (date, code)。
            factor_cols: 用于训练的因子列。
            label_col: 训练标签列，默认使用截面排名分位数。
            val_panel: 可选验证集，LightGBM 会用于 early stopping。
        """
        self._feature_names = [col for col in factor_cols if col in panel.columns]
        X, y, _ = self._extract_xy(panel, self._feature_names, label_col)
        if len(X) == 0:
            raise ValueError("No valid training rows for cross-sectional model")

        if self.verbose:
            print(f"[CrossSectionalRanker] training rows={len(X)}, features={len(self._feature_names)}")

        from sklearn.preprocessing import StandardScaler

        self._scaler = StandardScaler().fit(X)
        X_s = self._scaler.transform(X)

        if self.model_type == "lightgbm":
            metrics = self._fit_lgb(X_s, y, val_panel, label_col)
        elif self.model_type == "ridge":
            metrics = self._fit_ridge(X_s, y)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        self._train_dates = len(panel.index.get_level_values("date").unique())
        return metrics

    def _fit_lgb(self, X_s: np.ndarray, y: np.ndarray, val_panel: Optional[pd.DataFrame], label_col: str) -> dict:
        import lightgbm as lgb

        eval_set = None
        if val_panel is not None:
            X_v, y_v, _ = self._extract_xy(val_panel, self._feature_names, label_col)
            if len(X_v) > 0:
                X_v_s = self._scaler.transform(X_v)
                eval_set = [(X_v_s, y_v)]

        self._model = lgb.LGBMRegressor(**self.model_params)
        fit_kwargs = {"eval_set": eval_set} if eval_set else {}
        if eval_set:
            fit_kwargs["callbacks"] = [lgb.early_stopping(50)]
        self._model.fit(X_s, y, **fit_kwargs)

        train_pred = self._model.predict(X_s)
        from sklearn.metrics import r2_score

        return {
            "r2": float(r2_score(y, train_pred)),
            "spearman": _safe_corr(y, train_pred),
            "n_train": int(len(y)),
            "importance": self.feature_importance(),
        }

    def _fit_ridge(self, X_s: np.ndarray, y: np.ndarray) -> dict:
        from sklearn.linear_model import Ridge

        self._model = Ridge(**self.model_params)
        self._model.fit(X_s, y)
        train_pred = self._model.predict(X_s)
        return {
            "r2": float(self._model.score(X_s, y)),
            "spearman": _safe_corr(y, train_pred),
            "n_train": int(len(y)),
            "importance": self.feature_importance(),
        }

    def predict(self, panel: pd.DataFrame, factor_cols: Optional[list[str]] = None) -> pd.Series:
        """对截面面板输出模型预测得分。"""
        if self._model is None:
            raise RuntimeError("Cross-sectional model is not trained")

        cols = factor_cols or self._feature_names
        X, _, idx = self._extract_xy(panel, cols)
        if len(X) == 0:
            return pd.Series(dtype=float, name="prediction")
        if self._scaler is not None:
            X = self._scaler.transform(X)
        preds = self._model.predict(X)
        return pd.Series(preds, index=idx, name="prediction")

    def predict_rank(self, panel: pd.DataFrame, factor_cols: Optional[list[str]] = None) -> pd.DataFrame:
        """预测并按每日横截面计算排名分位数。"""
        result = panel.copy()
        prediction = self.predict(result, factor_cols)
        result["prediction"] = prediction
        result["rank_pct"] = np.nan

        for dt in result.index.get_level_values("date").unique():
            scores = result.loc[dt, "prediction"].dropna()
            if scores.empty:
                continue
            ranks = scores.rank(method="average", ascending=True)
            rank_pct = (ranks - 1) / max(len(ranks) - 1, 1)
            for code, value in rank_pct.items():
                result.loc[(dt, code), "rank_pct"] = value

        return result

    def feature_importance(self) -> pd.DataFrame:
        """返回模型的因子重要性。"""
        if self._model is None:
            return pd.DataFrame(columns=["feature", "importance"])
        if self.model_type == "lightgbm" and hasattr(self._model, "feature_importances_"):
            importance = self._model.feature_importances_
        elif self.model_type == "ridge" and hasattr(self._model, "coef_"):
            importance = np.abs(self._model.coef_)
        else:
            return pd.DataFrame(columns=["feature", "importance"])
        return pd.DataFrame({
            "feature": self._feature_names,
            "importance": importance,
        }).sort_values("importance", ascending=False)

    def _extract_xy(
        self,
        panel: pd.DataFrame,
        factor_cols: list[str],
        label_col: Optional[str] = None,
    ) -> tuple[np.ndarray, Optional[np.ndarray], pd.Index]:
        """从面板中提取特征矩阵、标签和对应索引。"""
        available = [col for col in factor_cols if col in panel.columns]
        cols = available + ([label_col] if label_col else [])
        data = panel[cols].replace([np.inf, -np.inf], np.nan).dropna(how="any")
        X = data[available].values.astype(np.float32)
        y = data[label_col].values.astype(np.float32) if label_col else None
        return X, y, data.index

    def summary(self) -> dict:
        """返回模型摘要，供 API 和前端展示使用。"""
        return {
            "model_type": self.model_type,
            "label_horizon": self.label_horizon,
            "n_features": len(self._feature_names),
            "train_dates": self._train_dates,
            "params": self.model_params,
        }


def _safe_corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2 or np.std(y_true) < 1e-12 or np.std(y_pred) < 1e-12:
        return 0.0
    corr = np.corrcoef(y_true, y_pred)[0, 1]
    return float(corr) if np.isfinite(corr) else 0.0
