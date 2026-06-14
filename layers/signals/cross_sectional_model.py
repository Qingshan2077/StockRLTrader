"""
截面多因子选股模型 — LightGBM Ranker & Regressor

核心设计:
  - 输入: 截面因子面板 (date, code) × 因子
  - 标签: 未来 N 日截面排名百分位 [0, 1]
  - 模型: LightGBM Lambdarank 或 回归
  - 训练: 时间序列切分 (不 shuffle)

与单股票模型的核心区别:
  - 训练数据是"所有股票 × 所有日期"的 panel
  - 标签是横截面相对强弱, 而非单股票涨跌幅
  - 推理时输出每只股票的预测排名

用法:
    model = CrossSectionalRanker()
    model.fit(panel_train, factor_cols, label_col)
    predictions = model.predict(panel_test, factor_cols)
"""

import numpy as np
import pandas as pd
from typing import Optional


class CrossSectionalRanker:
    """截面选股模型 — 预测横截面排名"""

    SUPPORTED_MODELS = ("lightgbm", "ridge")

    def __init__(
        self,
        model_type: str = "lightgbm",
        label_horizon: int = 5,
        model_params: Optional[dict] = None,
        verbose: bool = True,
    ):
        """
        Args:
            model_type: 'lightgbm' 或 'ridge'
            label_horizon: 标签窗口 (未来 N 日收益)
            model_params: sklearn/LightGBM 参数字典
            verbose: 打印训练信息
        """
        self.model_type = model_type
        self.label_horizon = label_horizon
        self.verbose = verbose
        self._model = None
        self._feature_names: list[str] = []
        self._scaler = None
        self._train_dates: int = 0

        # 默认参数
        if model_params is None:
            self.model_params = self._default_params(model_type)
        else:
            self.model_params = model_params

    def _default_params(self, model_type: str) -> dict:
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
        elif model_type == "ridge":
            return {
                "alpha": 1.0,
                "random_state": 42,
            }
        return {}

    def fit(
        self,
        panel: pd.DataFrame,
        factor_cols: list[str],
        label_col: str = "label_rank",
        val_panel: Optional[pd.DataFrame] = None,
    ) -> dict:
        """训练截面模型

        Args:
            panel: MultiIndex (date, code) 训练面板, 含因子 + label_col
            factor_cols: 特征列名
            label_col: 标签列 (截面排名百分位 [0,1] 或 未来收益)
            val_panel: 验证面板 (可选), 用于早停

        Returns:
            训练指标 dict
        """
        self._feature_names = factor_cols
        X, y, _ = self._extract_xy(panel, factor_cols, label_col)
        n = len(X)

        if self.verbose:
            print(f"  [截面模型] 训练数据: {n} 样本, {len(factor_cols)} 特征")

        # 标准化
        from sklearn.preprocessing import StandardScaler
        self._scaler = StandardScaler().fit(X)
        X_s = self._scaler.transform(X)

        if self.model_type == "lightgbm":
            metrics = self._fit_lgb(X_s, y, panel, val_panel, factor_cols, label_col)
        elif self.model_type == "ridge":
            metrics = self._fit_ridge(X_s, y)
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}")

        self._train_dates = len(panel.index.get_level_values("date").unique())
        return metrics

    def _fit_lgb(
        self, X_s, y, panel, val_panel, factor_cols, label_col
    ) -> dict:
        import lightgbm as lgb

        eval_set = None
        if val_panel is not None:
            X_v, y_v, _ = self._extract_xy(val_panel, factor_cols, label_col)
            if len(X_v) > 0:
                X_v_s = self._scaler.transform(X_v)
                eval_set = [(X_v_s, y_v)]

        self._model = lgb.LGBMRegressor(**self.model_params)
        self._model.fit(
            X_s, y,
            eval_set=eval_set,
            callbacks=[lgb.early_stopping(50)] if eval_set else None,
        )

        train_pred = self._model.predict(X_s)
        from sklearn.metrics import r2_score
        r2 = r2_score(y, train_pred)
        srcc = np.corrcoef(y, train_pred)[0, 1] if len(y) > 1 else 0.0

        if self.verbose:
            print(f"  [截面模型] LightGBM: R²={r2:.4f}, Spearman={srcc:.4f}")

        # 特征重要性
        imp = pd.DataFrame({
            "feature": self._feature_names,
            "importance": self._model.feature_importances_,
        }).sort_values("importance", ascending=False)

        return {"r2": r2, "spearman": srcc, "n_train": len(y), "importance": imp}

    def _fit_ridge(self, X_s, y) -> dict:
        from sklearn.linear_model import Ridge

        self._model = Ridge(**self.model_params)
        self._model.fit(X_s, y)
        train_pred = self._model.predict(X_s)
        r2 = self._model.score(X_s, y)
        srcc = np.corrcoef(y, train_pred)[0, 1] if len(y) > 1 else 0.0

        if self.verbose:
            print(f"  [截面模型] Ridge: R²={r2:.4f}, Spearman={srcc:.4f}")

        imp = pd.DataFrame({
            "feature": self._feature_names,
            "importance": np.abs(self._model.coef_),
        }).sort_values("importance", ascending=False)

        return {"r2": r2, "spearman": srcc, "n_train": len(y), "importance": imp}

    def predict(
        self,
        panel: pd.DataFrame,
        factor_cols: Optional[list[str]] = None,
        return_components: bool = False,
    ) -> pd.Series:
        """预测截面排名得分

        Args:
            panel: MultiIndex (date, code) 预测面板
            factor_cols: 特征列, 默认使用训练时的列
            return_components: 是否同时返回 rank_pct

        Returns:
            Series: index=(date, code), value=预测得分
        """
        if self._model is None:
            raise RuntimeError("模型尚未训练, 请先调用 fit()")

        cols = factor_cols or self._feature_names
        X, _, idx = self._extract_xy(panel, cols)

        if self._scaler is not None:
            X = self._scaler.transform(X)

        preds = self._model.predict(X)
        result = pd.Series(preds, index=idx, name="prediction")
        return result

    def predict_rank(
        self,
        panel: pd.DataFrame,
        factor_cols: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """预测并计算每日横截面排名百分位

        Returns:
            DataFrame: 含 prediction, rank_pct 列
        """
        pred = self.predict(panel, factor_cols)
        result = panel.copy()
        result["prediction"] = pred

        # 每日计算排名百分位
        dates = result.index.get_level_values("date").unique()
        for dt in dates:
            mask = result.index.get_level_values("date") == dt
            sub = result.loc[dt, "prediction"]
            rnk = sub.rank(method="average", ascending=True)
            result.loc[dt, "rank_pct"] = (rnk - 1) / max(len(rnk) - 1, 1)

        return result

    def feature_importance(self) -> pd.DataFrame:
        """返回特征重要性 DataFrame"""
        if self.model_type == "lightgbm" and self._model is not None:
            imp = self._model.feature_importances_
        elif self.model_type == "ridge" and self._model is not None:
            imp = np.abs(self._model.coef_)
        else:
            return pd.DataFrame()
        return pd.DataFrame({
            "feature": self._feature_names,
            "importance": imp,
        }).sort_values("importance", ascending=False)

    def _extract_xy(
        self,
        panel: pd.DataFrame,
        factor_cols: list[str],
        label_col: Optional[str] = None,
    ) -> tuple:
        """从面板提取 X, y, index"""
        avail = [c for c in factor_cols if c in panel.columns]
        if label_col:
            cols = avail + [label_col]
        else:
            cols = avail
        data = panel[cols].dropna(how="any")
        X = data[avail].values.astype(np.float32)
        y = data[label_col].values.astype(np.float32) if label_col else None
        return X, y, data.index

    def summary(self) -> dict:
        return {
            "model_type": self.model_type,
            "label_horizon": self.label_horizon,
            "n_features": len(self._feature_names),
            "train_dates": self._train_dates,
            "params": self.model_params,
        }
