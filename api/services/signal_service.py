from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
from sklearn.preprocessing import StandardScaler

from api.core.config import get_config
from api.services.data_service import DataService
from api.services.runtime_store import SignalRun, runtime_store
from api.services.serialization import json_safe
from layers.features import FeatureCache, FeaturePipeline, FeatureRegistry, LabelBuilder
from layers.signals.models.lightgbm_model import LightGBMModel
from layers.signals.models.linear_model import LinearModel
from layers.signals.models.mlp_model import MLPSignalModel
from layers.signals.models.xgboost_model import XGBoostModel


PRICE_COLS = {"Open", "High", "Low", "Close", "Volume", "Adj Close"}


class SignalService:
    def __init__(self) -> None:
        self.data_service = DataService()
        self.data_dir = get_config("system.data_dir", "stock_data")

    def list_runs(self) -> list[dict[str, Any]]:
        rows = []
        for run in runtime_store.list_signal_runs():
            rows.append({
                "run_id": run.run_id,
                "ticker": run.ticker,
                "models": run.models,
                "created_at": run.created_at,
                "metrics": json_safe(run.metrics),
            })
        return rows

    def train(
        self,
        ticker: str,
        models: list[str],
        horizon: int = 5,
        label_type: str = "regression",
        use_cache: bool = True,
    ) -> dict[str, Any]:
        df, _ = self.data_service.load_candles(ticker, processed=False, limit=None)
        registry = FeatureRegistry()
        cache = FeatureCache(data_dir=self.data_dir)
        pipeline = FeaturePipeline(registry=registry, cache=cache, benchmark_data=None)
        features = pipeline.build_features(df, ticker=ticker.upper(), use_cache=use_cache)
        label_builder = LabelBuilder(horizon=horizon, label_type=label_type)
        labels = label_builder.build_labels(features)
        features, labels = label_builder.align(features, labels)

        feature_cols = [c for c in features.columns if c not in PRICE_COLS]
        X_all = features[feature_cols].values.astype(np.float32)
        y_all = labels.values.astype(np.float32)
        n = len(X_all)
        train_end = int(n * 0.6)
        val_end = int(n * 0.8)

        X_train, y_train = X_all[:train_end], y_all[:train_end]
        X_val, y_val = X_all[train_end:val_end], y_all[train_end:val_end]
        X_test, y_test = X_all[val_end:], y_all[val_end:]

        scaler = StandardScaler().fit(X_train)
        X_train_s = scaler.transform(X_train)
        X_val_s = scaler.transform(X_val)
        X_test_s = scaler.transform(X_test)

        metrics: dict[str, Any] = {}
        predictions: dict[str, np.ndarray] = {}
        importance: dict[str, Any] = {}
        fitted_models = []

        for model_name in models:
            model = self._create_model(model_name)
            model_metrics = model.fit(X_train_s, y_train, X_val_s, y_val)
            preds = model.predict(X_test_s)
            metrics[model_name] = json_safe(model_metrics)
            predictions[model_name] = preds
            fitted_models.append(model_name)
            try:
                imp = model.get_importance()
                if imp:
                    importance[model_name] = json_safe(imp)
            except Exception:
                importance[model_name] = {}

        if predictions:
            ensemble_signal = np.mean(np.vstack(list(predictions.values())), axis=0)
        else:
            ensemble_signal = np.zeros(len(X_test), dtype=np.float64)

        run_id = runtime_store.new_id("sig")
        run = SignalRun(
            run_id=run_id,
            ticker=ticker.upper(),
            models=fitted_models,
            created_at=datetime.now().isoformat(),
            metrics=metrics,
            predictions=predictions,
            feature_importance=importance,
            test_features=features.iloc[val_end:].copy(),
            test_labels=y_test,
            ensemble_signal=ensemble_signal,
        )
        runtime_store.save_signal_run(run)

        return {
            "run_id": run_id,
            "ticker": ticker.upper(),
            "models": fitted_models,
            "metrics": json_safe(metrics),
            "feature_importance": json_safe(importance),
            "test_rows": int(len(X_test)),
            "feature_count": len(feature_cols),
            "cache_hit_rate": float(cache.hit_rate),
        }

    def build_ensemble(self, method: str, model_run_ids: list[str]) -> dict[str, Any]:
        if not model_run_ids:
            raise ValueError("model_run_ids is required")
        source = runtime_store.get_signal_run(model_run_ids[0])
        if source.predictions:
            source.ensemble_signal = np.mean(np.vstack(list(source.predictions.values())), axis=0)
        return {
            "run_id": source.run_id,
            "method": method,
            "ticker": source.ticker,
            "signal_rows": int(len(source.ensemble_signal)) if source.ensemble_signal is not None else 0,
        }

    def _create_model(self, model_name: str):
        name = model_name.lower()
        config = get_config(name, {})
        if name == "lightgbm":
            return LightGBMModel(name="lightgbm", config=config)
        if name == "xgboost":
            return XGBoostModel(name="xgboost", config=config)
        if name in {"ridge", "lasso"}:
            return LinearModel(name=name, config=config)
        if name == "mlp":
            return MLPSignalModel(name="mlp", config=config)
        raise ValueError(f"Unsupported model: {model_name}")
