from __future__ import annotations

import pandas as pd

from api.core.config import get_config
from api.services.data_service import DataService
from api.services.serialization import dataframe_records
from layers.features import FeatureCache, FeaturePipeline, FeatureRegistry, LabelBuilder


PRICE_COLS = {"Open", "High", "Low", "Close", "Volume", "Adj Close"}


class FeatureService:
    def __init__(self):
        self.data_service = DataService()
        self.data_dir = get_config("system.data_dir", "stock_data")

    def build_summary(
        self,
        ticker: str,
        use_cache: bool = True,
        horizon: int = 5,
        label_type: str = "regression",
    ) -> dict:
        df, _ = self.data_service.load_candles(ticker, processed=False, limit=None)
        benchmark = None
        try:
            benchmark, _ = self.data_service.load_candles(get_config("benchmark", "SPY"), processed=False, limit=None)
        except Exception:
            benchmark = None

        registry = FeatureRegistry()
        cache = FeatureCache(data_dir=self.data_dir)
        pipeline = FeaturePipeline(registry=registry, cache=cache, benchmark_data=benchmark)
        features = pipeline.build_features(df, ticker=ticker.upper(), use_cache=use_cache)

        labels = pd.Series(dtype=float)
        try:
            label_builder = LabelBuilder(horizon=horizon, label_type=label_type)
            labels = label_builder.build_labels(features)
            features, labels = label_builder.align(features, labels)
        except Exception:
            pass

        feature_cols = [c for c in features.columns if c not in PRICE_COLS]
        return {
            "ticker": ticker.upper(),
            "row_count": int(len(features)),
            "feature_count": len(feature_cols),
            "label_count": int(len(labels)),
            "cache_hit_rate": float(cache.hit_rate),
            "groups": registry.list_groups(),
            "columns": feature_cols,
            "sample": dataframe_records(features[feature_cols].tail(20), limit=20),
        }

    def analyze_single_asset(
        self,
        ticker: str,
        use_cache: bool = True,
        horizon: int = 5,
        label_type: str = "regression",
        top_n: int = 30,
    ) -> dict:
        df, _ = self.data_service.load_candles(ticker, processed=False, limit=None)
        registry = FeatureRegistry()
        cache = FeatureCache(data_dir=self.data_dir)
        pipeline = FeaturePipeline(registry=registry, cache=cache, benchmark_data=None)
        features = pipeline.build_features(df, ticker=ticker.upper(), use_cache=use_cache)
        label_builder = LabelBuilder(horizon=horizon, label_type=label_type)
        labels = label_builder.build_labels(features)
        features, labels = label_builder.align(features, labels)

        feature_cols = [c for c in features.columns if c not in PRICE_COLS]
        corr_rows = []
        for col in feature_cols:
            try:
                corr = features[col].corr(labels)
                if pd.notna(corr):
                    corr_rows.append({"feature": col, "target_corr": float(corr), "abs_corr": float(abs(corr))})
            except Exception:
                continue
        corr_rows.sort(key=lambda row: row["abs_corr"], reverse=True)
        top = corr_rows[:top_n]
        top_cols = [row["feature"] for row in top[: min(12, len(top))]]
        corr_matrix = []
        if len(top_cols) >= 2:
            matrix = features[top_cols].corr().reset_index().rename(columns={"index": "feature"})
            corr_matrix = dataframe_records(matrix)

        return {
            "ticker": ticker.upper(),
            "feature_count": len(feature_cols),
            "label_count": int(len(labels)),
            "top_target_correlations": top,
            "correlation_matrix": corr_matrix,
            "cache_hit_rate": float(cache.hit_rate),
        }
