from __future__ import annotations

from datetime import datetime
import pickle
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from api.core.config import get_config
from api.core.paths import resolve_project_path
from api.services.data_service import DataService
from api.services.serialization import dataframe_records, json_safe
from layers.backtest.cross_sectional_backtest import CrossSectionalBacktest
from layers.features import FeatureCache, FeaturePipeline, FeatureRegistry
from layers.features.cross_sectional import process_panel_factors
from layers.signals.cross_sectional_model import CrossSectionalRanker


PRICE_COLS = {"Open", "High", "Low", "Close", "Volume", "Adj Close", "Amount", "Turnover"}
LABEL_COLS = {"future_return", "label_1d", "label_rank"}
META_COLS = {"sector"}


class CrossSectionService:
    """截面多因子服务。

    这里复用 layers/features 下的完整 FeaturePipeline，并用
    CrossSectionalRanker 训练/加载模型，避免 API 层维护一套手工因子权重。
    """

    def __init__(self) -> None:
        self.data_service = DataService()
        self.data_dir = get_config("system.data_dir", "stock_data")
        self.results_dir = resolve_project_path(self.data_dir) / "cross_sectional_results"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.panel_path = self.results_dir / "panel.pkl"
        self.model_path = self.results_dir / "model.pkl"

    def train(
        self,
        limit: int = 100,
        model_type: str = "lightgbm",
        horizon: int = 5,
        force_rebuild: bool = False,
    ) -> dict[str, Any]:
        """训练截面模型，并保存到 stock_data/cross_sectional_results/model.pkl。"""
        if model_type not in CrossSectionalRanker.SUPPORTED_MODELS:
            raise ValueError(f"Unsupported cross-section model type: {model_type}")
        panel = self._build_local_panel(limit=limit, horizon=horizon, require_label=True, force_rebuild=force_rebuild)
        if panel.empty:
            raise ValueError("No local panel data is available for cross-section training")

        factor_cols = self._factor_columns(panel)
        if not factor_cols:
            raise ValueError("No factor columns are available for cross-section training")

        train, val, test = self._time_split(panel)
        model = CrossSectionalRanker(model_type=model_type, label_horizon=horizon, verbose=False)
        metrics = model.fit(train, factor_cols, label_col="label_rank", val_panel=val)
        importance = metrics.pop("importance", pd.DataFrame())

        payload = {
            "model": model,
            "factor_cols": factor_cols,
            "horizon": horizon,
            "model_type": model_type,
            "trained_at": datetime.now().isoformat(),
            "metrics": metrics,
            "importance": importance,
        }
        with self.model_path.open("wb") as fh:
            pickle.dump(payload, fh)

        return {
            "model_status": "trained",
            "model_path": str(self.model_path),
            "model_type": model_type,
            "horizon": horizon,
            "rows": int(len(panel)),
            "train_rows": int(len(train)),
            "val_rows": int(len(val)),
            "test_rows": int(len(test)),
            "feature_count": len(factor_cols),
            "metrics": json_safe(metrics),
            "importance": dataframe_records(importance, limit=50),
        }

    def local_rank(self, limit: int = 100) -> dict[str, Any]:
        model_bundle = self._load_model()
        if model_bundle is None:
            return {
                "count": 0,
                "records": [],
                "model_status": "missing",
                "message": "Train cross-section model first.",
            }

        panel = self._build_local_panel(
            limit=limit,
            horizon=int(model_bundle.get("horizon", 5)),
            require_label=False,
        )
        if panel.empty:
            return {"count": 0, "records": [], "model_status": "ready", "message": "No local panel data."}

        scored = self._predict_with_model(panel, model_bundle)
        scored = scored.dropna(subset=["rank_pct"])
        if scored.empty:
            return {"count": 0, "records": [], "model_status": "ready", "message": "No valid model predictions."}

        last_date = scored.index.get_level_values("date").max()
        latest = scored.loc[last_date].copy().sort_values("rank_pct", ascending=False)
        latest["code"] = latest.index
        latest["date"] = str(last_date)
        columns = self._display_columns(latest)
        return {
            "date": str(last_date),
            "count": len(latest),
            "model_status": "ready",
            "model_type": model_bundle.get("model_type"),
            "feature_count": len(model_bundle.get("factor_cols", [])),
            "columns": columns,
            "records": dataframe_records(latest.reset_index(drop=True)[columns], limit=limit),
        }

    def local_backtest(self, limit: int = 100, long_n: int = 10, short_n: int = 10) -> dict[str, Any]:
        model_bundle = self._load_model()
        if model_bundle is None:
            return {
                "metrics": {},
                "nav": [],
                "benchmark": [],
                "turnover": [],
                "model_status": "missing",
                "message": "Train cross-section model first.",
            }

        panel = self._build_local_panel(
            limit=limit,
            horizon=int(model_bundle.get("horizon", 5)),
            require_label=True,
        )
        if panel.empty:
            return {"metrics": {}, "nav": [], "turnover": [], "model_status": "ready", "message": "No labeled panel data."}

        scored = self._predict_with_model(panel, model_bundle).dropna(subset=["rank_pct", "label_1d"])
        if scored.empty:
            return {"metrics": {}, "nav": [], "turnover": [], "model_status": "ready", "message": "No valid backtest rows."}

        engine = CrossSectionalBacktest(long_n=long_n, short_n=short_n, verbose=False)
        result = engine.run(scored, rank_col="rank_pct", return_col="label_1d", sector_col="sector")
        return {
            "model_status": "ready",
            "metrics": json_safe(result.metrics),
            "nav": dataframe_records(result.nav.to_frame("nav")),
            "benchmark": dataframe_records(result.benchmark.to_frame("benchmark")),
            "turnover": dataframe_records(result.turnover.to_frame("turnover")),
        }

    def factor_report(self, limit: int = 100) -> dict[str, Any]:
        """对当前股票池的全部可用因子计算 RankIC。"""
        panel = self._build_local_panel(limit=limit, require_label=True)
        if panel.empty:
            return {"rank_ic": [], "factor_columns": [], "message": "No labeled panel data."}

        factor_cols = self._factor_columns(panel)
        rows = []
        for factor in factor_cols:
            daily_ic = []
            for dt in panel.index.get_level_values("date").unique().sort_values():
                chunk = panel.loc[dt]
                if factor not in chunk.columns:
                    continue
                valid = chunk[[factor, "future_return"]].replace([np.inf, -np.inf], np.nan).dropna()
                if len(valid) < 5 or valid[factor].std() < 1e-12 or valid["future_return"].std() < 1e-12:
                    continue
                rho, _ = spearmanr(valid[factor], valid["future_return"])
                if pd.notna(rho):
                    daily_ic.append(float(rho))
            if daily_ic:
                arr = np.asarray(daily_ic, dtype=float)
                rows.append({
                    "factor": factor,
                    "rank_ic_mean": float(arr.mean()),
                    "rank_ic_std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
                    "rank_ic_ir": float(arr.mean() / arr.std(ddof=1)) if len(arr) > 1 and arr.std(ddof=1) > 1e-12 else 0.0,
                    "positive_pct": float((arr > 0).mean()),
                    "observations": int(len(arr)),
                })
        rows.sort(key=lambda row: abs(row["rank_ic_mean"]), reverse=True)
        return {"rank_ic": rows, "factor_columns": factor_cols, "factor_count": len(factor_cols)}

    def _build_local_panel(
        self,
        limit: int = 100,
        horizon: int = 5,
        require_label: bool = True,
        force_rebuild: bool = False,
    ) -> pd.DataFrame:
        if not force_rebuild:
            cached = self._load_panel()
            if cached is not None and not cached.empty:
                return self._filter_panel(cached, require_label=require_label)

        rows = []
        tickers = self.data_service.list_tickers()[:limit]
        feature_cache = FeatureCache(data_dir=self.data_dir)

        for info in tickers:
            try:
                df, _ = self.data_service.load_candles(info.ticker, processed=False, limit=None)
            except Exception:
                continue
            if df.empty or "Close" not in df.columns:
                continue

            registry = FeatureRegistry()
            pipeline = FeaturePipeline(registry=registry, cache=feature_cache, benchmark_data=None)
            try:
                features = pipeline.build_features(df, ticker=info.ticker.upper(), use_cache=True)
            except Exception:
                continue
            if features is None or features.empty or "Close" not in features.columns:
                continue

            code = info.ticker.upper()
            frame = features.copy()
            frame["code"] = code
            frame["sector"] = self._sector_for(code)
            frame.index.name = "date"
            rows.append(frame.reset_index().set_index(["date", "code"]))

        if not rows:
            return pd.DataFrame()

        panel = pd.concat(rows).sort_index()
        panel = self._add_labels(panel, horizon=horizon)
        factor_cols = self._factor_columns(panel)
        if factor_cols:
            panel = process_panel_factors(
                panel,
                factor_cols,
                sector_col="sector",
                do_sector_neutral=True,
                do_market_neutral=False,
                verbose=False,
            )
        panel = panel.replace([np.inf, -np.inf], np.nan)
        self._save_panel(panel)
        return self._filter_panel(panel, require_label=require_label)

    def _add_labels(self, panel: pd.DataFrame, horizon: int) -> pd.DataFrame:
        result = panel.copy()
        result["future_return"] = np.nan
        result["label_1d"] = np.nan
        for code in result.index.get_level_values("code").unique():
            mask = result.index.get_level_values("code") == code
            close = result.loc[mask, "Close"].sort_index()
            result.loc[mask, "future_return"] = (close.shift(-horizon) / close - 1).values
            result.loc[mask, "label_1d"] = (close.shift(-1) / close - 1).values

        result["label_rank"] = result.groupby(level="date")["future_return"].rank(pct=True)
        return result

    def _predict_with_model(self, panel: pd.DataFrame, model_bundle: dict[str, Any]) -> pd.DataFrame:
        model = model_bundle["model"]
        factor_cols = [col for col in model_bundle.get("factor_cols", []) if col in panel.columns]
        if not factor_cols:
            raise ValueError("Saved cross-section model has no usable factor columns")
        return model.predict_rank(panel, factor_cols)

    def _factor_columns(self, panel: pd.DataFrame) -> list[str]:
        excluded = PRICE_COLS | LABEL_COLS | META_COLS
        return [
            col for col in panel.columns
            if col not in excluded
            and not col.endswith("_raw")
            and pd.api.types.is_numeric_dtype(panel[col])
        ]

    def _display_columns(self, frame: pd.DataFrame) -> list[str]:
        preferred = ["date", "code", "prediction", "rank_pct", "future_return", "label_1d"]
        top_factors = [col for col in self._factor_columns(frame)[:6] if col not in preferred]
        return [col for col in preferred + top_factors if col in frame.columns]

    def _time_split(self, panel: pd.DataFrame, train_ratio: float = 0.6, val_ratio: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        dates = panel.index.get_level_values("date").unique().sort_values()
        train_end = int(len(dates) * train_ratio)
        val_end = int(len(dates) * (train_ratio + val_ratio))
        train_dates = dates[:train_end]
        val_dates = dates[train_end:val_end]
        test_dates = dates[val_end:]
        return (
            panel.loc[panel.index.get_level_values("date").isin(train_dates)],
            panel.loc[panel.index.get_level_values("date").isin(val_dates)],
            panel.loc[panel.index.get_level_values("date").isin(test_dates)],
        )

    def _filter_panel(self, panel: pd.DataFrame, require_label: bool) -> pd.DataFrame:
        if require_label:
            return panel.dropna(subset=["future_return", "label_rank"])
        return panel

    def _load_model(self) -> dict[str, Any] | None:
        if not self.model_path.exists():
            return None
        with self.model_path.open("rb") as fh:
            return pickle.load(fh)

    def _load_panel(self) -> pd.DataFrame | None:
        if not self.panel_path.exists():
            return None
        try:
            return pd.read_parquet(self.panel_path)
        except Exception:
            try:
                return pd.read_pickle(self.panel_path)
            except Exception:
                return None

    def _save_panel(self, panel: pd.DataFrame) -> None:
        # 文件名沿用清单要求的 panel.pkl；内容优先用 parquet，便于快速加载。
        try:
            panel.to_parquet(self.panel_path)
        except Exception:
            panel.to_pickle(self.panel_path)

    def _sector_for(self, ticker: str) -> str:
        try:
            from layers.data.ashare_universe import get_sw_sector

            return get_sw_sector(self.data_service._a_share_numeric_symbol(ticker))
        except Exception:
            return "其他"
