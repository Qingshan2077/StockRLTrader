from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from api.services.data_service import DataService
from api.services.serialization import dataframe_records, json_safe
from layers.backtest.cross_sectional_backtest import CrossSectionalBacktest


class CrossSectionService:
    def __init__(self) -> None:
        self.data_service = DataService()

    def local_rank(self, limit: int = 100) -> dict[str, Any]:
        panel = self._build_local_panel(limit=limit, require_label=False)
        if panel.empty:
            return {"count": 0, "records": []}
        last_date = panel.index.get_level_values("date").max()
        latest = panel.loc[last_date].copy()
        latest = latest.sort_values("rank_pct", ascending=False)
        latest["code"] = latest.index
        latest["date"] = str(last_date)
        return {
            "date": str(last_date),
            "count": len(latest),
            "records": dataframe_records(latest.reset_index(drop=True), limit=limit),
        }

    def local_backtest(self, limit: int = 100, long_n: int = 10, short_n: int = 10) -> dict[str, Any]:
        panel = self._build_local_panel(limit=limit, require_label=True)
        if panel.empty:
            return {"metrics": {}, "nav": [], "turnover": []}
        engine = CrossSectionalBacktest(long_n=long_n, short_n=short_n, verbose=False)
        result = engine.run(panel, rank_col="rank_pct", return_col="future_return")
        return {
            "metrics": json_safe(result.metrics),
            "nav": dataframe_records(result.nav.to_frame("nav")),
            "benchmark": dataframe_records(result.benchmark.to_frame("benchmark")),
            "turnover": dataframe_records(result.turnover.to_frame("turnover")),
        }

    def factor_report(self, limit: int = 100) -> dict[str, Any]:
        panel = self._build_local_panel(limit=limit, require_label=True)
        if panel.empty:
            return {"rank_ic": [], "factor_columns": []}

        factor_cols = ["momentum_5", "momentum_20", "low_vol_20", "liquidity_20", "trend_60", "multi_factor_score"]
        rows = []
        for factor in factor_cols:
            daily_ic = []
            for dt in panel.index.get_level_values("date").unique().sort_values():
                chunk = panel.loc[dt]
                valid = chunk[[factor, "future_return"]].dropna()
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
        return {"rank_ic": rows, "factor_columns": factor_cols}

    def _build_local_panel(self, limit: int = 100, require_label: bool = True) -> pd.DataFrame:
        rows = []
        tickers = self.data_service.list_tickers()[:limit]
        for info in tickers:
            try:
                df, _ = self.data_service.load_candles(info.ticker, processed=True, limit=None)
            except Exception:
                continue
            if df.empty or "Close" not in df.columns:
                continue
            frame = pd.DataFrame(index=df.index)
            frame["code"] = info.ticker
            returns = df["Close"].pct_change()
            frame["momentum_5"] = df["Close"].pct_change(5)
            frame["momentum_20"] = df["Close"].pct_change(20)
            frame["low_vol_20"] = -returns.rolling(20).std()
            frame["trend_60"] = df["Close"] / df["Close"].rolling(60).mean() - 1
            if "Volume" in df.columns:
                frame["liquidity_20"] = np.log1p(df["Volume"].rolling(20).mean())
                frame["volume_change_20"] = df["Volume"].pct_change(20)
            else:
                frame["liquidity_20"] = 0.0
                frame["volume_change_20"] = 0.0
            frame["future_return"] = df["Close"].shift(-5) / df["Close"] - 1
            frame["sector"] = "unknown"
            required_features = ["momentum_5", "momentum_20", "low_vol_20", "liquidity_20", "trend_60"]
            required = required_features + (["future_return"] if require_label else [])
            rows.append(frame.dropna(subset=required))

        if not rows:
            return pd.DataFrame()

        all_rows = pd.concat(rows)
        all_rows.index.name = "date"
        all_rows = all_rows.reset_index().set_index(["date", "code"]).sort_index()
        factor_weights = {
            "momentum_5": 0.20,
            "momentum_20": 0.30,
            "low_vol_20": 0.20,
            "liquidity_20": 0.15,
            "trend_60": 0.15,
        }
        for factor in factor_weights:
            all_rows[f"{factor}_z"] = all_rows.groupby(level="date")[factor].transform(_cross_section_zscore)
        all_rows["multi_factor_score"] = 0.0
        for factor, weight in factor_weights.items():
            all_rows["multi_factor_score"] += all_rows[f"{factor}_z"] * weight
        all_rows["rank_pct"] = all_rows.groupby(level="date")["multi_factor_score"].rank(pct=True)
        required_final = ["rank_pct"] + (["future_return"] if require_label else [])
        return all_rows.dropna(subset=required_final)


def _cross_section_zscore(series: pd.Series) -> pd.Series:
    clipped = series.clip(series.quantile(0.01), series.quantile(0.99))
    std = clipped.std(ddof=0)
    if pd.isna(std) or std < 1e-12:
        return pd.Series(0.0, index=series.index)
    return (clipped - clipped.mean()) / std
