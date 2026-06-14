from __future__ import annotations

from pathlib import Path
import json
from datetime import datetime

import numpy as np
import pandas as pd

from api.core.config import get_config
from api.core.paths import resolve_project_path
from api.schemas.market_data import TickerInfo
from improved_data_engine import BatchDataEngine, DataEngine


PRICE_COLUMNS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]


class DataService:
    def __init__(self, data_dir: str | None = None, start_date: str | None = None):
        self.data_dir = data_dir or get_config("system.data_dir", "stock_data")
        self.start_date = start_date or get_config("system.start_date", "2015-01-01")
        self.data_path = resolve_project_path(self.data_dir)
        self.data_path.mkdir(parents=True, exist_ok=True)

    def list_tickers(self) -> list[TickerInfo]:
        batch = BatchDataEngine(data_dir=str(self.data_path), start_date=self.start_date)
        tickers = sorted(set(batch.list_available_data()))
        return [self._ticker_info(ticker) for ticker in tickers]

    def _ticker_info(self, ticker: str) -> TickerInfo:
        engine = DataEngine(ticker, data_dir=str(self.data_path), start_date=self.start_date)
        meta = engine._load_metadata()
        processed_path = self.data_path / f"{ticker.upper()}_processed.csv"
        raw_path = self.data_path / f"{ticker.upper()}_raw.csv"
        rows = int(meta.get("data_points") or 0)
        start = None
        end = None

        date_range = meta.get("date_range") or {}
        if isinstance(date_range, dict):
            start = date_range.get("start")
            end = date_range.get("end")

        if not rows:
            probe_path = processed_path if processed_path.exists() else raw_path
            if probe_path.exists():
                try:
                    df = pd.read_csv(probe_path, index_col=0, parse_dates=True)
                    rows = len(df)
                    if len(df):
                        start = str(df.index[0])
                        end = str(df.index[-1])
                except Exception:
                    rows = 0

        return TickerInfo(
            ticker=ticker.upper(),
            custom_name=meta.get("custom_name") or None,
            rows=rows,
            start=start,
            end=end,
            has_raw=raw_path.exists(),
            has_processed=processed_path.exists(),
        )

    def load_candles(self, ticker: str, processed: bool = True, limit: int | None = 500) -> tuple[pd.DataFrame, str]:
        ticker = ticker.upper().strip()

        source_path: Path | None = None
        for candidate in self._ticker_file_candidates(ticker):
            processed_path = self.data_path / f"{candidate}_processed.csv"
            raw_path = self.data_path / f"{candidate}_raw.csv"
            if processed and processed_path.exists():
                source_path = processed_path
                break
            if raw_path.exists():
                source_path = raw_path
                break
            if processed_path.exists():
                source_path = processed_path
                break

        if source_path is None:
            raise FileNotFoundError(f"No local data found for {ticker}")

        df = pd.read_csv(source_path, index_col=0, parse_dates=True)
        available = [c for c in PRICE_COLUMNS if c in df.columns]
        other = [c for c in df.columns if c not in available]
        df = df[available + other]
        if limit:
            df = df.tail(limit)
        return df, source_path.name

    def download(self, ticker: str, force_update: bool = False, start_date: str | None = None) -> dict:
        if self._is_a_share_ticker(ticker):
            return self._download_a_share(ticker, force_update=force_update, start_date=start_date)

        engine = DataEngine(
            ticker,
            data_dir=str(self.data_path),
            start_date=start_date or self.start_date,
        )
        raw = engine.fetch_data(force_update=force_update)
        processed = engine.add_technical_indicators()
        rows = len(processed) if processed is not None else len(raw) if raw is not None else 0
        return {"ticker": ticker.upper(), "success": rows > 0, "rows": rows, "message": "downloaded"}

    def batch_download(
        self,
        tickers: list[str],
        force_update: bool = False,
        start_date: str | None = None,
    ) -> list[dict]:
        results = []
        for ticker in tickers:
            try:
                results.append(self.download(ticker, force_update=force_update, start_date=start_date))
            except Exception as exc:
                results.append({
                    "ticker": ticker.upper(),
                    "success": False,
                    "rows": 0,
                    "message": str(exc),
                })
        return results

    def delete(self, ticker: str) -> dict:
        ticker = ticker.upper().strip()
        deleted = []
        for candidate in self._ticker_file_candidates(ticker):
            for suffix in ("raw.csv", "processed.csv", "meta.json", "raw.parquet", "raw.feather"):
                path = self.data_path / f"{candidate}_{suffix}"
                if path.exists():
                    path.unlink()
                    deleted.append(path.name)
        return {"ticker": ticker, "deleted": deleted, "success": bool(deleted)}

    def set_custom_name(self, ticker: str, custom_name: str) -> TickerInfo:
        engine = DataEngine(ticker, data_dir=str(self.data_path), start_date=self.start_date)
        engine.set_custom_name(custom_name)
        return self._ticker_info(ticker)

    def _download_a_share(self, ticker: str, force_update: bool = False, start_date: str | None = None) -> dict:
        try:
            import akshare as ak
        except ImportError as exc:
            raise ImportError("akshare is required for A-share download: pip install akshare") from exc

        display_ticker = ticker.upper().strip()
        numeric_symbol = self._a_share_numeric_symbol(ticker)
        raw_path = self.data_path / f"{display_ticker}_raw.csv"
        processed_path = self.data_path / f"{display_ticker}_processed.csv"
        meta_path = self.data_path / f"{display_ticker}_meta.json"

        if raw_path.exists() and processed_path.exists() and not force_update:
            processed = pd.read_csv(processed_path, index_col=0, parse_dates=True)
            return {"ticker": display_ticker, "success": len(processed) > 0, "rows": len(processed), "message": "loaded cached A-share data"}

        start = (start_date or self.start_date).replace("-", "")
        end = datetime.now().strftime("%Y%m%d")
        df = ak.stock_zh_a_hist(
            symbol=numeric_symbol,
            period="daily",
            start_date=start,
            end_date=end,
            adjust="qfq",
        )
        if df is None or df.empty:
            raise ValueError(f"No A-share data returned for {display_ticker}")

        raw = self._normalize_a_share_frame(df)
        raw.to_csv(raw_path)
        processed = self._add_basic_indicators(raw)
        processed.to_csv(processed_path)

        meta = {
            "ticker": display_ticker,
            "source": "akshare.stock_zh_a_hist",
            "last_update": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "data_points": len(processed),
            "date_range": {
                "start": str(processed.index[0]) if len(processed) else None,
                "end": str(processed.index[-1]) if len(processed) else None,
            },
        }
        with meta_path.open("w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2, ensure_ascii=False)

        return {"ticker": display_ticker, "success": len(processed) > 0, "rows": len(processed), "message": "downloaded A-share data"}

    @staticmethod
    def _is_a_share_ticker(ticker: str) -> bool:
        value = ticker.upper().strip()
        if value.endswith((".SZ", ".SH", ".BJ")):
            value = value[:6]
        if value.startswith(("SZ", "SH", "BJ")) and len(value) >= 8:
            value = value[2:8]
        return len(value) == 6 and value.isdigit() and value.startswith((
            "000", "001", "002", "003", "300",
            "600", "601", "603", "605", "688",
            "830", "831", "832", "833", "834", "835", "836", "837", "838", "839",
            "870", "871", "872", "873", "874", "875", "876", "877", "878", "879",
        ))

    @staticmethod
    def _a_share_numeric_symbol(ticker: str) -> str:
        value = ticker.upper().strip()
        if value.endswith((".SZ", ".SH", ".BJ")):
            value = value[:6]
        if value.startswith(("SZ", "SH", "BJ")) and len(value) >= 8:
            value = value[2:8]
        return value.zfill(6)

    def _ticker_file_candidates(self, ticker: str) -> list[str]:
        value = ticker.upper().strip()
        candidates = [value]
        if self._is_a_share_ticker(value):
            numeric = self._a_share_numeric_symbol(value)
            candidates.extend([
                numeric,
                f"{numeric}.SZ",
                f"{numeric}.SH",
                f"{numeric}.BJ",
                f"SZ{numeric}",
                f"SH{numeric}",
                f"BJ{numeric}",
            ])
        seen: set[str] = set()
        return [item for item in candidates if not (item in seen or seen.add(item))]

    @staticmethod
    def _normalize_a_share_frame(df: pd.DataFrame) -> pd.DataFrame:
        rename_map = {
            "\u65e5\u671f": "Date",
            "\u5f00\u76d8": "Open",
            "\u6536\u76d8": "Close",
            "\u6700\u9ad8": "High",
            "\u6700\u4f4e": "Low",
            "\u6210\u4ea4\u91cf": "Volume",
            "\u6210\u4ea4\u989d": "Amount",
            "\u632f\u5e45": "Amplitude",
            "\u6da8\u8dcc\u5e45": "ChangePct",
            "\u6da8\u8dcc\u989d": "Change",
            "\u6362\u624b\u7387": "Turnover",
            "date": "Date",
            "open": "Open",
            "close": "Close",
            "high": "High",
            "low": "Low",
            "volume": "Volume",
            "amount": "Amount",
        }
        frame = df.rename(columns=rename_map).copy()
        if "Date" in frame.columns:
            frame = frame.set_index("Date")
        frame.index = pd.to_datetime(frame.index)
        frame = frame.sort_index()
        keep = [col for col in ["Open", "High", "Low", "Close", "Volume", "Amount", "Turnover"] if col in frame.columns]
        frame = frame[keep].copy()
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            if col not in frame.columns:
                raise ValueError(f"A-share data is missing required column: {col}")
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
        for col in ["Amount", "Turnover"]:
            if col in frame.columns:
                frame[col] = pd.to_numeric(frame[col], errors="coerce")
        return frame.dropna(subset=["Open", "High", "Low", "Close"])

    @staticmethod
    def _add_basic_indicators(df: pd.DataFrame) -> pd.DataFrame:
        frame = df.copy()
        close = frame["Close"]
        volume = frame["Volume"]
        frame["SMA_10"] = close.rolling(10).mean()
        frame["SMA_50"] = close.rolling(50).mean()
        frame["SMA_200"] = close.rolling(200).mean()
        frame["Dist_SMA_10"] = close / frame["SMA_10"] - 1
        frame["Dist_SMA_50"] = close / frame["SMA_50"] - 1
        frame["EMA_12"] = close.ewm(span=12, adjust=False).mean()
        frame["EMA_26"] = close.ewm(span=26, adjust=False).mean()
        frame["RSI"] = DataService._rsi(close)
        macd = frame["EMA_12"] - frame["EMA_26"]
        frame["MACD_12_26_9"] = macd
        frame["MACDs_12_26_9"] = macd.ewm(span=9, adjust=False).mean()
        frame["MACDh_12_26_9"] = frame["MACD_12_26_9"] - frame["MACDs_12_26_9"]
        bb_mid = close.rolling(20).mean()
        bb_std = close.rolling(20).std()
        bb_upper = bb_mid + 2 * bb_std
        bb_lower = bb_mid - 2 * bb_std
        frame["BB_Width"] = (bb_upper - bb_lower) / bb_mid
        frame["BB_Position"] = (close - bb_lower) / (bb_upper - bb_lower)
        high_low = frame["High"] - frame["Low"]
        high_close = (frame["High"] - close.shift(1)).abs()
        low_close = (frame["Low"] - close.shift(1)).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        frame["ATR"] = true_range.rolling(14).mean()
        frame["Vol_Change"] = volume.pct_change()
        frame["Vol_SMA_20"] = volume.rolling(20).mean()
        frame["Vol_Ratio"] = volume / frame["Vol_SMA_20"]
        frame["Returns"] = close.pct_change()
        frame["Returns_5d"] = close.pct_change(5)
        frame["Returns_20d"] = close.pct_change(20)
        frame["Trend_Strength"] = (frame["SMA_10"] - frame["SMA_50"]) / frame["SMA_50"]
        return frame.replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def _rsi(close: pd.Series, length: int = 14) -> pd.Series:
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(length).mean()
        loss = (-delta.clip(upper=0)).rolling(length).mean()
        rs = gain / loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))
