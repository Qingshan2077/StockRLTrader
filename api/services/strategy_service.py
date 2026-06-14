from __future__ import annotations

from api.services.data_service import DataService
from api.services.serialization import dataframe_records, json_safe


class StrategyService:
    def __init__(self):
        self.data_service = DataService()

    def macd_backtest(
        self,
        ticker: str,
        initial_balance: float,
        commission: float,
        smooth_threshold: float,
    ) -> dict:
        from macd_strategy import MACDStrategy

        df, _ = self.data_service.load_candles(ticker, processed=True, limit=None)
        strategy = MACDStrategy(
            df,
            initial_balance=initial_balance,
            commission=commission,
            smooth_threshold=smooth_threshold,
        )
        result_df, metrics = strategy.backtest()
        return {
            "ticker": ticker.upper(),
            "metrics": json_safe(metrics or {}),
            "trades": json_safe(strategy.get_trade_details()),
            "signals_summary": json_safe(strategy.get_signals_summary()),
            "rows": dataframe_records(result_df, limit=1000),
        }
