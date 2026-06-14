from __future__ import annotations

from api.services.data_service import DataService
from api.services.serialization import dataframe_records, json_safe


class PredictionService:
    def __init__(self):
        self.data_service = DataService()

    def probability(self, ticker: str, horizons: list[int]) -> dict:
        from predictor import ProbabilityPredictor

        df, _ = self.data_service.load_candles(ticker, processed=True, limit=None)
        predictor = ProbabilityPredictor(df)
        predictor.create_targets(horizons=horizons)
        metrics = predictor.train(horizons=horizons)
        probs = predictor.predict_future(df.iloc[-1:])
        return {
            "ticker": ticker.upper(),
            "probabilities": {str(k): float(v) for k, v in probs.items()},
            "metrics": json_safe(metrics or {}),
        }

    def advanced_forecast(
        self,
        ticker: str,
        days: int,
        price_horizon: int,
        trend_horizons: list[int],
        method: str,
    ) -> dict:
        from advanced_predictor import AdvancedPredictor

        df, _ = self.data_service.load_candles(ticker, processed=True, limit=None)
        predictor = AdvancedPredictor(df)
        predictor.create_price_targets(horizons=trend_horizons)
        predictor.create_trend_targets(horizons=trend_horizons)
        price_metrics = predictor.train_price_model(horizon=price_horizon)
        trend_metrics = predictor.train_trend_models(horizons=trend_horizons)
        forecast = predictor.generate_future_trend_line(df.iloc[-1:], days=days, method=method)
        trend = predictor.predict_future_trend(df.iloc[-1:], days_ahead=price_horizon)
        return {
            "ticker": ticker.upper(),
            "forecast": {
                "rows": dataframe_records(forecast) if hasattr(forecast, "to_dict") else json_safe(forecast),
            },
            "trend": json_safe(trend or {}),
            "metrics": json_safe({"price": price_metrics or {}, "trend": trend_metrics or {}}),
        }
