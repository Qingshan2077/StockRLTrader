from typing import Any

from pydantic import BaseModel


class ProbabilityPredictionRequest(BaseModel):
    ticker: str
    horizons: list[int] = [1, 5, 10]


class ProbabilityPredictionResponse(BaseModel):
    ticker: str
    probabilities: dict[str, float]
    metrics: dict[str, Any] = {}


class AdvancedForecastRequest(BaseModel):
    ticker: str
    days: int = 30
    price_horizon: int = 10
    trend_horizons: list[int] = [1, 5, 10, 20, 30]
    method: str = "mixed"


class AdvancedForecastResponse(BaseModel):
    ticker: str
    forecast: dict[str, Any]
    trend: dict[str, Any] = {}
    metrics: dict[str, Any] = {}
