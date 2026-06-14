from typing import Any

from pydantic import BaseModel


class MacdBacktestRequest(BaseModel):
    ticker: str
    initial_balance: float = 10000.0
    commission: float = 0.001
    smooth_threshold: float = 0.02


class MacdBacktestResponse(BaseModel):
    ticker: str
    metrics: dict[str, Any]
    trades: list[dict[str, Any]]
    signals_summary: dict[str, Any]
    rows: list[dict[str, Any]]
