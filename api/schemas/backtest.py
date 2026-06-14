from typing import Any, Literal

from pydantic import BaseModel


class BacktestRunRequest(BaseModel):
    ticker: str
    model_run_id: str | None = None
    mode: Literal["signal_only", "signal_risk", "full"] = "signal_risk"
    initial_balance: float = 10000.0
    commission: float = 0.001
    slippage: float = 0.0005
    max_position: float = 1.0


class BacktestRunResponse(BaseModel):
    job_id: str | None = None
    run_id: str | None = None
    status: str
    result: dict[str, Any] = {}


class BacktestRunListResponse(BaseModel):
    runs: list[dict[str, Any]]
