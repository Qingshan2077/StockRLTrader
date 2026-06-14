from typing import Any

from pydantic import BaseModel


class RLTrainRequest(BaseModel):
    ticker: str
    algorithm: str = "PPO"
    timesteps: int = 50000
    signal_run_id: str | None = None
    pnl_weight: float = 1.0
    cost_weight: float = 0.1
    turnover_weight: float = 0.05
    drawdown_weight: float = 0.1


class RLRunResponse(BaseModel):
    job_id: str | None = None
    status: str
    result: dict[str, Any] = {}
