from typing import Any

from pydantic import BaseModel


class SignalTrainRequest(BaseModel):
    ticker: str
    models: list[str] = ["lightgbm", "ridge"]
    label_type: str = "regression"
    horizon: int = 5
    use_cache: bool = True


class EnsembleRequest(BaseModel):
    method: str = "weighted"
    model_run_ids: list[str] = []


class SignalRunResponse(BaseModel):
    run_id: str | None = None
    job_id: str | None = None
    status: str
    result: dict[str, Any] = {}


class SignalRunListResponse(BaseModel):
    runs: list[dict[str, Any]]
