from typing import Any

from pydantic import BaseModel, Field


class ApiError(BaseModel):
    detail: str


class DataFramePayload(BaseModel):
    columns: list[str]
    records: list[dict[str, Any]]
    row_count: int


class SystemSummary(BaseModel):
    name: str
    version: str
    stage: str
    data_dir: str
    model_dir: str
    log_dir: str
    available_tickers: int


class HealthResponse(BaseModel):
    status: str = "ok"
    service: str = "stocktrader-api"
    version: str = Field(default="0.1.0")
