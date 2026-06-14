from typing import Any

from pydantic import BaseModel, Field


class TickerInfo(BaseModel):
    ticker: str
    custom_name: str | None = None
    rows: int = 0
    start: str | None = None
    end: str | None = None
    has_raw: bool = False
    has_processed: bool = False


class TickerListResponse(BaseModel):
    tickers: list[TickerInfo]


class CandleResponse(BaseModel):
    ticker: str
    source: str
    row_count: int
    columns: list[str]
    records: list[dict[str, Any]]


class DownloadRequest(BaseModel):
    force_update: bool = False
    start_date: str | None = None


class BatchDownloadRequest(BaseModel):
    tickers: list[str]
    force_update: bool = False
    start_date: str | None = None


class DownloadResponse(BaseModel):
    ticker: str
    success: bool
    rows: int = 0
    message: str = ""


class BatchDownloadResponse(BaseModel):
    results: list[DownloadResponse]


class DeleteTickerResponse(BaseModel):
    ticker: str
    deleted: list[str]
    success: bool


class CustomNameRequest(BaseModel):
    custom_name: str = Field(default="", max_length=80)
