from typing import Any, Literal

from .models import ErrorSummary, ExperimentDetail, StrictModel

Policy = Literal['rl', 'cash', 'buy_hold', 'half', 'trend']


class ErrorEnvelope(StrictModel):
    error: ErrorSummary
    request_id: str


class SeriesPoint(StrictModel):
    date: str
    nav: float
    cash: float
    shares: float
    weight: float
    requested_weight: float | None
    executed_weight: float | None
    turnover: float
    cost: float


class SeriesResponse(StrictModel):
    seed: int
    policy: Policy
    points: list[SeriesPoint]
    original_count: int
    returned_count: int
    is_sampled: bool


class TradeRow(StrictModel):
    date: str
    side: Literal['buy', 'sell']
    shares: float
    price: float
    cost: float


class TradesResponse(StrictModel):
    seed: int
    policy: Policy
    items: list[TradeRow]
    total_rows: int
    next_cursor: str | None
    has_more: bool


class ComparisonResponse(StrictModel):
    experiments: list[ExperimentDetail]
    same_conditions: bool
    differences: list[dict[str, Any]]
    notes: list[str]


class EmptyRequest(StrictModel):
    pass
