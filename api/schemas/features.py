from typing import Any

from pydantic import BaseModel


class FeatureBuildRequest(BaseModel):
    use_cache: bool = True
    horizon: int = 5
    label_type: str = "regression"


class FeatureSummaryResponse(BaseModel):
    ticker: str
    row_count: int
    feature_count: int
    label_count: int = 0
    cache_hit_rate: float = 0.0
    groups: list[str] = []
    columns: list[str] = []
    sample: list[dict[str, Any]] = []
