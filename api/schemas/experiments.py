from typing import Any

from pydantic import BaseModel


class ExperimentCreateRequest(BaseModel):
    name: str
    config: dict[str, Any] = {}
    tags: list[str] = []


class ExperimentResponse(BaseModel):
    exp_id: str
    name: str
    status: str | None = None
    tags: list[str] = []
    config: dict[str, Any] = {}
    final_metrics: dict[str, Any] = {}
    created_at: str | None = None
    completed_at: str | None = None


class ExperimentListResponse(BaseModel):
    experiments: list[ExperimentResponse]


class ExperimentCompareRequest(BaseModel):
    exp_ids: list[str]
    metrics: list[str] | None = None
