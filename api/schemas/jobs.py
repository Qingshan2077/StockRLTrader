from typing import Any, Literal

from pydantic import BaseModel


JobStatus = Literal["queued", "running", "completed", "failed", "cancelled"]


class JobResponse(BaseModel):
    job_id: str
    name: str
    status: JobStatus
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    progress: float = 0.0
    message: str = ""
    error: str | None = None


class JobListResponse(BaseModel):
    jobs: list[JobResponse]


class JobResultResponse(BaseModel):
    job_id: str
    status: JobStatus
    result: Any = None


class PipelineJobRequest(BaseModel):
    ticker: str = "AAPL"
    skip_rl: bool = True
    walk_forward: bool = False
    multi_model: bool = False
    baseline_only: bool = False
