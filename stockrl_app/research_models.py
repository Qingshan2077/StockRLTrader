"""Public v2 response projections: storage paths and ownership tokens stay private."""
from typing import Any, Literal

from stockrl.research.contracts import ResearchPreview, ResearchProtocolDraft, ResearchProtocol, ResearchReport
from .models import StrictModel, JobDetail


class MarketPreviewResponse(StrictModel):
    qualification: Literal['ready', 'incomplete', 'unsupported']
    blocking_reasons: list[str]
    issues: list[str]
    file_hashes: dict[str, str]
    fingerprint: str
    metadata: dict[str, Any] | None
    market_profile: dict[str, Any] | None
    rows: int


class MarketDatasetResponse(MarketPreviewResponse):
    dataset_id: str
    created_at: str


class MarketDatasetList(StrictModel):
    items: list[MarketDatasetResponse]


class ResearchPreviewResponse(ResearchPreview):
    exposure_warnings: list[str]
    protocol_sha256: str
    canonical_request: ResearchProtocolDraft
    maximum_runtime_seconds: int


class ResearchSubmission(StrictModel):
    research_id: str
    job_id: str
    protocol_sha256: str


class ResearchResumeResponse(StrictModel):
    research_id: str
    job_id: str


class ResearchSummary(StrictModel):
    research_id: str
    job_id: str
    created_at: str
    technical_status: str
    protocol_sha256: str
    qualification: Literal['exploratory', 'declared_holdout']
    hypothesis: str


class ResearchListResponse(StrictModel):
    items: list[ResearchSummary]
    has_more: bool
    next_cursor: str | None


class ResearchAttemptResponse(StrictModel):
    attempt_id: str
    job_id: str
    status: str
    created_at: str


class ResearchUnitProgress(StrictModel):
    instrument_id: str
    fold_id: str
    seed: int
    status: str
    error_json: str | None


class ResearchDetailResponse(ResearchReport):
    research_id: str
    job_id: str
    job_status: JobDetail
    created_at: str
    protocol: ResearchProtocol
    protocol_sha256: str
    preview: ResearchPreviewResponse
    attempts: list[ResearchAttemptResponse]
    unit_progress: list[ResearchUnitProgress]
