"""Strict JSON contracts, including separate internal storage records."""
from datetime import date, datetime, timezone
from typing import TYPE_CHECKING, Annotated, Any, Generic, Literal, TypeVar
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

if TYPE_CHECKING:
    from stockrl.env import TradingConfig

JobStatus = Literal['queued', 'running', 'cancelling', 'succeeded', 'failed', 'cancelled', 'interrupted']
JobKind = Literal['train', 'replay']
Phase = Literal['preparing', 'training', 'validating', 'evaluating', 'publishing']
Integrity = Literal['pending', 'complete', 'partial', 'corrupt', 'unsupported']
Seed = Annotated[int, Field(ge=0, le=4294967295)]
T = TypeVar('T')


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec='microseconds').replace('+00:00', 'Z')


class StrictModel(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True, allow_inf_nan=False, validate_default=True)


class TradingConfigModel(StrictModel):
    initial_cash: float = Field(default=10000.0, gt=0)
    commission: float = Field(default=.001, ge=0, lt=1)
    slippage: float = Field(default=.0005, ge=0, lt=1)
    sell_tax: float = Field(default=0.0, ge=0, lt=1)
    min_commission: float = Field(default=0.0, ge=0)
    lot_size: int = Field(default=1, ge=1)
    max_participation: float = Field(default=.01, ge=0, le=1)
    t_plus_one: bool = True
    drawdown_penalty: float = Field(default=0.0, ge=0)
    turnover_penalty: float = Field(default=0.0, ge=0)

    def to_core(self) -> 'TradingConfig':
        from stockrl.env import TradingConfig
        return TradingConfig(**self.model_dump())


class ExperimentRequest(StrictModel):
    dataset_id: str
    purpose: Literal['technical_validation', 'research'] = 'technical_validation'
    research_question: str = Field(default='', max_length=10000)
    comparison_notes: str = Field(default='', max_length=10000)
    start_date: str | None = None
    end_date: str | None = None
    algorithm: Literal['PPO', 'SAC'] = 'PPO'
    timesteps: int = Field(default=10000, ge=2, le=1000000)
    seeds: list[Seed] = Field(default_factory=lambda: [42], min_length=1, max_length=10)
    episode_length: int | None = Field(default=126, ge=1)
    train_ratio: float = Field(default=.6, gt=0, lt=1)
    val_ratio: float = Field(default=.2, gt=0, lt=1)
    trading_config: TradingConfigModel = Field(default_factory=TradingConfigModel)

    @field_validator('dataset_id')
    @classmethod
    def uuid_id(cls, value: str) -> str:
        return str(UUID(value))

    @field_validator('start_date', 'end_date')
    @classmethod
    def calendar_date(cls, value: str | None) -> str | None:
        if value is not None:
            parsed = date.fromisoformat(value)
            if parsed.isoformat() != value:
                raise ValueError('日期必须使用 YYYY-MM-DD 格式')
        return value

    @field_validator('research_question', 'comparison_notes')
    @classmethod
    def trim_text(cls, value: str) -> str:
        return value.strip()

    @model_validator(mode='after')
    def valid_combination(self) -> 'ExperimentRequest':
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError('seed 不可重复')
        if self.train_ratio + self.val_ratio >= 1:
            raise ValueError('训练与验证比例之和必须小于 1')
        if self.purpose == 'research' and not self.research_question:
            raise ValueError('研究实验必须填写研究问题')
        if self.start_date and self.end_date and self.start_date > self.end_date:
            raise ValueError('开始日期不得晚于结束日期')
        return self


class DatasetMetadata(StrictModel):
    display_name: str | None = Field(default=None, max_length=200)
    adjustment: str = Field(default='unknown', min_length=1, max_length=100)
    quote_unit: str = Field(default='unknown', min_length=1, max_length=100)
    source_description: str = Field(default='', max_length=10000)


class DemoDatasetRequest(DatasetMetadata):
    rows: int = Field(default=756, ge=2, le=100000)
    seed: Seed = 42


class LocalDatasetRequest(DatasetMetadata):
    source_id: str = Field(min_length=1, max_length=100)


class LocalSource(StrictModel):
    source_id: str
    display_name: str


class Dataset(StrictModel):
    dataset_id: str
    source_kind: Literal['csv', 'demo', 'local']
    display_name: str
    source_description: str = ''
    is_synthetic: bool
    adjustment: str = 'unknown'
    quote_unit: str = 'unknown'
    rows: int
    first_date: str
    last_date: str
    snapshot_sha256: str
    original_sha256: str | None = None
    created_at: str


class DatasetRecord(Dataset):
    snapshot_relative_path: str
    original_relative_path: str | None = None
    source_reference: str | None = None

    def public(self) -> Dataset:
        return Dataset.model_validate({k: getattr(self, k) for k in Dataset.model_fields})


class BarRow(StrictModel):
    Date: str
    Open: float
    High: float
    Low: float
    Close: float
    Volume: float


class DatasetPreview(StrictModel):
    dataset: Dataset
    total_rows: int
    sample: list[BarRow]
    trainable: bool
    warnings: list[str] = Field(default_factory=list)


class SplitInterval(StrictModel):
    start: int
    end: int
    reward_count: int
    observation_start_date: str
    first_reward_date: str
    last_reward_date: str


class ExperimentPreview(StrictModel):
    canonical_request: ExperimentRequest
    request_sha256: str
    filtered_data_sha256: str
    splits: dict[str, SplitInterval]
    seed_count: int
    warnings: list[str] = Field(default_factory=list)


class SubmissionResult(StrictModel):
    experiment_id: str
    job_id: str


class ErrorSummary(StrictModel):
    code: str
    message: str
    details: list[dict[str, Any]] | dict[str, Any] | None = None


class JobDetail(StrictModel):
    job_id: str
    experiment_id: str
    kind: JobKind
    status: JobStatus = 'queued'
    revision: int = 0
    phase: Phase | None = None
    seed: int | None = None
    seed_index: int | None = None
    seed_count: int = 1
    requested_steps_total: int = 0
    actual_steps_total: int = 0
    training_fraction: float | None = 0.0
    completed_seeds: int = 0
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    heartbeat_at: str | None = None
    worker_available: bool = False
    recovery_waiting: bool = False
    error: ErrorSummary | None = None


class JobRecord(JobDetail):
    owner_token: str | None = None
    cancel_requested: bool = False
    progress_steps: dict[str, int] = Field(default_factory=dict)
    publish_target: str | None = None
    publish_manifest_sha256: str | None = None
    publish_owner_token: str | None = None
    publish_revision: int | None = None

    def public(self) -> JobDetail:
        return JobDetail.model_validate({k: getattr(self, k) for k in JobDetail.model_fields})


class StateEventPayload(StrictModel):
    status: JobStatus
    revision: int


class PhaseEventPayload(StrictModel):
    phase: Phase
    seed: int | None = None
    seed_index: int | None = None


class ProgressEventPayload(StrictModel):
    seed: int | None = None
    seed_index: int | None = None
    actual_steps_total: int
    requested_steps_total: int
    completed_seeds: int
    training_fraction: float | None


class JobEvent(StrictModel):
    job_id: str
    seq: int
    occurred_at: str
    event_type: Literal['state', 'phase', 'progress', 'error']
    payload: StateEventPayload | PhaseEventPayload | ProgressEventPayload | ErrorSummary


class EventPage(StrictModel):
    items: list[JobEvent]
    next_seq: int
    has_more: bool


class Page(StrictModel, Generic[T]):
    items: list[T]
    next_cursor: str | None = None
    has_more: bool = False


class Versions(StrictModel):
    artifact_schema_version: int | Literal['legacy-v0', 'unknown'] = 1
    core_semantics_version: int | Literal['unknown'] = 1
    metrics_version: int | Literal['unknown'] = 1


class ArtifactRecord(StrictModel):
    artifact_id: str
    experiment_id: str
    kind: str
    filename: str
    size: int = Field(ge=0)
    sha256: str
    download_url: str


class ArtifactStorageRecord(ArtifactRecord):
    root_id: Literal['output', 'app'] = 'output'
    relative_path: str
    created_at: str

    def public(self) -> ArtifactRecord:
        return ArtifactRecord.model_validate({k: getattr(self, k) for k in ArtifactRecord.model_fields})


class LegacyConfiguration(StrictModel):
    algorithm: Literal['PPO', 'SAC'] | None = None
    trading_config: TradingConfigModel | None = None
    timesteps: int | None = Field(default=None, ge=2)
    seeds: list[Seed] = Field(default_factory=list)
    episode_length: int | None = Field(default=None, ge=1)
    train_ratio: float | None = Field(default=None, gt=0, lt=1)
    val_ratio: float | None = Field(default=None, gt=0, lt=1)


class ExperimentDetail(StrictModel):
    experiment_id: str
    kind: JobKind = 'train'
    source_experiment_id: str | None = None
    job_id: str | None = None
    run_id: str | None = None
    created_at: str
    request: ExperimentRequest | None = None
    request_sha256: str | None = None
    filtered_data_sha256: str | None = None
    legacy_config: LegacyConfiguration | None = None
    dataset: Dataset | None = None
    data_label: str | None = None
    is_synthetic: bool | None = None
    splits: dict[str, SplitInterval] = Field(default_factory=dict)
    versions: Versions = Field(default_factory=Versions)
    integrity: Integrity = 'pending'
    replayable: bool = False
    replay_block_reason: str | None = '实验尚未完成。'
    runs: list[dict[str, Any]] = Field(default_factory=list)
    aggregate: dict[str, Any] | None = None
    baseline_aggregate: dict[str, dict[str, Any]] = Field(default_factory=dict)
    artifacts: list[ArtifactRecord] = Field(default_factory=list)


class ExperimentRecord(ExperimentDetail):
    dataset_id: str | None = None
    bars_relative_path: str | None = None
    output_relative_path: str | None = None
    legacy_fingerprint: str | None = None

    def public(self) -> ExperimentDetail:
        return ExperimentDetail.model_validate({k: getattr(self, k) for k in ExperimentDetail.model_fields})


class WorkerInstance(StrictModel):
    owner_token: str
    started_at: str
    heartbeat_at: str
    recovery_waiting: bool = False
    active_job_id: str | None = None
    pid: int | None = None


class Capabilities(StrictModel):
    api_version: int = 1
    database_schema_version: int = 1
    artifact_schema_version: int = 1
    core_semantics_version: int = 1
    metrics_version: int = 1
    algorithms: list[str] = Field(default_factory=lambda: ['PPO', 'SAC'])
    policies: list[str] = Field(default_factory=lambda: ['rl', 'cash', 'buy_hold', 'half', 'trend'])
    defaults: dict[str, Any]
    limits: dict[str, Any]
    worker_available: bool = False
