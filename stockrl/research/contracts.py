"""Versioned, serializable research contracts; no training or storage side effects."""
from __future__ import annotations
from datetime import date
import hashlib
import json
from typing import Annotated, Any, Literal
from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator

PositiveInt = Annotated[StrictInt, Field(gt=0)]

class ResearchError(ValueError):
    def __init__(self, code: str, message: str = ''):
        self.code = code
        super().__init__(f'{code}: {message}')

class Contract(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True, allow_inf_nan=False)

class TrainingBudget(Contract):
    requested_timesteps: Annotated[StrictInt, Field(gt=0, le=1000000)] = 100000
    episode_length: PositiveInt = 126
    algorithm: Literal['PPO', 'SAC'] = 'PPO'

class AcceptanceThresholds(Contract):
    minimum_folds: Literal[4] = 4
    required_seeds: tuple[int, ...] = (42,43,44,45,46)
    comparable_fraction: Literal[0.8] = 0.8
    winning_fold_fraction: Literal[0.6] = 0.6
    maximum_drawdown_difference: Literal[0.02] = 0.02
    risk_absolute_tolerance: Literal[0.02] = 0.02
    risk_relative_tolerance: Literal[0.2] = 0.2
    minimum_assets: Literal[5] = 5
    winning_asset_fraction: Literal[0.6] = 0.6

class SessionWindow(Contract):
    calendar_start: str
    calendar_end: str
    start_session: str
    end_session: str
    initial_session: str
    reward_sessions: tuple[str, ...]
    warmup_sessions: tuple[str, ...] = ()

    @model_validator(mode='after')
    def causal_window(self):
        values = (self.calendar_start, self.calendar_end, self.start_session, self.end_session,
                  self.initial_session, *self.reward_sessions, *self.warmup_sessions)
        if any(date.fromisoformat(value).isoformat() != value for value in values):
            raise ValueError('window dates must use ISO calendar format')
        if (not self.reward_sessions or tuple(sorted(set(self.reward_sessions))) != self.reward_sessions
                or self.start_session != self.reward_sessions[0]
                or not self.initial_session < self.start_session
                or not self.calendar_start <= self.start_session <= self.reward_sessions[-1] < self.calendar_end <= self.end_session):
            raise ValueError('window reward boundaries must be ordered and half-open')
        if (len(self.warmup_sessions) > 252 or tuple(sorted(set(self.warmup_sessions))) != self.warmup_sessions
                or self.warmup_sessions and self.warmup_sessions[-1] != self.initial_session):
            raise ValueError('warmup must be ordered, at most252 sessions and end at initial observation')
        return self

class FoldPlan(Contract):
    fold_id: str
    instrument_id: str | None = None
    train: SessionWindow
    validation: SessionWindow
    test: SessionWindow

    @model_validator(mode='after')
    def disjoint_rewards(self):
        if (self.train.end_session != self.validation.start_session
                or self.validation.end_session != self.test.start_session
                or self.train.reward_sessions[-1] != self.validation.initial_session
                or self.validation.reward_sessions[-1] != self.test.initial_session):
            raise ValueError('fold must have consecutive nonoverlapping train, validation, test rewards')
        return self

    @property
    def train_rewards(self): return self.train.reward_sessions
    @property
    def validation_rewards(self): return self.validation.reward_sessions
    @property
    def test_rewards(self): return self.test.reward_sessions

class ResearchProtocolDraft(Contract):
    protocol_id: str = Field(min_length=1)
    version: Literal[2] = 2
    hypothesis: str = 'Can autonomous RL allocation improve net performance consistently against transparent references at comparable risk?'
    scope: Literal['single_asset','asset_set'] = 'single_asset'
    instrument_ids: tuple[str, ...]
    dataset_ids: tuple[str, ...]
    dataset_fingerprints: dict[str, str]
    market_profile_ids: tuple[str, ...]
    market_profile_fingerprints: dict[str, str]
    asset_selection_note: str = Field(min_length=1)
    first_test_session: str
    fold_count: Annotated[StrictInt, Field(gt=0, le=250)] = 4
    seeds: tuple[StrictInt, ...] = (42,43,44,45,46)
    training_budget: TrainingBudget = Field(default_factory=TrainingBudget)
    checkpoint_budget: Annotated[StrictInt, Field(gt=0,le=20)] = 20
    primary_reference_ids: tuple[Literal['fixed_50','matched_fixed'], ...] = ('fixed_50','matched_fixed')
    cost_scenarios: tuple[Literal['base','execution_x2','execution_x3'], ...] = ('base','execution_x2','execution_x3')
    qualification: Literal['exploratory','declared_holdout'] = 'exploratory'
    acceptance_thresholds: AcceptanceThresholds = Field(default_factory=AcceptanceThresholds)

    @model_validator(mode='after')
    def validate_protocol(self):
        if date.fromisoformat(self.first_test_session).isoformat() != self.first_test_session:
            raise ValueError('first_test_session must be ISO date')
        if any(seed < 0 or seed > 2**32-1 for seed in self.seeds):
            raise ValueError('seeds must be uint32')
        for name in ('instrument_ids','dataset_ids','market_profile_ids','seeds'):
            values = getattr(self,name)
            if not values or len(set(values)) != len(values):
                raise ValueError(f'{name} must be nonempty and unique')
        if self.scope == 'single_asset' and len(self.instrument_ids) != 1:
            raise ValueError('single_asset requires one instrument')
        for ids, hashes in ((self.dataset_ids,self.dataset_fingerprints),(self.market_profile_ids,self.market_profile_fingerprints)):
            if set(ids) != set(hashes) or any(len(h)!=64 or any(c not in '0123456789abcdef' for c in h) for h in hashes.values()):
                raise ValueError('exact SHA-256 fingerprints required')
        if self.primary_reference_ids != ('fixed_50','matched_fixed') or self.cost_scenarios != ('base','execution_x2','execution_x3'):
            raise ValueError('v2 references and cost scenarios are fixed')
        if self.acceptance_thresholds.required_seeds != (42,43,44,45,46):
            raise ValueError('qualification seed standard is fixed')
        return self

class ResearchProtocol(ResearchProtocolDraft):
    fold_plan: tuple[FoldPlan, ...]
    checkpoint_steps: tuple[PositiveInt, ...]

    @model_validator(mode='after')
    def validate_checkpoints(self):
        from stockrl.research.folds import checkpoint_steps
        if self.checkpoint_steps != checkpoint_steps(self.training_budget,self.checkpoint_budget):
            raise ValueError('checkpoint steps differ from locked budget schedule')
        if any(fold.instrument_id not in (None, *self.instrument_ids) for fold in self.fold_plan):
            raise ValueError('fold instrument is outside protocol')
        for instrument in self.instrument_ids:
            folds = [fold for fold in self.fold_plan if fold.instrument_id in (None, instrument)]
            if len(folds) > self.fold_count or len({fold.fold_id for fold in folds}) != len(folds):
                raise ValueError('duplicate or excess folds')
            rewards = [session for fold in folds for session in fold.test.reward_sessions]
            if rewards != sorted(set(rewards)):
                raise ValueError('test folds must be ordered and nonoverlapping')
        return self

    def lock(self) -> LockedProtocol:
        canonical = json.dumps(self.model_dump(mode='json'), sort_keys=True, separators=(',',':'),ensure_ascii=False,allow_nan=False)
        return LockedProtocol(canonical_json=canonical,sha256=hashlib.sha256(canonical.encode('utf-8')).hexdigest())

class LockedProtocol(Contract):
    canonical_json: str
    sha256: str

    def verify(self) -> ResearchProtocol:
        if hashlib.sha256(self.canonical_json.encode('utf-8')).hexdigest() != self.sha256:
            raise ValueError('protocol fingerprint mismatch')
        protocol = ResearchProtocol.model_validate_json(self.canonical_json)
        if protocol.lock().canonical_json != self.canonical_json:
            raise ValueError('protocol JSON is not canonical')
        return protocol

class BudgetPreview(Contract):
    unit_count: int
    requested_total_steps: int
    rollout_upper_bound: int
    actual_steps_per_unit: int
    rollout_steps: int
    checkpoint_steps: tuple[int,...]
    checkpoint_evaluations: int
    calibration_evaluations: int
    test_evaluations: int
    extra_evaluations: int

class ResearchPreview(Contract):
    fold_plan: tuple[FoldPlan,...]
    budget: BudgetPreview
    checkpoint_steps: tuple[int,...]
    qualification: Literal['exploratory','declared_holdout']
    blockers: tuple[str,...] = ()
    omitted_folds: tuple[str,...] = ()

class UnitKey(Contract):
    instrument_id: str
    fold_id: str
    seed: StrictInt

class StrategyMetrics(Contract):
    net_return: float = 0
    cagr: float | None = 0
    annualized_volatility: float | None = 0
    sharpe: float | None = None
    max_drawdown: float = 0
    fees: float = 0
    turnover: float = 0
    average_exposure: float = 0
    trade_count: int = 0
    reward_sessions: int = 0
    final_nav: float | None = None
    final_receivables: float = 0

class CheckpointResult(Contract):
    checkpoint_id: str
    actual_steps: int
    gradient_updates: int
    model_sha256: str
    validation_log_return: float
    evaluated_at: str
    metrics: StrategyMetrics = Field(default_factory=StrategyMetrics)
    validation_reward: float | None = None

class CostResult(Contract):
    scenario_id: Literal['base','execution_x2','execution_x3']
    model_sha256: str
    metrics: dict[str, StrategyMetrics]
    risk_comparable: bool | None = None
    artifact_paths: dict[str,str] = Field(default_factory=dict)

class UnitResult(Contract):
    key: UnitKey
    status: Literal['completed','failed','cancelled','pending','running'] = 'completed'
    cost_results: tuple[CostResult,...] = ()
    candidates: tuple[CheckpointResult,...] = ()
    selected_checkpoint_id: str | None = None
    selection_reason: str | None = None
    matched_weight: float | None = None
    calibration_error: float | None = None
    actual_steps: int = 0
    fingerprints_valid: bool = True
    rule_coverage_complete: bool = True
    error: str | None = None

class ExposureRecord(Contract):
    instrument_id: str
    start_session: str
    end_session: str
    scope: Literal['whole_period_knowledge','evaluation','exposed_test']
    reason: str
    recorded_at: str
    protocol_id: str | None = None
    source: str

class ResearchReport(Contract):
    protocol_id: str
    technical_status: Literal['completed','failed','cancelled','incomplete']
    qualification: Literal['exploratory','declared_holdout']
    economic_outcome: Literal['insufficient_evidence','no_added_value','candidate_edge']
    provisional_outcome: Literal['insufficient_evidence','no_added_value','candidate_edge'] | None = None
    reasons: tuple[str,...] = ()
    assets: dict[str,Any] = Field(default_factory=dict)
    units: tuple[UnitResult,...] = ()

class DiagnosticsReport(Contract):
    boundary_action_ratio: float | None = None
    clipping_ratio: float | None = None
    near_half_ratio: float | None = None
    warnings: tuple[str,...] = ()
    unavailable: tuple[str,...] = ()
    feature_stats: dict[str,Any] = Field(default_factory=dict)
    regimes: dict[str,Any] = Field(default_factory=dict)
    training_log: tuple[dict[str,Any],...] = ()
    summary: dict[str,Any] = Field(default_factory=dict)
