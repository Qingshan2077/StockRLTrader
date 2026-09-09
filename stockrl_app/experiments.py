"""Preview and atomic idempotent enqueue, independent of HTTP and training libraries."""
from datetime import date
import hashlib
import os
from uuid import UUID, uuid4
from typing import Any

import pandas as pd
from pydantic import ValidationError

from stockrl.protocol import split_intervals

from .datasets import DatasetService, bars_bytes
from .errors import AppError
from .models import (Capabilities, ExperimentDetail, ExperimentPreview, ExperimentRecord, ExperimentRequest,
                     JobRecord, Page, SplitInterval, SubmissionResult, utc_now)
from .settings import AppSettings, confined_path
from .storage.common import canonical_json
from .storage.database import Database
from .storage.experiments import ExperimentRepository
from .storage.jobs import JobRepository


def normalize_request(request: ExperimentRequest | dict[str, Any]) -> ExperimentRequest:
    try:
        # Revalidate even model instances: model_copy/update and mutable lists can bypass validators.
        return ExperimentRequest.model_validate(request.model_dump() if isinstance(request, ExperimentRequest) else request)
    except ValidationError as exc:
        details = [{'field': '.'.join(str(part) for part in error['loc']), 'reason': error['msg']}
                   for error in exc.errors(include_input=False, include_context=False, include_url=False)]
        raise AppError('INVALID_REQUEST', '实验配置无效，请检查标记字段。', 422, details) from exc


def request_hash(request: ExperimentRequest) -> str:
    return hashlib.sha256(canonical_json(request.model_dump(mode='json')).encode()).hexdigest()


def idempotency_key(value: str) -> str:
    try:
        return str(UUID(value))
    except (ValueError, TypeError, AttributeError) as exc:
        raise AppError('INVALID_IDEMPOTENCY_KEY', '提交标识必须是 UUID。') from exc


class ExperimentService:
    def __init__(self, settings: AppSettings, db: Database) -> None:
        self.settings = settings
        self.db = db
        self.datasets = DatasetService(settings, db)
        self.repository = ExperimentRepository(db)
        self.jobs = JobRepository(db, settings.worker_unavailable_seconds)

    def _prepare(self, request: ExperimentRequest) -> tuple[ExperimentPreview, pd.DataFrame]:
        bars = self.datasets.read_bars(request.dataset_id)
        # Compare market calendar dates without UTC conversion, including an entire end date.
        if request.start_date:
            bars = bars[bars.index.date >= date.fromisoformat(request.start_date)]
        if request.end_date:
            bars = bars[bars.index.date <= date.fromisoformat(request.end_date)]
        try:
            intervals = split_intervals(bars, request.train_ratio, request.val_ratio)
            request.trading_config.to_core()
        except ValueError as exc:
            raise AppError('INVALID_EXPERIMENT_DATA', '数据不足或交易配置无效，无法创建此实验。', 422,
                           [{'field': 'dataset_id', 'reason': str(exc)}]) from exc
        dataset = self.datasets.get(request.dataset_id)
        warnings = []
        if dataset.is_synthetic:
            warnings.append('合成演示数据仅用于软件验证，不构成投资研究证据。')
        if dataset.adjustment == 'unknown':
            warnings.append('价格调整口径未声明。')
        if dataset.quote_unit == 'unknown':
            warnings.append('计价单位未声明。')
        return ExperimentPreview(canonical_request=request, request_sha256=request_hash(request),
            filtered_data_sha256=hashlib.sha256(bars_bytes(bars)).hexdigest(),
            splits={name: SplitInterval.model_validate(interval) for name, interval in intervals.items()},
            seed_count=len(request.seeds), warnings=warnings), bars

    def preview(self, request: ExperimentRequest | dict[str, Any]) -> ExperimentPreview:
        return self._prepare(normalize_request(request))[0]

    def submit(self, request: ExperimentRequest | dict[str, Any], key: str) -> SubmissionResult:
        request, key = normalize_request(request), idempotency_key(key)
        digest = request_hash(request)
        # Previously acknowledged submissions survive later capacity or source-integrity problems.
        with self.db.connection() as connection:
            existing = self.repository.idempotent_result(connection, key, digest)
        if existing:
            return existing
        preview, bars = self._prepare(request)
        result = SubmissionResult(experiment_id=str(uuid4()), job_id=str(uuid4()))
        # Files and hashes are prepared before the short write transaction. Unreferenced files are
        # removed on a losing concurrent submission; crashes can leave only an unindexed snapshot.
        directory = confined_path(self.settings.app_dir, f'experiment-bars/{result.experiment_id}')
        directory.mkdir(parents=True, exist_ok=False)
        path = confined_path(directory, 'bars.csv')
        keep = False
        try:
            with path.open('xb') as stream:
                stream.write(bars_bytes(bars))
                stream.flush()
                os.fsync(stream.fileno())
            now = utc_now()
            experiment = ExperimentRecord(experiment_id=result.experiment_id, job_id=result.job_id,
                created_at=now, request=request, dataset_id=request.dataset_id, dataset=self.datasets.get(request.dataset_id),
                request_sha256=digest, filtered_data_sha256=preview.filtered_data_sha256, splits=preview.splits,
                bars_relative_path=path.relative_to(self.settings.app_dir).as_posix())
            experiment.data_label = experiment.dataset.display_name
            experiment.is_synthetic = experiment.dataset.is_synthetic
            job = JobRecord(job_id=result.job_id, experiment_id=result.experiment_id, kind='train',
                            created_at=now, seed_count=len(request.seeds), requested_steps_total=request.timesteps * len(request.seeds))
            with self.db.transaction() as connection:
                existing = self.repository.idempotent_result(connection, key, digest)
                if existing:
                    return existing
                if self.jobs.queued_count(connection) >= self.settings.queue_capacity:
                    raise AppError('QUEUE_CAPACITY', '等待任务已达到上限，请待任务开始后重试。', 429)
                self.repository.insert(experiment, connection)
                self.jobs.insert(job, connection)
                self.repository.register_key(connection, key, digest, result, now)
            keep = True
            return result
        finally:
            if not keep:
                path.unlink(missing_ok=True)
                directory.rmdir()

    def get(self, experiment_id: str) -> ExperimentDetail:
        record = self.repository.get(experiment_id)
        if record.kind == 'research':
            raise AppError('RESEARCH_ROUTE_REQUIRED', '请使用研究页面读取此结果。', 409)
        return record.public()

    def list(self, *, limit: int = 20, cursor: str | None = None, kind: str | None = None,
             integrity: str | None = None, algorithm: str | None = None,
             dataset_id: str | None = None, status: str | None = None) -> Page[ExperimentDetail]:
        page = self.repository.list(limit=limit, cursor=cursor, kind=kind, integrity=integrity,
                                    algorithm=algorithm, dataset_id=dataset_id, status=status)
        return Page[ExperimentDetail](items=[record.public() for record in page.items],
                                      next_cursor=page.next_cursor, has_more=page.has_more)

    def capabilities(self) -> Capabilities:
        defaults = ExperimentRequest(dataset_id='00000000-0000-0000-0000-000000000000').model_dump(mode='json', exclude={'dataset_id'})
        return Capabilities(defaults=defaults, limits={
            'timesteps': {'min': 2, 'max': 1000000}, 'seeds': {'min_count': 1, 'max_count': 10, 'min': 0, 'max': 4294967295},
            'episode_length': {'min': 1, 'nullable': True}, 'train_ratio': {'exclusive_min': 0, 'exclusive_max': 1},
            'val_ratio': {'exclusive_min': 0, 'exclusive_max': 1}, 'ratio_sum_exclusive_max': 1,
            'minimum_rewards': {'train': 20, 'validation': 5, 'test': 5},
            'upload_bytes': self.settings.upload_limit_bytes, 'dataset_rows': self.settings.max_dataset_rows,
            'queued_jobs': self.settings.queue_capacity, 'executing_jobs': 1,
            'preview_rows': 100, 'page_default': 20, 'page_max': 100, 'event_page_default': 100, 'event_page_max': 500,
            'series_points_default': 5000, 'series_points_min': 100, 'series_points_max': 5000,
            'demo_defaults': {'rows': 756, 'seed': 42}, 'task_time_limit_seconds': self.settings.task_time_limit_seconds,
            'heartbeat_interval_seconds': self.settings.heartbeat_interval_seconds,
            'worker_unavailable_seconds': self.settings.worker_unavailable_seconds,
            'cancellation_grace_seconds': self.settings.cancellation_grace_seconds,
            'termination_wait_seconds': self.settings.termination_wait_seconds,
            'trading_config': request_schema_trading_limits()}, worker_available=self.jobs.worker_available())


def request_schema_trading_limits() -> dict[str, Any]:
    from .models import TradingConfigModel
    return TradingConfigModel.model_json_schema()['properties']
