"""Submit a new reproducibility task using only registered, fingerprinted sources."""
from pathlib import Path
from uuid import uuid4
import hashlib

from .artifact_protocol import hash_file, read_json
from .errors import AppError
from .experiments import idempotency_key
from .legacy import fixed_files, known_protocol
from .models import ExperimentRecord, JobRecord, SubmissionResult, utc_now
from .results import ResultService
from .settings import AppSettings, confined_path
from .storage.common import canonical_json
from .storage.database import Database
from .storage.experiments import ExperimentRepository
from .storage.jobs import JobRepository


def resolve_replay_source(settings: AppSettings, db: Database, source_id: str) -> Path:
    result_service = ResultService(settings, db)
    record = ExperimentRepository(db).get(source_id)
    if not record.replayable or record.integrity != 'complete' or not record.output_relative_path:
        raise AppError('REPLAY_UNAVAILABLE', record.replay_block_reason or '此实验不能重放。', 409)
    artifacts = result_service.artifacts.list(source_id)
    indexed = {item.filename: item for item in artifacts}
    prefix = 'replay-source/' if 'replay-source/summary.json' in indexed else ''
    summary_artifact = indexed.get(prefix + 'summary.json')
    if summary_artifact is None:
        raise AppError('ARTIFACT_CORRUPT', '缺少已登记的来源摘要。', 409)
    source = result_service.artifact_path(summary_artifact).parent
    summary = read_json(source / 'summary.json')
    if not known_protocol(summary):
        raise AppError('UNKNOWN_PROTOCOL', '不能确认来源训练协议，重放已禁用。', 409)
    for name in fixed_files(summary, reports=False):
        item = indexed.get(prefix + name)
        if item is None or result_service.artifact_path(item) != confined_path(source, name, must_exist=True):
            raise AppError('ARTIFACT_CORRUPT', '重放输入未登记或已改变。', 409)
    if hash_file(source / 'bars.csv') != summary['data_sha256']:
        raise AppError('ARTIFACT_CORRUPT', '来源行情指纹不一致。', 409)
    return source


class ReplayService:
    def __init__(self, settings: AppSettings, db: Database):
        self.settings, self.db = settings, db

    def submit(self, source_id: str, key: str) -> SubmissionResult:
        key = idempotency_key(key)
        repository = ExperimentRepository(self.db)
        # Scope idempotence to source identity, not mutable files. A successful retry
        # still returns the same IDs if the user's files subsequently disappear.
        request_hash = hashlib.sha256(canonical_json({'kind': 'replay', 'source_experiment_id': source_id}).encode()).hexdigest()
        with self.db.connection() as connection:
            existing = repository.idempotent_result(connection, key, request_hash)
        if existing:
            return existing
        resolve_replay_source(self.settings, self.db, source_id)
        source = repository.get(source_id)
        now = utc_now()
        experiment_id, job_id = str(uuid4()), str(uuid4())
        result = SubmissionResult(experiment_id=experiment_id, job_id=job_id)
        experiment = ExperimentRecord(experiment_id=experiment_id, kind='replay', source_experiment_id=source_id,
            job_id=job_id, created_at=now, request=source.request, legacy_config=source.legacy_config,
            dataset=source.dataset, dataset_id=source.dataset_id,
            data_label=source.data_label, is_synthetic=source.is_synthetic, splits=source.splits,
            request_sha256=source.request_sha256, filtered_data_sha256=source.filtered_data_sha256,
            bars_relative_path=source.bars_relative_path, versions=source.versions)
        job = JobRecord(job_id=job_id, experiment_id=experiment_id, kind='replay', created_at=now,
                        seed_count=len(source.runs), requested_steps_total=0, training_fraction=None)
        with self.db.transaction() as connection:
            existing = repository.idempotent_result(connection, key, request_hash)
            if existing:
                return existing
            if JobRepository.queued_count(connection) >= self.settings.queue_capacity:
                raise AppError('QUEUE_FULL', '任务队列已满，请稍后重试。', 429)
            repository.insert(experiment, connection)
            JobRepository(self.db).insert(job, connection)
            repository.register_key(connection, key, request_hash, result, now)
        return result
