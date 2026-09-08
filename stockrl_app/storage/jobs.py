"""Durable job state; ownership and revision checks are enforced under one write lock."""
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
import sqlite3

from ..errors import AppError
from ..models import (ArtifactStorageRecord, ErrorSummary, EventPage, ExperimentRecord, JobDetail,
                      JobEvent, JobRecord, JobStatus, Page, Phase, PhaseEventPayload,
                      ProgressEventPayload, StateEventPayload, WorkerInstance, utc_now)
from .artifacts import ArtifactRepository
from .common import page_rows
from .database import Database
from .experiments import ExperimentRepository

TERMINAL_STATUSES = frozenset({'succeeded', 'failed', 'cancelled', 'interrupted'})
TRANSITIONS = {'queued': {'running', 'cancelled'},
               'running': {'cancelling', 'succeeded', 'failed', 'interrupted'},
               'cancelling': {'cancelled', 'failed', 'interrupted'}}
EVENT_PAYLOADS = {'state': StateEventPayload, 'phase': PhaseEventPayload,
                  'progress': ProgressEventPayload, 'error': ErrorSummary}


class JobRepository:
    def __init__(self, db: Database, worker_unavailable_seconds: int = 30) -> None:
        self.db = db
        self.worker_unavailable_seconds = worker_unavailable_seconds

    def get(self, job_id: str, connection: sqlite3.Connection | None = None) -> JobRecord:
        if connection is None:
            with self.db.connection() as reader:
                return self.get(job_id, reader)
        row = connection.execute('SELECT payload_json FROM jobs WHERE job_id=?', (job_id,)).fetchone()
        if row is None:
            raise AppError('JOB_NOT_FOUND', '任务不存在。', 404)
        return JobRecord.model_validate_json(row['payload_json'])

    def detail(self, job_id: str) -> JobDetail:
        with self.db.connection() as connection:
            record = self.get(job_id, connection)
            worker = self.latest_worker(connection)
        record.worker_available = self.worker_available(worker)
        record.recovery_waiting = bool(worker and worker.recovery_waiting)
        return record.public()

    def list(self, *, limit: int = 20, cursor: str | None = None, status: str | None = None,
             kind: str | None = None) -> Page[JobRecord]:
        filters = {key: value for key, value in {'status': status, 'kind': kind}.items() if value is not None}
        with self.db.connection() as connection:
            return page_rows(connection, 'jobs', 'job_id', JobRecord, limit=limit, cursor=cursor, filters=filters)

    def list_details(self, *, limit: int = 20, cursor: str | None = None, status: str | None = None,
                     kind: str | None = None) -> Page[JobDetail]:
        page = self.list(limit=limit, cursor=cursor, status=status, kind=kind)
        worker = self.latest_worker()
        available = self.worker_available(worker)
        items = []
        for record in page.items:
            record.worker_available = available
            record.recovery_waiting = bool(worker and worker.recovery_waiting)
            items.append(record.public())
        return Page[JobDetail](items=items, next_cursor=page.next_cursor, has_more=page.has_more)

    def insert(self, record: JobRecord, connection: sqlite3.Connection) -> None:
        if record.status != 'queued' or record.revision != 0 or record.owner_token is not None:
            raise AppError('STATE_CONFLICT', '新任务必须处于等待状态。', 409)
        connection.execute('''INSERT INTO jobs(job_id,experiment_id,kind,status,created_at,revision,owner_token,
            cancel_requested,payload_json) VALUES (?,?,?,?,?,?,?,?,?)''',
            (record.job_id, record.experiment_id, record.kind, record.status, record.created_at,
             record.revision, record.owner_token, int(record.cancel_requested), record.model_dump_json()))
        self._event(connection, record.job_id, 'state', StateEventPayload(status='queued', revision=0))

    @staticmethod
    def queued_count(connection: sqlite3.Connection) -> int:
        return connection.execute("SELECT count(*) FROM jobs WHERE status='queued'").fetchone()[0]

    @staticmethod
    def _owned(record: JobRecord, owner_token: str, revision: int) -> None:
        if record.owner_token != owner_token or not owner_token or record.revision != revision:
            raise AppError('STATE_CONFLICT', '任务状态已更新，请重新读取。', 409)

    @staticmethod
    def _save(connection: sqlite3.Connection, record: JobRecord, previous_revision: int) -> None:
        record.revision = previous_revision + 1
        result = connection.execute('''UPDATE jobs SET status=?,revision=?,owner_token=?,cancel_requested=?,payload_json=?
            WHERE job_id=? AND revision=?''', (record.status, record.revision, record.owner_token,
            int(record.cancel_requested), record.model_dump_json(), record.job_id, previous_revision))
        if result.rowcount != 1:
            raise AppError('STATE_CONFLICT', '任务状态已更新，请重新读取。', 409)

    def claim_next(self, owner_token: str) -> JobRecord | None:
        if not owner_token:
            raise AppError('INVALID_OWNER', '执行实例标识无效。', 409)
        with self.db.transaction() as connection:
            if connection.execute("SELECT 1 FROM jobs WHERE status IN ('running','cancelling') LIMIT 1").fetchone():
                return None
            row = connection.execute("SELECT job_id FROM jobs WHERE status='queued' ORDER BY created_at,job_id LIMIT 1").fetchone()
            if row is None:
                return None
            record = self.get(row['job_id'], connection)
            record.owner_token = owner_token
            record.status = 'running'
            record.started_at = utc_now()
            record.heartbeat_at = record.started_at
            record.phase = 'preparing'
            self._save(connection, record, record.revision)
            self._event(connection, record.job_id, 'state', StateEventPayload(status=record.status, revision=record.revision))
            self._event(connection, record.job_id, 'phase', PhaseEventPayload(phase='preparing'))
            return record

    def transition(self, job_id: str, status: JobStatus, *, owner_token: str, revision: int,
                   error: ErrorSummary | None = None,
                   connection: sqlite3.Connection | None = None) -> JobRecord:
        if status == 'succeeded':
            raise AppError('PUBLISH_REQUIRED', '成功状态必须经核验发布事务写入。', 409)
        if connection is None:
            with self.db.transaction() as transaction:
                return self.transition(job_id, status, owner_token=owner_token, revision=revision,
                                       error=error, connection=transaction)
        record = self.get(job_id, connection)
        self._owned(record, owner_token, revision)
        if status not in TRANSITIONS.get(record.status, set()) or status == 'running':
            raise AppError('STATE_CONFLICT', '不允许此任务状态变更。', 409)
        record.status = status
        if status == 'cancelling':
            record.cancel_requested = True
        if status in TERMINAL_STATUSES:
            record.finished_at = utc_now()
        record.error = error
        self._save(connection, record, revision)
        self._event(connection, job_id, 'state', StateEventPayload(status=status, revision=record.revision))
        if error:
            self._event(connection, job_id, 'error', error)
        if status in TERMINAL_STATUSES:
            experiment = ExperimentRepository(self.db).get(record.experiment_id, connection)
            experiment.integrity = 'partial'
            experiment.replayable = False
            experiment.replay_block_reason = '任务未完整完成，不能重放。'
            ExperimentRepository(self.db).update(experiment, connection)
        return record

    def request_cancel(self, job_id: str) -> JobRecord:
        with self.db.transaction() as connection:
            record = self.get(job_id, connection)
            if record.status in TERMINAL_STATUSES or record.status == 'cancelling':
                return record
            record.status = 'cancelled' if record.status == 'queued' else 'cancelling'
            record.cancel_requested = True
            if record.status == 'cancelled':
                record.finished_at = utc_now()
            self._save(connection, record, record.revision)
            self._event(connection, job_id, 'state', StateEventPayload(status=record.status, revision=record.revision))
            if record.status == 'cancelled':
                experiment = ExperimentRepository(self.db).get(record.experiment_id, connection)
                experiment.replay_block_reason = '任务已取消，没有完整产物。'
                ExperimentRepository(self.db).update(experiment, connection)
            return record

    def update_progress(self, job_id: str, *, owner_token: str, revision: int,
                        phase: Phase | None = None, seed: int | None = None,
                        seed_index: int | None = None, actual_steps: int | None = None,
                        completed_seeds: int | None = None) -> JobRecord:
        with self.db.transaction() as connection:
            record = self.get(job_id, connection)
            self._owned(record, owner_token, revision)
            if record.status != 'running':
                raise AppError('STATE_CONFLICT', '任务已不在运行状态。', 409)
            old_phase, old_seed = record.phase, record.seed
            experiment = ExperimentRepository(self.db).get(record.experiment_id, connection)
            request = experiment.request
            allowed_seeds = request.seeds if request is not None else []
            if request is None and record.kind == 'replay' and experiment.source_experiment_id:
                source = ExperimentRepository(self.db).get(experiment.source_experiment_id, connection)
                allowed_seeds = [run['seed'] for run in source.runs if type(run.get('seed')) is int]
            if seed_index is not None and (type(seed_index) is not int or not 0 <= seed_index < record.seed_count):
                raise AppError('INVALID_PROGRESS', 'seed 顺序无效。', 409)
            if seed is not None:
                if seed not in allowed_seeds:
                    raise AppError('INVALID_PROGRESS', 'seed 不属于此任务。', 409)
                expected = allowed_seeds.index(seed)
                if seed_index is not None and seed_index != expected:
                    raise AppError('INVALID_PROGRESS', 'seed 顺序与配置不一致。', 409)
                record.seed, record.seed_index = seed, expected
            if phase is not None:
                record.phase = PhaseEventPayload(phase=phase).phase
            if actual_steps is not None:
                if type(actual_steps) is not int or actual_steps < 0 or record.seed is None or record.kind != 'train':
                    raise AppError('INVALID_PROGRESS', '训练步数无效。', 409)
                name = str(record.seed)
                record.progress_steps[name] = max(actual_steps, record.progress_steps.get(name, 0))
                record.actual_steps_total = sum(record.progress_steps.values())
                budget = request.timesteps
                record.training_fraction = sum(min(count, budget) for count in record.progress_steps.values()) / record.requested_steps_total
            if completed_seeds is not None:
                if type(completed_seeds) is not int or not record.completed_seeds <= completed_seeds <= record.seed_count:
                    raise AppError('INVALID_PROGRESS', '已完成 seed 数无效。', 409)
                record.completed_seeds = completed_seeds
            self._save(connection, record, revision)
            if record.phase != old_phase or record.seed != old_seed:
                self._event(connection, job_id, 'phase', PhaseEventPayload(phase=record.phase, seed=record.seed, seed_index=record.seed_index))
            last = connection.execute("SELECT occurred_at FROM job_events WHERE job_id=? AND event_type='progress' ORDER BY seq DESC LIMIT 1", (job_id,)).fetchone()
            now = datetime.now(timezone.utc)
            if last is None or (now - datetime.fromisoformat(last['occurred_at'].replace('Z', '+00:00'))).total_seconds() >= 1:
                self._event(connection, job_id, 'progress', ProgressEventPayload(seed=record.seed, seed_index=record.seed_index,
                    actual_steps_total=record.actual_steps_total, requested_steps_total=record.requested_steps_total,
                    completed_seeds=record.completed_seeds, training_fraction=record.training_fraction))
            return record

    @staticmethod
    def _event(connection: sqlite3.Connection, job_id: str, event_type: str,
               payload: StateEventPayload | PhaseEventPayload | ProgressEventPayload | ErrorSummary) -> JobEvent:
        if event_type not in EVENT_PAYLOADS or not isinstance(payload, EVENT_PAYLOADS[event_type]):
            raise AppError('INVALID_EVENT', '事件数据不符合约定。', 409)
        seq = connection.execute('SELECT coalesce(max(seq),0)+1 FROM job_events WHERE job_id=?', (job_id,)).fetchone()[0]
        event = JobEvent(job_id=job_id, seq=seq, occurred_at=utc_now(), event_type=event_type, payload=payload)
        connection.execute('INSERT INTO job_events VALUES (?,?,?,?,?)',
                           (job_id, seq, event.occurred_at, event_type, event.model_dump_json()))
        return event

    def append_event(self, job_id: str, event_type: str,
                     payload: StateEventPayload | PhaseEventPayload | ProgressEventPayload | ErrorSummary,
                     *, owner_token: str, revision: int) -> JobEvent:
        """Diagnostic events only; state/phase/progress events are emitted by their mutations."""
        if event_type != 'error':
            raise AppError('INVALID_EVENT', '状态和进度事件必须随状态更新保存。', 409)
        with self.db.transaction() as connection:
            record = self.get(job_id, connection)
            self._owned(record, owner_token, revision)
            if record.status in TERMINAL_STATUSES:
                raise AppError('STATE_CONFLICT', '已结束任务不可追加执行事件。', 409)
            return self._event(connection, job_id, event_type, payload)

    def events(self, job_id: str, *, after_seq: int = 0, limit: int = 100) -> EventPage:
        if type(after_seq) is not int or after_seq < 0 or type(limit) is not int or not 1 <= limit <= 500:
            raise AppError('INVALID_PAGINATION', '事件游标或条数无效。')
        with self.db.connection() as connection:
            self.get(job_id, connection)
            rows = connection.execute('SELECT payload_json FROM job_events WHERE job_id=? AND seq>? ORDER BY seq LIMIT ?',
                                      (job_id, after_seq, limit + 1)).fetchall()
        items = [JobEvent.model_validate_json(row['payload_json']) for row in rows[:limit]]
        return EventPage(items=items, next_seq=items[-1].seq if items else after_seq, has_more=len(rows) > limit)

    def heartbeat(self, owner_token: str, *, active_job_id: str | None = None,
                  recovery_waiting: bool = False, pid: int | None = None) -> WorkerInstance:
        with self.db.transaction() as connection:
            row = connection.execute('SELECT payload_json FROM worker_instances WHERE owner_token=?', (owner_token,)).fetchone()
            now = utc_now()
            worker = WorkerInstance(owner_token=owner_token, started_at=WorkerInstance.model_validate_json(row[0]).started_at if row else now,
                                    heartbeat_at=now, active_job_id=active_job_id, recovery_waiting=recovery_waiting, pid=pid)
            if active_job_id:
                job = self.get(active_job_id, connection)
                if job.owner_token != owner_token or job.status not in ('running', 'cancelling'):
                    raise AppError('STATE_CONFLICT', '执行实例已不拥有此任务。', 409)
                # Heartbeats do not advance work revision or invalidate in-flight progress updates.
                job.heartbeat_at = now
                connection.execute('UPDATE jobs SET payload_json=? WHERE job_id=? AND owner_token=?',
                                   (job.model_dump_json(), active_job_id, owner_token))
            connection.execute('''INSERT INTO worker_instances VALUES (?,?,?,?,?,?,?) ON CONFLICT(owner_token) DO UPDATE SET
                heartbeat_at=excluded.heartbeat_at,recovery_waiting=excluded.recovery_waiting,
                active_job_id=excluded.active_job_id,pid=excluded.pid,payload_json=excluded.payload_json''',
                (owner_token, worker.started_at, now, int(recovery_waiting), active_job_id, pid, worker.model_dump_json()))
            return worker

    def latest_worker(self, connection: sqlite3.Connection | None = None) -> WorkerInstance | None:
        if connection is None:
            with self.db.connection() as reader:
                return self.latest_worker(reader)
        row = connection.execute('SELECT payload_json FROM worker_instances ORDER BY heartbeat_at DESC,owner_token LIMIT 1').fetchone()
        return WorkerInstance.model_validate_json(row[0]) if row else None

    def worker_available(self, worker: WorkerInstance | None = None) -> bool:
        worker = worker or self.latest_worker()
        if worker is None:
            return False
        last = datetime.fromisoformat(worker.heartbeat_at.replace('Z', '+00:00'))
        return datetime.now(timezone.utc) - last <= timedelta(seconds=self.worker_unavailable_seconds)

    def publish_intent(self, job_id: str, *, owner_token: str, revision: int,
                       target_relative_path: str, manifest_sha256: str) -> JobRecord:
        from pathlib import PurePosixPath, PureWindowsPath
        path = PurePosixPath(target_relative_path)
        if not target_relative_path or path.is_absolute() or '..' in path.parts or '\\' in target_relative_path or PureWindowsPath(target_relative_path).drive:
            raise AppError('INVALID_PUBLISH', '发布目标必须是受控相对目录。', 409)
        if len(manifest_sha256) != 64 or any(char not in '0123456789abcdef' for char in manifest_sha256):
            raise AppError('INVALID_PUBLISH', '发布指纹无效。', 409)
        with self.db.transaction() as connection:
            record = self.get(job_id, connection)
            self._owned(record, owner_token, revision)
            if record.status != 'running' or record.cancel_requested:
                raise AppError('STATE_CONFLICT', '任务已停止或正在取消，不能发布。', 409)
            record.publish_target = target_relative_path
            record.publish_manifest_sha256 = manifest_sha256
            record.publish_owner_token = owner_token
            record.publish_revision = revision + 1
            record.phase = 'publishing'
            self._save(connection, record, revision)
            self._event(connection, job_id, 'phase', PhaseEventPayload(phase='publishing', seed=record.seed, seed_index=record.seed_index))
            return record

    def publish_success(self, job_id: str, *, owner_token: str, revision: int,
                        experiment: ExperimentRecord, artifacts: list[ArtifactStorageRecord],
                        move: Callable[[], None]) -> JobRecord:
        """Caller verifies files/hashes before intent; callback performs only final rename (or recovery no-op)."""
        with self.db.transaction() as connection:
            record = self.get(job_id, connection)
            self._owned(record, owner_token, revision)
            if (record.status != 'running' or record.cancel_requested or not record.publish_manifest_sha256
                    or record.publish_owner_token != owner_token or record.publish_revision != revision
                    or record.publish_target != experiment.output_relative_path
                    or record.experiment_id != experiment.experiment_id or experiment.integrity != 'complete'
                    or record.completed_seeds != record.seed_count or not artifacts):
                raise AppError('PUBLISH_CONFLICT', '发布凭据或任务状态不一致。', 409)
            if any(item.experiment_id != experiment.experiment_id for item in artifacts):
                raise AppError('PUBLISH_CONFLICT', '产物不属于此实验。', 409)
            move()
            repository = ArtifactRepository(self.db)
            for item in artifacts:
                repository.insert(item, connection)
            experiment.artifacts = [item.public() for item in artifacts]
            ExperimentRepository(self.db).update(experiment, connection)
            record.status = 'succeeded'
            record.finished_at = utc_now()
            if record.kind == 'train':
                record.training_fraction = 1.0
            self._save(connection, record, revision)
            self._event(connection, job_id, 'state', StateEventPayload(status='succeeded', revision=record.revision))
            return record
