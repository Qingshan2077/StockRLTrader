import sqlite3
from uuid import uuid4

import pytest

from stockrl_app.datasets import DatasetService
from stockrl_app.errors import AppError
from stockrl_app.experiments import ExperimentService
from stockrl_app.models import DemoDatasetRequest, ExperimentRequest
from stockrl_app.settings import AppSettings
from stockrl_app.storage.database import Database
from stockrl_app.storage.jobs import JobRepository
from stockrl_app.storage.migrations import backup_database, initialize_database


@pytest.fixture
def foundation(tmp_path):
    settings = AppSettings(project_root=tmp_path)
    initialize_database(settings.database_path)
    db = Database(settings.database_path)
    dataset = DatasetService(settings, db).create_demo(DemoDatasetRequest(rows=60))
    return settings, db, dataset


def test_open_does_not_initialize_unknown_database(tmp_path):
    db = Database(tmp_path / 'missing.sqlite3')
    with pytest.raises(AppError) as error:
        with db.connection():
            pass
    assert error.value.code == 'DATABASE_NOT_INITIALIZED'
    assert not db.path.exists()


def test_reconnect_idempotency_and_foreign_keys(foundation):
    settings, db, dataset = foundation
    key = str(uuid4())
    request = ExperimentRequest(dataset_id=dataset.dataset_id)
    first = ExperimentService(settings, db).submit(request, key)
    second = ExperimentService(settings, Database(db.path)).submit(request, key)
    assert first == second
    with db.connection() as connection:
        assert connection.execute('PRAGMA foreign_keys').fetchone()[0] == 1
        assert connection.execute('PRAGMA busy_timeout').fetchone()[0] == 5000
        assert connection.execute('PRAGMA journal_mode').fetchone()[0] == 'wal'
        assert connection.execute('SELECT count(*) FROM jobs').fetchone()[0] == 1
    with pytest.raises(sqlite3.IntegrityError), db.transaction() as connection:
        connection.execute("INSERT INTO job_events(job_id, seq, occurred_at, event_type, payload_json) VALUES ('missing',1,'now','state','{}')")


def test_revision_and_owner_cannot_overwrite(foundation):
    settings, db, dataset = foundation
    result = ExperimentService(settings, db).submit(ExperimentRequest(dataset_id=dataset.dataset_id), str(uuid4()))
    jobs = JobRepository(db)
    running = jobs.claim_next('owner-a')
    assert running.job_id == result.job_id
    with pytest.raises(AppError):
        jobs.transition(running.job_id, 'failed', owner_token='owner-b', revision=running.revision)
    cancelling = jobs.request_cancel(running.job_id)
    with pytest.raises(AppError):
        jobs.transition(running.job_id, 'succeeded', owner_token='owner-a', revision=running.revision)
    assert jobs.get(running.job_id).status == cancelling.status == 'cancelling'


def test_backup_contains_recent_wal_commit_and_future_version_rejected(foundation, tmp_path):
    settings, db, dataset = foundation
    result = ExperimentService(settings, db).submit(ExperimentRequest(dataset_id=dataset.dataset_id), str(uuid4()))
    destination = tmp_path / 'backup.sqlite3'
    backup_database(db.path, destination)
    assert JobRepository(Database(destination)).get(result.job_id).status == 'queued'
    with sqlite3.connect(db.path) as connection:
        connection.execute('PRAGMA user_version = 999')
    with pytest.raises(AppError) as error, db.transaction():
        pass
    assert error.value.code == 'SCHEMA_INCOMPATIBLE'


def test_initialization_failure_rolls_back_schema(tmp_path, monkeypatch):
    from stockrl_app.storage import migrations
    path = tmp_path / 'new.sqlite3'
    monkeypatch.setattr(migrations, 'DDL', (*migrations.DDL[:2], 'INVALID SQL STATEMENT'))
    with pytest.raises(sqlite3.OperationalError):
        initialize_database(path)
    with sqlite3.connect(path) as connection:
        assert connection.execute('PRAGMA user_version').fetchone()[0] == 0
        assert connection.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall() == []


def test_progress_is_cumulative_and_events_have_stable_sequence(foundation):
    settings, db, dataset = foundation
    request = ExperimentRequest(dataset_id=dataset.dataset_id, seeds=[42, 7], timesteps=10)
    result = ExperimentService(settings, db).submit(request, str(uuid4()))
    jobs = JobRepository(db)
    record = jobs.claim_next('worker')
    record = jobs.update_progress(record.job_id, owner_token='worker', revision=record.revision,
                                   phase='training', seed=42, seed_index=0, actual_steps=16, completed_seeds=1)
    assert record.actual_steps_total == 16 and record.training_fraction == .5
    record = jobs.update_progress(record.job_id, owner_token='worker', revision=record.revision,
                                   phase='training', seed=7, seed_index=1, actual_steps=2)
    assert record.actual_steps_total == 18 and record.training_fraction == .6
    first = jobs.events(result.job_id, limit=2)
    remaining = jobs.events(result.job_id, after_seq=first.next_seq)
    sequences = [event.seq for event in first.items + remaining.items]
    assert sequences == list(range(1, len(sequences) + 1))
    assert first.has_more
    assert not remaining.has_more


def test_cancel_is_idempotent_and_blocks_further_claims(foundation):
    settings, db, dataset = foundation
    experiments = ExperimentService(settings, db)
    result = experiments.submit(ExperimentRequest(dataset_id=dataset.dataset_id), str(uuid4()))
    jobs = JobRepository(db)
    queued_cancel = jobs.request_cancel(result.job_id)
    assert queued_cancel.status == 'cancelled'
    assert jobs.request_cancel(result.job_id).revision == queued_cancel.revision
    assert jobs.claim_next('worker') is None
    experiments.submit(ExperimentRequest(dataset_id=dataset.dataset_id), str(uuid4()))
    active = jobs.claim_next('worker')
    experiments.submit(ExperimentRequest(dataset_id=dataset.dataset_id), str(uuid4()))
    assert jobs.claim_next('another-worker') is None
    cancelling = jobs.request_cancel(active.job_id)
    assert cancelling.status == 'cancelling'
    assert jobs.claim_next('another-worker') is None


def test_publish_intent_survives_reconnect_and_cancellation_wins(foundation):
    settings, db, dataset = foundation
    result = ExperimentService(settings, db).submit(ExperimentRequest(dataset_id=dataset.dataset_id), str(uuid4()))
    jobs = JobRepository(db)
    record = jobs.claim_next('worker')
    record = jobs.publish_intent(record.job_id, owner_token='worker', revision=record.revision,
                                 target_relative_path='published-new', manifest_sha256='a' * 64)
    reconnected = JobRepository(Database(db.path)).get(result.job_id)
    assert reconnected.publish_manifest_sha256 == 'a' * 64
    assert reconnected.publish_owner_token == 'worker'
    assert reconnected.publish_revision == record.revision
    jobs.request_cancel(record.job_id)
    with pytest.raises(AppError):
        jobs.publish_intent(record.job_id, owner_token='worker', revision=record.revision,
                             target_relative_path='published-new', manifest_sha256='a' * 64)


def test_heartbeat_preserves_work_revision_and_public_record_hides_owner(foundation):
    settings, db, dataset = foundation
    result = ExperimentService(settings, db).submit(ExperimentRequest(dataset_id=dataset.dataset_id), str(uuid4()))
    jobs = JobRepository(db)
    record = jobs.claim_next('private-token')
    jobs.heartbeat('private-token', active_job_id=record.job_id)
    assert jobs.get(record.job_id).revision == record.revision
    detail = jobs.detail(result.job_id)
    assert detail.worker_available
    assert 'owner_token' not in detail.model_dump()
    assert 'publish_target' not in detail.model_dump()
