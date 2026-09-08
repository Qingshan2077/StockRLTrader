"""Explicit creation and backup. No module import or repository performs migration."""
import os
from pathlib import Path
import sqlite3

from ..errors import AppError
from ..models import utc_now
from .database import Database, SCHEMA_VERSION, check_schema

DDL = (
    'CREATE TABLE schema_migrations(version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)',
    '''CREATE TABLE datasets(dataset_id TEXT PRIMARY KEY, created_at TEXT NOT NULL,
        snapshot_sha256 TEXT NOT NULL, payload_json TEXT NOT NULL)''',
    '''CREATE TABLE experiments(experiment_id TEXT PRIMARY KEY, created_at TEXT NOT NULL,
        dataset_id TEXT REFERENCES datasets(dataset_id), kind TEXT NOT NULL CHECK(kind IN ('train','replay')),
        source_experiment_id TEXT REFERENCES experiments(experiment_id),
        job_id TEXT UNIQUE REFERENCES jobs(job_id) DEFERRABLE INITIALLY DEFERRED,
        integrity TEXT NOT NULL CHECK(integrity IN ('pending','complete','partial','corrupt','unsupported')),
        output_relative_path TEXT, legacy_fingerprint TEXT, payload_json TEXT NOT NULL,
        UNIQUE(output_relative_path, legacy_fingerprint))''',
    '''CREATE TABLE jobs(job_id TEXT PRIMARY KEY, experiment_id TEXT NOT NULL UNIQUE REFERENCES experiments(experiment_id),
        kind TEXT NOT NULL CHECK(kind IN ('train','replay')), status TEXT NOT NULL
        CHECK(status IN ('queued','running','cancelling','succeeded','failed','cancelled','interrupted')),
        created_at TEXT NOT NULL, revision INTEGER NOT NULL DEFAULT 0 CHECK(revision>=0), owner_token TEXT,
        cancel_requested INTEGER NOT NULL DEFAULT 0 CHECK(cancel_requested IN (0,1)),
        payload_json TEXT NOT NULL)''',
    '''CREATE TABLE job_events(job_id TEXT NOT NULL REFERENCES jobs(job_id), seq INTEGER NOT NULL CHECK(seq>0),
        occurred_at TEXT NOT NULL, event_type TEXT NOT NULL CHECK(event_type IN ('state','phase','progress','error')),
        payload_json TEXT NOT NULL, PRIMARY KEY(job_id,seq))''',
    '''CREATE TABLE artifacts(artifact_id TEXT PRIMARY KEY, experiment_id TEXT NOT NULL REFERENCES experiments(experiment_id),
        root_id TEXT NOT NULL CHECK(root_id IN ('app','output')), relative_path TEXT NOT NULL,
        kind TEXT NOT NULL, size INTEGER NOT NULL CHECK(size>=0), sha256 TEXT NOT NULL,
        payload_json TEXT NOT NULL, UNIQUE(experiment_id,root_id,relative_path))''',
    '''CREATE TABLE idempotency_keys(key TEXT PRIMARY KEY, request_sha256 TEXT NOT NULL,
        experiment_id TEXT NOT NULL REFERENCES experiments(experiment_id),
        job_id TEXT NOT NULL REFERENCES jobs(job_id), created_at TEXT NOT NULL)''',
    '''CREATE TABLE worker_instances(owner_token TEXT PRIMARY KEY, started_at TEXT NOT NULL,
        heartbeat_at TEXT NOT NULL, recovery_waiting INTEGER NOT NULL DEFAULT 0,
        active_job_id TEXT REFERENCES jobs(job_id), pid INTEGER, payload_json TEXT NOT NULL)''',
    'CREATE INDEX datasets_page ON datasets(created_at DESC,dataset_id DESC)',
    'CREATE INDEX experiments_page ON experiments(created_at DESC,experiment_id DESC)',
    'CREATE INDEX jobs_queue ON jobs(status,created_at,job_id)',
    'CREATE INDEX workers_heartbeat ON worker_instances(heartbeat_at DESC)',
)


def initialize_database(path: str | Path) -> Database:
    """Initialize an empty file or return a known current database; reject all other schemas."""
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, isolation_level=None, timeout=5)
    try:
        connection.execute('PRAGMA foreign_keys=ON')
        connection.execute('PRAGMA busy_timeout=5000')
        tables = connection.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        version = connection.execute('PRAGMA user_version').fetchone()[0]
        if tables or version:
            check_schema(connection)
            return Database(path)
        connection.execute('PRAGMA journal_mode=WAL')
        connection.execute('BEGIN IMMEDIATE')
        try:
            # Recheck under the write lock, including simultaneous explicit initialization.
            tables = connection.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
            if tables:
                check_schema(connection)
            else:
                for statement in DDL:
                    connection.execute(statement)
                connection.execute('INSERT INTO schema_migrations VALUES (?,?)', (SCHEMA_VERSION, utc_now()))
                connection.execute(f'PRAGMA user_version={SCHEMA_VERSION}')
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
    finally:
        connection.close()
    return Database(path)


def backup_database(source: str | Path, destination: str | Path) -> Path:
    """Use SQLite online backup to include committed WAL pages; never overwrite a backup."""
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source == destination:
        raise AppError('BACKUP_CONFLICT', '备份不能覆盖应用数据库。', 409)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(destination, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise AppError('BACKUP_CONFLICT', '备份文件已存在，请指定新文件名。', 409) from exc
    os.close(descriptor)
    try:
        with Database(source).connection() as current:
            backup = sqlite3.connect(destination)
            try:
                current.backup(backup)
                check_schema(backup)
                if backup.execute('PRAGMA integrity_check').fetchone()[0] != 'ok':
                    raise AppError('BACKUP_INVALID', '备份完整性检查失败。', 409)
                if backup.execute('PRAGMA foreign_key_check').fetchone() is not None:
                    raise AppError('BACKUP_INVALID', '备份关联完整性检查失败。', 409)
            finally:
                backup.close()
    except BaseException:
        destination.unlink(missing_ok=True)
        raise
    return destination


def migrate_database(path: str | Path, backup_path: str | Path) -> Path:
    """Explicit maintenance entry. V1 has no older application schema to upgrade."""
    with Database(path).connection():
        pass
    return backup_database(path, backup_path)
