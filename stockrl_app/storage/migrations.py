"""Explicit creation and backup. No module import or repository performs migration."""
import os
from datetime import datetime, timezone
from pathlib import Path
import sqlite3

from ..errors import AppError
from ..models import utc_now
from .database import Database, SCHEMA_VERSION, check_schema, migration_lock_path

DDL = (
    'CREATE TABLE schema_migrations(version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)',
    '''CREATE TABLE datasets(dataset_id TEXT PRIMARY KEY, created_at TEXT NOT NULL,
        snapshot_sha256 TEXT NOT NULL, payload_json TEXT NOT NULL)''',
    '''CREATE TABLE experiments(experiment_id TEXT PRIMARY KEY, created_at TEXT NOT NULL,
        dataset_id TEXT REFERENCES datasets(dataset_id), kind TEXT NOT NULL CHECK(kind IN ('train','replay','research')),
        source_experiment_id TEXT REFERENCES experiments(experiment_id),
        job_id TEXT UNIQUE REFERENCES jobs(job_id) DEFERRABLE INITIALLY DEFERRED,
        integrity TEXT NOT NULL CHECK(integrity IN ('pending','complete','partial','corrupt','unsupported')),
        output_relative_path TEXT, legacy_fingerprint TEXT, payload_json TEXT NOT NULL,
        UNIQUE(output_relative_path, legacy_fingerprint))''',
    '''CREATE TABLE jobs(job_id TEXT PRIMARY KEY, experiment_id TEXT NOT NULL REFERENCES experiments(experiment_id),
        kind TEXT NOT NULL CHECK(kind IN ('train','replay','research')), status TEXT NOT NULL
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


V2_DDL = (
    '''CREATE TABLE market_datasets_v2(dataset_id TEXT PRIMARY KEY, created_at TEXT NOT NULL,
        snapshot_sha256 TEXT NOT NULL, bundle_relative_path TEXT NOT NULL, payload_json TEXT NOT NULL)''',
    '''CREATE TABLE researches(research_id TEXT PRIMARY KEY REFERENCES experiments(experiment_id),
        created_at TEXT NOT NULL, status TEXT NOT NULL, protocol_fingerprint TEXT NOT NULL,
        execution_environment_fingerprint TEXT NOT NULL, latest_attempt_id TEXT REFERENCES research_attempts(attempt_id)
        DEFERRABLE INITIALLY DEFERRED, payload_json TEXT NOT NULL)''',
    '''CREATE TABLE research_attempts(attempt_id TEXT PRIMARY KEY, research_id TEXT NOT NULL REFERENCES researches(research_id),
        job_id TEXT NOT NULL UNIQUE REFERENCES jobs(job_id), status TEXT NOT NULL,
        created_at TEXT NOT NULL, payload_json TEXT NOT NULL)''',
    '''CREATE UNIQUE INDEX research_active_attempt ON research_attempts(research_id)
        WHERE status IN ('queued','running','cancelling')''',
    '''CREATE TABLE research_units(execution_key TEXT NOT NULL UNIQUE,
        research_id TEXT NOT NULL REFERENCES researches(research_id), instrument_id TEXT NOT NULL,
        fold_id TEXT NOT NULL, seed INTEGER NOT NULL, status TEXT NOT NULL,
        attempt_id TEXT REFERENCES research_attempts(attempt_id), revision INTEGER NOT NULL DEFAULT 0 CHECK(revision>=0),
        owner_token TEXT, artifact_root TEXT, manifest_sha256 TEXT, started_at TEXT, finished_at TEXT,
        error_json TEXT, payload_json TEXT NOT NULL, PRIMARY KEY(research_id,instrument_id,fold_id,seed))''',
    '''CREATE TABLE exposure_records(exposure_id TEXT PRIMARY KEY, research_id TEXT REFERENCES researches(research_id),
        instrument_id TEXT NOT NULL, interval_start TEXT NOT NULL, interval_end TEXT NOT NULL,
        exposed_at TEXT NOT NULL, payload_json TEXT NOT NULL)''',
    '''CREATE TABLE artifact_protocol_registry(artifact_id TEXT PRIMARY KEY,
        research_id TEXT NOT NULL REFERENCES researches(research_id),
        execution_key TEXT REFERENCES research_units(execution_key), protocol_fingerprint TEXT NOT NULL,
        manifest_sha256 TEXT NOT NULL, payload_json TEXT NOT NULL)''',
    'CREATE INDEX market_datasets_v2_page ON market_datasets_v2(created_at DESC,dataset_id DESC)',
    'CREATE INDEX researches_page ON researches(created_at DESC,research_id DESC)',
    'CREATE INDEX research_units_status ON research_units(research_id,status)',
    'CREATE INDEX research_attempts_history ON research_attempts(research_id,created_at,attempt_id)',
    'CREATE INDEX exposures_interval ON exposure_records(instrument_id,interval_start,interval_end)',
    'CREATE INDEX artifact_protocols ON artifact_protocol_registry(protocol_fingerprint)',
    'CREATE INDEX jobs_experiment ON jobs(experiment_id)',
)
DDL = (*DDL, *V2_DDL)


def initialize_database(path: str | Path) -> Database:
    """Initialize an empty file or return a known current database; reject all other schemas."""
    path = Path(path).resolve()
    if migration_lock_path(path).exists():
        raise AppError('MIGRATION_BUSY', '存储正在离线维护，请稍后重试。', 503)
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
        current = sqlite3.connect(source.as_uri() + '?mode=ro', uri=True)
        try:
            _check_supported_schema(current)
            backup = sqlite3.connect(destination)
            try:
                current.backup(backup)
                _check_supported_schema(backup)
                if backup.execute('PRAGMA integrity_check').fetchone()[0] != 'ok':
                    raise AppError('BACKUP_INVALID', '备份完整性检查失败。', 409)
                if backup.execute('PRAGMA foreign_key_check').fetchone() is not None:
                    raise AppError('BACKUP_INVALID', '备份关联完整性检查失败。', 409)
            finally:
                backup.close()
        finally:
            current.close()
    except BaseException:
        destination.unlink(missing_ok=True)
        raise
    return destination


def _check_supported_schema(connection: sqlite3.Connection) -> int:
    version = connection.execute('PRAGMA user_version').fetchone()[0]
    if version == SCHEMA_VERSION:
        check_schema(connection)
    elif version == 1:
        tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        required = {'schema_migrations', 'datasets', 'experiments', 'jobs', 'job_events',
                    'artifacts', 'idempotency_keys', 'worker_instances'}
        if not required.issubset(tables) or connection.execute(
                'SELECT max(version) FROM schema_migrations').fetchone()[0] != 1:
            raise AppError('SCHEMA_INCOMPATIBLE', '旧版本存储记录不一致。', 503)
    else:
        raise AppError('SCHEMA_INCOMPATIBLE', '不支持此存储版本。', 503)
    return version


def migrate_database(path: str | Path, backup_path: str | Path) -> Path:
    """Offline upgrade: exclude writers, back up committed WAL, then change one transaction.

    Stop API and worker processes before running this command. The sidecar prevents
    new application connections; BEGIN IMMEDIATE excludes writers throughout backup
    and migration. A crashed maintenance lock is intentionally retained for inspection.
    Rollback is a full backup restore with the corresponding application version.
    """
    path = Path(path).resolve()
    lock = migration_lock_path(path)
    if not path.is_file():
        raise AppError('DATABASE_NOT_INITIALIZED', '应用存储尚未初始化。', 503)
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise AppError('MIGRATION_BUSY', '存储维护锁已存在，请先停止写入进程并检查。', 409) from exc
    os.close(descriptor)
    connection = None
    try:
        connection = sqlite3.connect(path.as_uri() + '?mode=rw', uri=True, isolation_level=None, timeout=0)
        connection.execute('PRAGMA foreign_keys=OFF')
        try:
            connection.execute('BEGIN IMMEDIATE')
        except sqlite3.OperationalError as exc:
            raise AppError('MIGRATION_BUSY', '存储仍有写入者，请停止所有写入进程。', 409) from exc
        version = _check_supported_schema(connection)
        now = datetime.now(timezone.utc)
        for (heartbeat,) in connection.execute('SELECT heartbeat_at FROM worker_instances'):
            try:
                seen = datetime.fromisoformat(heartbeat.replace('Z', '+00:00'))
                active = (now - seen).total_seconds() < 30
            except (ValueError, TypeError):
                active = True
            if active:
                raise AppError('MIGRATION_BUSY', '工作进程仍在线，请停止后再执行迁移。', 409)
        backup = backup_database(path, backup_path)
        if version == 1:
            # Foreign keys are disabled only on this exclusive writer. Do not rename
            # old tables: SQLite would rewrite incoming references to temporary names.
            for table in ('experiments', 'jobs'):
                statement = next(sql for sql in DDL if sql.startswith(f'CREATE TABLE {table}('))
                connection.execute(statement.replace(f'CREATE TABLE {table}(', f'CREATE TABLE {table}_v2(', 1))
                connection.execute(f'INSERT INTO {table}_v2 SELECT * FROM {table}')
            connection.execute('DROP TABLE jobs')
            connection.execute('DROP TABLE experiments')
            connection.execute('ALTER TABLE experiments_v2 RENAME TO experiments')
            connection.execute('ALTER TABLE jobs_v2 RENAME TO jobs')
            connection.execute('CREATE INDEX experiments_page ON experiments(created_at DESC,experiment_id DESC)')
            connection.execute('CREATE INDEX jobs_queue ON jobs(status,created_at,job_id)')
            for statement in V2_DDL:
                connection.execute(statement)
            connection.execute('INSERT INTO schema_migrations VALUES (?,?)', (SCHEMA_VERSION, utc_now()))
            connection.execute(f'PRAGMA user_version={SCHEMA_VERSION}')
        check_schema(connection)
        if connection.execute('PRAGMA foreign_key_check').fetchone() is not None:
            raise AppError('MIGRATION_INVALID', '迁移关联完整性检查失败，已回滚。', 409)
        if connection.execute('PRAGMA integrity_check').fetchone()[0] != 'ok':
            raise AppError('MIGRATION_INVALID', '迁移完整性检查失败，已回滚。', 409)
        connection.commit()
        return backup
    except BaseException:
        if connection is not None:
            connection.rollback()
        raise
    finally:
        if connection is not None:
            connection.close()
        lock.unlink(missing_ok=True)
