from contextlib import contextmanager
from pathlib import Path
import sqlite3
from typing import Iterator

from ..errors import AppError

SCHEMA_VERSION = 2
REQUIRED_TABLES = frozenset({'schema_migrations', 'datasets', 'experiments', 'jobs', 'job_events',
                             'artifacts', 'idempotency_keys', 'worker_instances', 'market_datasets_v2',
                             'researches', 'research_units', 'research_attempts', 'exposure_records',
                             'artifact_protocol_registry'})


def migration_lock_path(path: Path) -> Path:
    return path.with_name(path.name + '.migration.lock')


def check_schema(connection: sqlite3.Connection) -> None:
    version = connection.execute('PRAGMA user_version').fetchone()[0]
    tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if version != SCHEMA_VERSION or not REQUIRED_TABLES.issubset(tables):
        raise AppError('SCHEMA_INCOMPATIBLE', '存储版本不兼容，请使用本地维护命令检查。', 503)
    recorded = connection.execute('SELECT max(version) FROM schema_migrations').fetchone()[0]
    if recorded != SCHEMA_VERSION:
        raise AppError('SCHEMA_INCOMPATIBLE', '存储版本记录不一致，请恢复一致性备份。', 503)


class Database:
    def __init__(self, path: str | Path, busy_timeout_ms: int = 5000) -> None:
        self.path = Path(path).resolve()
        self.busy_timeout_ms = busy_timeout_ms

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        if migration_lock_path(self.path).exists():
            raise AppError('MIGRATION_BUSY', '存储正在离线维护，请稍后重试。', 503)
        if self.path.resolve() != self.path:
            raise AppError('DATABASE_PATH_CHANGED', '应用数据库的实际位置已变更，请检查配置。', 503)
        if not self.path.is_file():
            raise AppError('DATABASE_NOT_INITIALIZED', '应用存储尚未初始化。', 503)
        connection = None
        try:
            connection = sqlite3.connect(self.path.as_uri() + '?mode=rw', uri=True,
                                         timeout=self.busy_timeout_ms / 1000, isolation_level=None)
            connection.row_factory = sqlite3.Row
            connection.execute('PRAGMA foreign_keys=ON')
            connection.execute(f'PRAGMA busy_timeout={int(self.busy_timeout_ms)}')
            check_schema(connection)
            if connection.execute('PRAGMA journal_mode').fetchone()[0].lower() != 'wal':
                raise AppError('SCHEMA_INCOMPATIBLE', '应用存储未配置 WAL，请使用本地维护命令检查。', 503)
            yield connection
        except sqlite3.DatabaseError as exc:
            if isinstance(exc, sqlite3.IntegrityError):
                raise
            raise AppError('DATABASE_UNAVAILABLE', '应用存储暂时不可用，请稍后重试。', 503) from exc
        finally:
            if connection is not None:
                connection.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        with self.connection() as connection:
            connection.execute('BEGIN IMMEDIATE')
            try:
                check_schema(connection)
                yield connection
                connection.commit()
            except BaseException:
                connection.rollback()
                raise


def initialize_database(path: str | Path) -> Database:
    """Compatibility export for the explicit launcher; initialization remains opt-in."""
    from .migrations import initialize_database as initialize
    return initialize(path)
