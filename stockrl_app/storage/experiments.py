import sqlite3

from ..errors import AppError
from ..models import ExperimentRecord, Page, SubmissionResult
from .common import page_rows
from .database import Database


class ExperimentRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    def get(self, experiment_id: str, connection: sqlite3.Connection | None = None) -> ExperimentRecord:
        if connection is None:
            with self.db.connection() as reader:
                return self.get(experiment_id, reader)
        row = connection.execute('SELECT payload_json FROM experiments WHERE experiment_id=?', (experiment_id,)).fetchone()
        if row is None:
            raise AppError('EXPERIMENT_NOT_FOUND', '实验不存在。', 404)
        return ExperimentRecord.model_validate_json(row['payload_json'])

    def insert(self, record: ExperimentRecord, connection: sqlite3.Connection) -> None:
        connection.execute('''INSERT INTO experiments(experiment_id,created_at,dataset_id,kind,source_experiment_id,
            job_id,integrity,output_relative_path,legacy_fingerprint,payload_json) VALUES (?,?,?,?,?,?,?,?,?,?)''',
            (record.experiment_id, record.created_at, record.dataset_id, record.kind, record.source_experiment_id,
             record.job_id, record.integrity, record.output_relative_path, record.legacy_fingerprint, record.model_dump_json()))

    def update(self, record: ExperimentRecord, connection: sqlite3.Connection) -> None:
        """Internal result indexing only: immutable request/dataset/source fields cannot change."""
        old = self.get(record.experiment_id, connection)
        for field in ('request', 'request_sha256', 'dataset_id', 'dataset', 'kind', 'source_experiment_id',
                      'job_id', 'created_at', 'filtered_data_sha256', 'bars_relative_path', 'splits',
                      'legacy_config', 'data_label', 'is_synthetic'):
            if getattr(old, field) != getattr(record, field):
                raise AppError('IMMUTABLE_EXPERIMENT', '已提交实验的配置与数据不可修改。', 409)
        connection.execute('''UPDATE experiments SET integrity=?, output_relative_path=?,
            legacy_fingerprint=?,payload_json=? WHERE experiment_id=?''',
            (record.integrity, record.output_relative_path, record.legacy_fingerprint,
             record.model_dump_json(), record.experiment_id))

    def list(self, *, limit: int = 20, cursor: str | None = None, kind: str | None = None,
             integrity: str | None = None, algorithm: str | None = None,
             dataset_id: str | None = None, status: str | None = None) -> Page[ExperimentRecord]:
        filters = {name: value for name, value in {'kind': kind, 'integrity': integrity, 'algorithm': algorithm,
                   'dataset_id': dataset_id, 'status': status}.items() if value is not None}
        with self.db.connection() as connection:
            return page_rows(connection, 'experiments', 'experiment_id', ExperimentRecord,
                             limit=limit, cursor=cursor, filters=filters, filter_columns={
                                 'algorithm': "coalesce(json_extract(payload_json,'$.request.algorithm'),json_extract(payload_json,'$.legacy_config.algorithm'))",
                                 'status': '(SELECT status FROM jobs WHERE jobs.job_id=experiments.job_id)'})

    @staticmethod
    def idempotent_result(connection: sqlite3.Connection, key: str, request_sha256: str) -> SubmissionResult | None:
        row = connection.execute('SELECT * FROM idempotency_keys WHERE key=?', (key,)).fetchone()
        if row is None:
            return None
        if row['request_sha256'] != request_sha256:
            raise AppError('IDEMPOTENCY_CONFLICT', '此提交标识已用于不同配置，请创建新实验。', 409)
        return SubmissionResult(experiment_id=row['experiment_id'], job_id=row['job_id'])

    @staticmethod
    def register_key(connection: sqlite3.Connection, key: str, request_sha256: str,
                     result: SubmissionResult, created_at: str) -> None:
        connection.execute('INSERT INTO idempotency_keys VALUES (?,?,?,?,?)',
                           (key, request_sha256, result.experiment_id, result.job_id, created_at))
