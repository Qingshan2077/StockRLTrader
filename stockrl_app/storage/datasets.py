import sqlite3

from ..errors import AppError
from ..models import DatasetRecord, Page
from .common import page_rows
from .database import Database


class DatasetRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    def insert(self, record: DatasetRecord, connection: sqlite3.Connection | None = None) -> None:
        if connection is None:
            with self.db.transaction() as transaction:
                self.insert(record, transaction)
            return
        connection.execute('INSERT INTO datasets(dataset_id,created_at,snapshot_sha256,payload_json) VALUES (?,?,?,?)',
                           (record.dataset_id, record.created_at, record.snapshot_sha256, record.model_dump_json()))

    def get(self, dataset_id: str) -> DatasetRecord:
        with self.db.connection() as connection:
            row = connection.execute('SELECT payload_json FROM datasets WHERE dataset_id=?', (dataset_id,)).fetchone()
        if row is None:
            raise AppError('DATASET_NOT_FOUND', '数据集不存在。', 404)
        return DatasetRecord.model_validate_json(row['payload_json'])

    def list(self, *, limit: int = 20, cursor: str | None = None) -> Page[DatasetRecord]:
        with self.db.connection() as connection:
            return page_rows(connection, 'datasets', 'dataset_id', DatasetRecord, limit=limit, cursor=cursor)
