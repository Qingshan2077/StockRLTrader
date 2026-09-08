import sqlite3

from ..errors import AppError
from ..models import ArtifactStorageRecord
from .database import Database


class ArtifactRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    def insert(self, record: ArtifactStorageRecord, connection: sqlite3.Connection) -> None:
        connection.execute('''INSERT INTO artifacts(artifact_id,experiment_id,root_id,relative_path,kind,size,sha256,payload_json)
            VALUES (?,?,?,?,?,?,?,?)''', (record.artifact_id, record.experiment_id, record.root_id,
            record.relative_path, record.kind, record.size, record.sha256, record.model_dump_json()))

    def get(self, artifact_id: str) -> ArtifactStorageRecord:
        with self.db.connection() as connection:
            row = connection.execute('SELECT payload_json FROM artifacts WHERE artifact_id=?', (artifact_id,)).fetchone()
        if row is None:
            raise AppError('ARTIFACT_NOT_FOUND', '产物不存在。', 404)
        return ArtifactStorageRecord.model_validate_json(row['payload_json'])

    def list(self, experiment_id: str) -> list[ArtifactStorageRecord]:
        with self.db.connection() as connection:
            rows = connection.execute('SELECT payload_json FROM artifacts WHERE experiment_id=? ORDER BY relative_path', (experiment_id,)).fetchall()
        return [ArtifactStorageRecord.model_validate_json(row['payload_json']) for row in rows]
