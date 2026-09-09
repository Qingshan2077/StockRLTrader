"""Research state and unit publication fenced by the existing job owner."""
from __future__ import annotations

import json

from ..errors import AppError
from ..models import utc_now
from .common import canonical_json
from .database import Database
from .jobs import JobRepository


class ResearchRepository:
    def __init__(self, db: Database):
        self.db = db

    def get(self, research_id: str, connection=None) -> dict:
        if connection is None:
            with self.db.connection() as reader:
                return self.get(research_id, reader)
        row = connection.execute('SELECT payload_json FROM researches WHERE research_id=?', (research_id,)).fetchone()
        if row is None:
            raise AppError('RESEARCH_NOT_FOUND', '研究不存在。', 404)
        return json.loads(row['payload_json'])

    def save(self, record: dict, connection) -> None:
        connection.execute('UPDATE researches SET status=?,latest_attempt_id=?,payload_json=? WHERE research_id=?',
                           (record['technical_status'], record['latest_attempt_id'], canonical_json(record), record['research_id']))

    def units(self, research_id: str, connection=None) -> list[dict]:
        if connection is None:
            with self.db.connection() as reader:
                return self.units(research_id, reader)
        rows = connection.execute('SELECT * FROM research_units WHERE research_id=? ORDER BY instrument_id,fold_id,seed',
                                  (research_id,)).fetchall()
        return [{**dict(row), 'result': json.loads(row['payload_json'])} for row in rows]

    def attempts(self, research_id: str, connection=None) -> list[dict]:
        if connection is None:
            with self.db.connection() as reader:
                return self.attempts(research_id, reader)
        return [dict(row) for row in connection.execute(
            'SELECT attempt_id,job_id,status,created_at FROM research_attempts WHERE research_id=? ORDER BY created_at',
            (research_id,)).fetchall()]

    def exposures(self, connection=None) -> list[dict]:
        if connection is None:
            with self.db.connection() as reader:
                return self.exposures(reader)
        return [json.loads(row[0]) for row in connection.execute('SELECT payload_json FROM exposure_records')]

    @staticmethod
    def assert_owner(connection, research: dict, job_id: str, owner_token: str):
        row = connection.execute('SELECT payload_json FROM jobs WHERE job_id=?', (job_id,)).fetchone()
        if row is None:
            raise AppError('STATE_CONFLICT', '研究执行权已失效。', 409)
        job = json.loads(row[0])
        if (research['job_id'] != job_id or job['owner_token'] != owner_token or not owner_token
                or job['status'] != 'running' or job['cancel_requested']):
            raise AppError('STATE_CONFLICT', '研究已停止、取消或执行权已变更。', 409)
        return job

    def claim_unit(self, research_id: str, key: dict, *, job_id: str, owner_token: str) -> dict:
        with self.db.transaction() as connection:
            research = self.get(research_id, connection)
            self.assert_owner(connection, research, job_id, owner_token)
            row = connection.execute('SELECT * FROM research_units WHERE research_id=? AND instrument_id=? AND fold_id=? AND seed=?',
                                     (research_id, key['instrument_id'], key['fold_id'], key['seed'])).fetchone()
            if row is None or row['status'] == 'succeeded':
                raise AppError('STATE_CONFLICT', '研究单元不存在或已完整发布。', 409)
            now = utc_now()
            revision = row['revision'] + 1
            connection.execute('''UPDATE research_units SET status='running',attempt_id=?,revision=?,owner_token=?,
                started_at=?,finished_at=NULL,error_json=NULL WHERE execution_key=? AND revision=?''',
                (research['latest_attempt_id'], revision, owner_token, now, row['execution_key'], row['revision']))
            research['technical_status'] = 'running'
            self.save(research, connection)
            connection.execute("UPDATE research_attempts SET status='running' WHERE attempt_id=?", (research['latest_attempt_id'],))
            return {**dict(row), 'revision': revision, 'owner_token': owner_token,
                    'attempt_id': research['latest_attempt_id'], 'status': 'running'}

    def complete_unit(self, claimed: dict, result: dict, *, job_id: str, artifact_root: str,
                      manifest_sha256: str) -> None:
        with self.db.transaction() as connection:
            research = self.get(claimed['research_id'], connection)
            self.assert_owner(connection, research, job_id, claimed['owner_token'])
            updated = connection.execute('''UPDATE research_units SET status='succeeded',artifact_root=?,manifest_sha256=?,
                finished_at=?,payload_json=?,revision=revision+1 WHERE execution_key=? AND revision=? AND owner_token=?
                AND attempt_id=? AND status='running' ''',
                (artifact_root, manifest_sha256, utc_now(), canonical_json(result), claimed['execution_key'],
                 claimed['revision'], claimed['owner_token'], research['latest_attempt_id']))
            if updated.rowcount != 1:
                raise AppError('STATE_CONFLICT', '研究单元发布权已失效。', 409)
            connection.execute('''INSERT INTO artifact_protocol_registry
                (artifact_id,research_id,execution_key,protocol_fingerprint,manifest_sha256,payload_json) VALUES(?,?,?,?,?,?)''',
                (claimed['execution_key'], claimed['research_id'], claimed['execution_key'], research['protocol_sha256'],
                 manifest_sha256, canonical_json({'artifact_root': artifact_root})))

    def fail_unit(self, claimed: dict, *, job_id: str, error: dict) -> None:
        with self.db.transaction() as connection:
            research = self.get(claimed['research_id'], connection)
            self.assert_owner(connection, research, job_id, claimed['owner_token'])
            connection.execute('''UPDATE research_units SET status='failed',error_json=?,finished_at=?,revision=revision+1
                WHERE execution_key=? AND revision=? AND owner_token=? AND attempt_id=? AND status='running' ''',
                (canonical_json(error), utc_now(), claimed['execution_key'], claimed['revision'], claimed['owner_token'],
                 research['latest_attempt_id']))

    def sync_terminal(self, job_id: str) -> None:
        """Reflect existing coordinator decisions; never revive an interrupted optimizer."""
        with self.db.transaction() as connection:
            job = JobRepository(self.db).get(job_id, connection)
            if job.kind != 'research' or job.status not in ('failed', 'cancelled', 'interrupted'):
                return
            research = self.get(job.experiment_id, connection)
            connection.execute('UPDATE research_attempts SET status=? WHERE job_id=?', (job.status, job_id))
            if research['job_id'] != job_id:
                return
            research['technical_status'] = job.status
            self.save(research, connection)
            connection.execute('''UPDATE research_units SET status=?,finished_at=?,revision=revision+1
                WHERE research_id=? AND status='running' ''', (job.status, utc_now(), job.experiment_id))
