"""Locked research submission and explicit whole-unit resume, never HTTP training."""
from __future__ import annotations

import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import subprocess
import sys
from uuid import uuid4

from stockrl.research.contracts import ResearchProtocolDraft, ResearchProtocol, UnitResult, UnitKey
from stockrl.research.folds import preview_research

from .errors import AppError
from .experiments import idempotency_key
from .market_datasets import MarketDatasetService
from .models import ExperimentRecord, JobRecord, SubmissionResult, Versions, utc_now
from .research_artifacts import verify_research_bundle
from .settings import confined_path
from .storage.common import canonical_json
from .storage.experiments import ExperimentRepository
from .storage.jobs import JobRepository, TERMINAL_STATUSES
from .storage.researches import ResearchRepository


def environment_fingerprint(settings) -> dict:
    """Digest executable source bytes, including untracked source, without loading learners."""
    hashes = {}
    for directory in ('stockrl', 'stockrl_app', 'api', 'scripts'):
        for path in sorted((settings.project_root / directory).rglob('*.py')):
            hashes[path.relative_to(settings.project_root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    for name in ('pyproject.toml', 'requirements.txt'):
        path = settings.project_root / name
        if path.is_file():
            hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    dependencies = {}
    for name in ('numpy', 'pandas', 'torch', 'gymnasium', 'stable-baselines3'):
        try:
            dependencies[name] = version(name)
        except PackageNotFoundError:
            dependencies[name] = 'missing'
    try:
        commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=settings.project_root,
                                capture_output=True, text=True, timeout=5, check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        commit = None
    result = {'code_commit': commit, 'python_version': sys.version, 'python_implementation': sys.implementation.name,
              'dirty_tree_sha256': hashlib.sha256(canonical_json(hashes).encode()).hexdigest(),
              'dependencies': dependencies}
    result['sha256'] = hashlib.sha256(canonical_json(result).encode()).hexdigest()
    return result


def protocol_units(protocol: ResearchProtocol):
    for instrument in protocol.instrument_ids:
        for fold in protocol.fold_plan:
            if fold.instrument_id in (None, instrument):
                for seed in protocol.seeds:
                    yield UnitKey(instrument_id=instrument, fold_id=fold.fold_id, seed=seed)


def unit_path(key: UnitKey) -> str:
    instrument = hashlib.sha256(key.instrument_id.encode()).hexdigest()[:24]
    fold = hashlib.sha256(key.fold_id.encode()).hexdigest()[:16]
    return f'units/{instrument}/{fold}/seed_{key.seed}'


class ResearchService:
    def __init__(self, settings, db):
        self.settings, self.db = settings, db
        self.datasets = MarketDatasetService(settings, db)
        self.repository = ResearchRepository(db)
        self.jobs = JobRepository(db, settings.worker_unavailable_seconds)

    def _prepare(self, draft: ResearchProtocolDraft):
        import pandas as pd
        from stockrl.market.validation import canonical_instrument_id
        calendars, bundles = {}, {}
        blockers = []
        profiles = {}
        for dataset_id in draft.dataset_ids:
            bundle = self.datasets.load(dataset_id)
            instrument = canonical_instrument_id(bundle.metadata)
            if instrument in bundles or instrument not in draft.instrument_ids:
                raise AppError('DATA_INVALID', '每个研究标的必须恰好对应一个规范身份的数据包。', 422)
            if bundle.fingerprint != draft.dataset_fingerprints[dataset_id]:
                raise AppError('DATASET_CHANGED', '数据指纹与预注册请求不符。', 422)
            profile_id = bundle.market_profile.profile_id
            profile_hash = bundle.file_hashes['market-profile.json']
            if profile_id in profiles and profiles[profile_id] != profile_hash:
                raise AppError('DATA_INVALID', '同名市场规则对应不同内容。', 422)
            profiles[profile_id] = profile_hash
            for costs in (bundle.market_profile, *bundle.market_profile.fee_intervals):
                if costs.commission_rate * 3 >= 1 or costs.slippage * 3 >= 1:
                    blockers.append('COST_SCENARIO_INVALID')
            bundles[instrument] = bundle
            calendars[instrument] = pd.DataFrame([{'session': s.session, 'is_open': s.is_open} for s in bundle.sessions])
        if set(bundles) != set(draft.instrument_ids) or profiles != draft.market_profile_fingerprints:
            raise AppError('DATA_INVALID', '标的或市场规则集合与已登记数据不一致。', 422)
        preview = preview_research(draft, calendars, blockers=tuple(sorted(set(blockers))))
        exposure_warnings = []
        for exposure in self.repository.exposures():
            for fold in preview.fold_plan:
                if (exposure['instrument_id'] == fold.instrument_id and exposure['scope'] == 'exposed_test'
                        and exposure['start_session'] <= fold.test.reward_sessions[-1]
                        and exposure['end_session'] >= fold.test.reward_sessions[0]):
                    exposure_warnings.append('EXPOSED_TEST_OVERLAP')
        qualification = 'exploratory' if exposure_warnings else draft.qualification
        if any(bundle.metadata.get('is_synthetic', False) or 'synthetic' in str(bundle.metadata.get('source', '')).lower()
               for bundle in bundles.values()):
            qualification = 'exploratory'
            exposure_warnings.append('SYNTHETIC_DATA')
        canonical_draft = draft.model_copy(update={'qualification': qualification})
        protocol = ResearchProtocol(**canonical_draft.model_dump(), fold_plan=preview.fold_plan,
                                    checkpoint_steps=preview.checkpoint_steps)
        result = {**preview.model_dump(mode='json'), 'qualification': qualification,
                  'exposure_warnings': sorted(set(exposure_warnings)), 'protocol_sha256': protocol.lock().sha256,
                  'canonical_request': canonical_draft.model_dump(mode='json'),
                  'maximum_runtime_seconds': preview.budget.unit_count * self.settings.task_time_limit_seconds}
        return result, protocol

    def preview(self, draft: ResearchProtocolDraft) -> dict:
        return self._prepare(draft)[0]

    def submit(self, draft: ResearchProtocolDraft, key: str) -> dict:
        key = idempotency_key(key)
        digest = hashlib.sha256(canonical_json(draft.model_dump(mode='json')).encode()).hexdigest()
        experiments = ExperimentRepository(self.db)
        with self.db.connection() as connection:
            existing = experiments.idempotent_result(connection, key, digest)
            if existing:
                record = self.repository.get(existing.experiment_id, connection)
                return {'research_id': existing.experiment_id, 'job_id': existing.job_id, 'protocol_sha256': record['protocol_sha256']}
        preview, protocol = self._prepare(draft)
        if preview['blockers']:
            raise AppError('RESEARCH_PLAN_INVALID', '研究计划存在阻断项。', 422, {'blockers': preview['blockers']})
        research_id, job_id, attempt_id = (str(uuid4()) for _ in range(3))
        now = utc_now()
        environment = environment_fingerprint(self.settings)
        record = {'research_id': research_id, 'job_id': job_id, 'latest_attempt_id': attempt_id, 'created_at': now,
                  'technical_status': 'queued', 'protocol': protocol.model_dump(mode='json'),
                  'protocol_sha256': protocol.lock().sha256, 'environment': environment, 'preview': preview}
        parent = ExperimentRecord(experiment_id=research_id, job_id=job_id, kind='research', created_at=now,
                                  request_sha256=record['protocol_sha256'], versions=Versions(artifact_schema_version=2,
                                  core_semantics_version=2, metrics_version=2), replay_block_reason='研究单元请在研究页面查看。')
        job = JobRecord(job_id=job_id, experiment_id=research_id, kind='research', created_at=now,
                        seed_count=preview['budget']['unit_count'], requested_steps_total=preview['budget']['requested_total_steps'])
        with self.db.transaction() as connection:
            existing = experiments.idempotent_result(connection, key, digest)
            if existing:
                current = self.repository.get(existing.experiment_id, connection)
                return {'research_id': existing.experiment_id, 'job_id': existing.job_id, 'protocol_sha256': current['protocol_sha256']}
            if self.jobs.queued_count(connection) >= self.settings.queue_capacity:
                raise AppError('QUEUE_CAPACITY', '等待任务已达到上限。', 429)
            prior = connection.execute("SELECT protocol_fingerprint FROM researches WHERE json_extract(payload_json,'$.protocol.protocol_id')=?",
                                       (protocol.protocol_id,)).fetchall()
            if any(row[0] != record['protocol_sha256'] for row in prior):
                raise AppError('PROTOCOL_ID_CONFLICT', '该协议编号已用于不同计划，请使用新的协议编号。', 409)
            experiments.insert(parent, connection)
            self.jobs.insert(job, connection)
            connection.execute('''INSERT INTO researches(research_id,created_at,status,protocol_fingerprint,
                execution_environment_fingerprint,latest_attempt_id,payload_json) VALUES(?,?,?,?,?,?,?)''',
                (research_id, now, 'queued', record['protocol_sha256'], environment['sha256'], attempt_id, canonical_json(record)))
            connection.execute('''INSERT INTO research_attempts(attempt_id,research_id,job_id,status,created_at,payload_json)
                VALUES(?,?,?,?,?,?)''', (attempt_id, research_id, job_id, 'queued', now, '{}'))
            for unit in protocol_units(protocol):
                execution_key = hashlib.sha256(canonical_json([research_id, unit.model_dump()]).encode()).hexdigest()
                connection.execute('''INSERT INTO research_units(execution_key,research_id,instrument_id,fold_id,seed,status,payload_json)
                    VALUES(?,?,?,?,?,'queued','{}')''', (execution_key, research_id, unit.instrument_id, unit.fold_id, unit.seed))
            experiments.register_key(connection, key, digest, SubmissionResult(experiment_id=research_id, job_id=job_id), now)
        return {'research_id': research_id, 'job_id': job_id, 'protocol_sha256': record['protocol_sha256']}

    def get(self, research_id: str) -> dict:
        from stockrl.research.report import summarize_research
        record = self.repository.get(research_id)
        self.repository.sync_terminal(record['job_id'])
        record = self.repository.get(research_id)
        units = self.repository.units(research_id)
        protocol = ResearchProtocol.model_validate(record['protocol'])
        order = {(key.instrument_id, key.fold_id, key.seed): index for index, key in enumerate(protocol_units(protocol))}
        units.sort(key=lambda row: order[(row['instrument_id'], row['fold_id'], row['seed'])])
        results = []
        for row in units:
            if row['status'] == 'succeeded':
                root = confined_path(self.settings.research_dir, row['artifact_root'], must_exist=True)
                verify_research_bundle(root, expected_hash=row['manifest_sha256'])
                if json.loads((root / 'unit_result.json').read_text(encoding='utf-8')) != row['result']:
                    raise AppError('ARTIFACT_CORRUPT', '研究单元索引与已封存结果不一致。', 409)
                results.append(UnitResult.model_validate(row['result']))
            else:
                status = {'queued': 'pending', 'interrupted': 'failed'}.get(row['status'], row['status'])
                results.append(UnitResult(key=UnitKey(instrument_id=row['instrument_id'], fold_id=row['fold_id'], seed=row['seed']),
                                          status=status, error=row['error_json']))
        if protocol.lock().sha256 != record['protocol_sha256']:
            raise AppError('ARTIFACT_CORRUPT', '研究协议指纹不一致。', 409)
        report = summarize_research(protocol, results)
        job_detail = self.jobs.detail(record['job_id'])
        public_report = report.model_dump(mode='json')
        if job_detail.status == 'succeeded':
            parent_root = confined_path(self.settings.research_dir, research_id, must_exist=True)
            if not record.get('parent_manifest_sha256'):
                raise AppError('ARTIFACT_CORRUPT', '研究汇总缺少已登记的清单指纹。', 409)
            parent_manifest = verify_research_bundle(parent_root, expected_hash=record['parent_manifest_sha256'])
            if (parent_manifest.get('protocol_sha256') != record['protocol_sha256']
                    or json.loads((parent_root / 'report.json').read_text(encoding='utf-8')) != public_report
                    or json.loads((parent_root / 'protocol.json').read_text(encoding='utf-8')) != record['protocol']):
                raise AppError('ARTIFACT_CORRUPT', '研究汇总与已封存协议或单元不一致。', 409)
        elif report.technical_status == 'completed':
            public_report.update(technical_status='incomplete', economic_outcome='insufficient_evidence',
                                 provisional_outcome=None, reasons=[*public_report['reasons'], 'PARENT_NOT_PUBLISHED'])
        if job_detail.status in ('cancelled', 'failed', 'interrupted'):
            public_report['technical_status'] = 'incomplete' if job_detail.status == 'interrupted' else job_detail.status
        return {**public_report, 'research_id': research_id, 'job_id': record['job_id'],
                'job_status': job_detail.model_dump(mode='json'),
                'created_at': record['created_at'], 'protocol': record['protocol'], 'protocol_sha256': record['protocol_sha256'],
                'preview': record['preview'], 'attempts': self.repository.attempts(research_id),
                'unit_progress': [{k: row[k] for k in ('instrument_id', 'fold_id', 'seed', 'status', 'error_json')} for row in units]}

    def list(self, *, limit: int = 20, cursor: str | None = None) -> dict:
        if not 1 <= limit <= 100:
            raise AppError('INVALID_PAGINATION', '每页条数必须介于 1 和 100。', 422)
        with self.db.connection() as connection:
            after = None
            if cursor:
                after = connection.execute('SELECT created_at,research_id FROM researches WHERE research_id=?', (cursor,)).fetchone()
                if after is None:
                    raise AppError('INVALID_CURSOR', '研究分页游标无效。', 422)
            where = ' WHERE (created_at,research_id)<(?,?)' if after else ''
            params = (after['created_at'], after['research_id'], limit + 1) if after else (limit + 1,)
            rows = connection.execute('SELECT payload_json FROM researches' + where + ' ORDER BY created_at DESC,research_id DESC LIMIT ?', params).fetchall()
        records = [json.loads(row[0]) for row in rows[:limit]]
        for record in records:
            job = self.jobs.get(record['job_id'])
            record['technical_status'] = 'completed' if job.status == 'succeeded' else job.status
        return {'items': [{k: record[k] for k in ('research_id', 'job_id', 'created_at', 'technical_status', 'protocol_sha256')}
                          | {'qualification': record['protocol']['qualification'], 'hypothesis': record['protocol']['hypothesis']}
                          for record in records], 'has_more': len(rows) > limit,
                'next_cursor': records[-1]['research_id'] if len(rows) > limit else None}

    def diagnostics(self, research_id: str, instrument_id: str, fold_id: str, seed: int) -> dict:
        for row in self.repository.units(research_id):
            if (row['instrument_id'], row['fold_id'], row['seed']) == (instrument_id, fold_id, seed):
                if row['status'] != 'succeeded':
                    break
                root = confined_path(self.settings.research_dir, row['artifact_root'], must_exist=True)
                verify_research_bundle(root, expected_hash=row['manifest_sha256'])
                return json.loads(confined_path(root, 'diagnostics.json', must_exist=True).read_text(encoding='utf-8'))
        raise AppError('DIAGNOSTICS_UNAVAILABLE', '该研究单元尚无完整诊断。', 409)

    def resume(self, research_id: str, key: str) -> dict:
        key = idempotency_key(key)
        digest = hashlib.sha256(f'resume:{research_id}'.encode()).hexdigest()
        experiments = ExperimentRepository(self.db)
        with self.db.connection() as connection:
            existing = experiments.idempotent_result(connection, key, digest)
            if existing:
                return {'research_id': research_id, 'job_id': existing.job_id}
        record = self.repository.get(research_id)
        self.repository.sync_terminal(record['job_id'])
        protocol = ResearchProtocol.model_validate(record['protocol'])
        if protocol.lock().sha256 != record['protocol_sha256'] or environment_fingerprint(self.settings) != record['environment']:
            raise AppError('RESEARCH_RESUME_INCOMPATIBLE', '代码、依赖或协议已变化；请创建新研究。', 409)
        # Check immutable input bytes, but never re-qualify this locked study against later exposures.
        for dataset_id in protocol.dataset_ids:
            if self.datasets.load(dataset_id).fingerprint != protocol.dataset_fingerprints[dataset_id]:
                raise AppError('RESEARCH_RESUME_INCOMPATIBLE', '数据指纹已变化。', 409)
        for row in self.repository.units(research_id):
            if row['status'] == 'succeeded':
                root = confined_path(self.settings.research_dir, row['artifact_root'], must_exist=True)
                verify_research_bundle(root, expected_hash=row['manifest_sha256'])
        job_id, attempt_id, now = str(uuid4()), str(uuid4()), utc_now()
        with self.db.transaction() as connection:
            existing = experiments.idempotent_result(connection, key, digest)
            if existing:
                return {'research_id': research_id, 'job_id': existing.job_id}
            record = self.repository.get(research_id, connection)
            old_job = self.jobs.get(record['job_id'], connection)
            if old_job.status not in TERMINAL_STATUSES or old_job.status == 'succeeded':
                raise AppError('STATE_CONFLICT', '研究仍在执行或已完整完成。', 409)
            if self.jobs.queued_count(connection) >= self.settings.queue_capacity:
                raise AppError('QUEUE_CAPACITY', '等待任务已达到上限。', 429)
            connection.execute('UPDATE research_attempts SET status=? WHERE job_id=?', (old_job.status, old_job.job_id))
            job = JobRecord(job_id=job_id, experiment_id=research_id, kind='research', created_at=now,
                            seed_count=old_job.seed_count, requested_steps_total=old_job.requested_steps_total)
            self.jobs.insert(job, connection)
            connection.execute('INSERT INTO research_attempts(attempt_id,research_id,job_id,status,created_at,payload_json) VALUES(?,?,?,?,?,?)',
                               (attempt_id, research_id, job_id, 'queued', now, '{}'))
            record.update(job_id=job_id, latest_attempt_id=attempt_id, technical_status='queued')
            self.repository.save(record, connection)
            parent = experiments.get(research_id, connection)
            parent.job_id = job_id
            connection.execute('UPDATE experiments SET job_id=?,payload_json=? WHERE experiment_id=?',
                               (job_id, parent.model_dump_json(), research_id))
            experiments.register_key(connection, key, digest, SubmissionResult(experiment_id=research_id, job_id=job_id), now)
        return {'research_id': research_id, 'job_id': job_id}
