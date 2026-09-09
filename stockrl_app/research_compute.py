"""Sequential research units within the existing computing child."""
import json
from time import monotonic
from uuid import uuid4

from stockrl.control import ExecutionControl, ExperimentCancelled
from stockrl.research.contracts import ResearchProtocol, UnitResult

from .errors import AppError
from .artifact_protocol import hash_file
from .market_datasets import MarketDatasetService
from .models import StateEventPayload, utc_now
from .research_artifacts import seal_research_bundle, verify_research_bundle, write_json_atomic
from .researches import environment_fingerprint, protocol_units, unit_path
from .settings import confined_path
from .storage.common import canonical_json
from .storage.experiments import ExperimentRepository
from .storage.jobs import JobRepository
from .storage.researches import ResearchRepository


def execute_research(settings, db, job, control):
    from stockrl.market.validation import canonical_instrument_id
    from stockrl.research.runner import run_unit
    from stockrl.research.report import summarize_research
    repository = ResearchRepository(db)
    record = repository.get(job.experiment_id)
    protocol = ResearchProtocol.model_validate(record['protocol'])
    if protocol.lock().sha256 != record['protocol_sha256'] or environment_fingerprint(settings) != record['environment']:
        raise AppError('RESEARCH_RESUME_INCOMPATIBLE', '研究提交后代码、环境或协议发生变化。', 409)
    root = confined_path(settings.research_dir, job.experiment_id)
    root.mkdir(parents=True, exist_ok=True)
    protocol_path = root / 'protocol.json'
    if protocol_path.exists():
        if json.loads(protocol_path.read_text(encoding='utf-8')) != record['protocol']:
            raise AppError('ARTIFACT_CORRUPT', '已冻结协议文件发生变化。', 409)
    else:
        write_json_atomic(protocol_path, record['protocol'])
    datasets = MarketDatasetService(settings, db)
    bundles = {}
    for identifier in protocol.dataset_ids:
        bundle = datasets.load(identifier)
        if bundle.fingerprint != protocol.dataset_fingerprints[identifier]:
            raise AppError('DATASET_CHANGED', '研究数据指纹已变化。', 409)
        bundles[canonical_instrument_id(bundle.metadata)] = bundle
    results = []
    for index, key in enumerate(protocol_units(protocol)):
        control.check()
        row = next(row for row in repository.units(job.experiment_id) if
                   (row['instrument_id'], row['fold_id'], row['seed']) == (key.instrument_id, key.fold_id, key.seed))
        relative = f'{job.experiment_id}/{unit_path(key)}'
        target = confined_path(settings.research_dir, relative)
        if row['status'] == 'succeeded':
            verify_research_bundle(target, expected_hash=row['manifest_sha256'], check=control.check)
            results.append(UnitResult.model_validate(row['result']))
            continue
        claimed = repository.claim_unit(job.experiment_id, key.model_dump(), job_id=job.job_id, owner_token=job.owner_token)
        fold = next(f for f in protocol.fold_plan if f.fold_id == key.fold_id and f.instrument_id in (None, key.instrument_id))
        # Conservatively record the interval when the unit starts, including a later interrupted test.
        with db.transaction() as connection:
            for scope, window in (('evaluation', fold.train), ('evaluation', fold.validation), ('exposed_test', fold.test)):
                exposure = dict(instrument_id=key.instrument_id, start_session=window.reward_sessions[0],
                                end_session=window.reward_sessions[-1], scope=scope, reason='研究执行单元已开始；保守登记计划区间',
                                recorded_at=utc_now(), protocol_id=protocol.protocol_id, source=job.experiment_id)
                connection.execute('INSERT INTO exposure_records VALUES(?,?,?,?,?,?,?)',
                    (str(uuid4()), job.experiment_id, key.instrument_id, exposure['start_session'], exposure['end_session'],
                     exposure['recorded_at'], canonical_json(exposure)))
        started = monotonic()
        class UnitControl(ExecutionControl):
            def check(self):
                control.check()
                if monotonic() - started >= settings.task_time_limit_seconds:
                    raise AppError('UNIT_TIME_LIMIT_EXCEEDED', '研究单元超过执行时限。', 409)
        unit_control = UnitControl(event_callback=control.emit)
        try:
            if target.exists():
                # A crash after atomic rename but before indexing is recoverable without retraining.
                manifest = verify_research_bundle(target, check=unit_control.check)
                if (manifest.get('protocol_sha256') != record['protocol_sha256'] or manifest.get('key') != key.model_dump()
                        or manifest.get('environment') != record['environment'] or manifest.get('research_id') != job.experiment_id):
                    raise AppError('ARTIFACT_CORRUPT', '未索引单元的身份或环境不符。', 409)
                result = UnitResult.model_validate_json((target / 'unit_result.json').read_text(encoding='utf-8'))
                from .artifact_protocol import hash_file
                digest = hash_file(target / 'manifest.json')
            else:
                stage = confined_path(settings.research_dir, f'.staging/{job.job_id}/{uuid4()}')
                stage.mkdir(parents=True, exist_ok=False)
                result = run_unit(protocol, key, stage, unit_control, bundle=bundles[key.instrument_id])
                if result.key != key or result.status != 'completed':
                    raise AppError('RESEARCH_UNIT_FAILED', '研究单元没有完整计算结果。', 409)
                write_json_atomic(stage / 'unit_result.json', result.model_dump(mode='json'))
                digest = seal_research_bundle(stage, {'research_id': job.experiment_id, 'key': key.model_dump(),
                    'protocol_sha256': record['protocol_sha256'], 'environment': record['environment'],
                    'attempt_id': claimed['attempt_id'], 'owner_token': job.owner_token}, check=unit_control.check)
                verify_research_bundle(stage, expected_hash=digest, check=unit_control.check)
                unit_control.check()
                target.parent.mkdir(parents=True, exist_ok=True)
                # Same-volume rename; a later cancelled publication remains unindexed for explicit resume.
                stage.rename(target)
            repository.complete_unit(claimed, result.model_dump(mode='json'), job_id=job.job_id,
                                     artifact_root=relative, manifest_sha256=digest)
            results.append(result)
        except ExperimentCancelled:
            raise
        except Exception as exc:
            control.check()
            failure = {'code': getattr(exc, 'code', 'RESEARCH_UNIT_FAILED'), 'message': str(exc)}
            repository.fail_unit(claimed, job_id=job.job_id, error=failure)
            results.append(UnitResult(key=key, status='failed', error=failure['message']))
        with db.transaction() as connection:
            current = JobRepository(db).get(job.job_id, connection)
            repository.assert_owner(connection, repository.get(job.experiment_id, connection), job.job_id, job.owner_token)
            current.completed_seeds = len([r for r in results if r.status == 'completed'])
            current.actual_steps_total = sum(r.actual_steps for r in results)
            current.training_fraction = (index + 1) / current.seed_count
            JobRepository._save(connection, current, current.revision)
    if (root / 'manifest.json').exists():
        existing = verify_research_bundle(root, check=control.check)
        if existing.get('protocol_sha256') != record['protocol_sha256'] or existing.get('environment') != record['environment']:
            raise AppError('ARTIFACT_CORRUPT', '既有研究汇总不匹配。', 409)
        return
    report = summarize_research(protocol, results)
    write_json_atomic(root / 'report.json', report.model_dump(mode='json'))
    if any(result.status != 'completed' for result in results):
        raise AppError('RESEARCH_INCOMPLETE', '部分研究单元失败；完整单元保留，可检查后显式续跑。', 409)
    control.check()
    seal_research_bundle(root, {'research_id': job.experiment_id, 'protocol_sha256': record['protocol_sha256'],
                               'environment': record['environment'], 'job_id': job.job_id,
                               'unit_manifests': {row['execution_key']: row['manifest_sha256'] for row in repository.units(job.experiment_id)}},
                         check=control.check)


class ResearchPublisher:
    def __init__(self, settings, db):
        self.settings, self.db = settings, db

    def publish(self, job, *, check_deadline=lambda: None):
        repository = ResearchRepository(self.db)
        record = repository.get(job.experiment_id)
        root = confined_path(self.settings.research_dir, job.experiment_id, must_exist=True)
        manifest = verify_research_bundle(root, check=check_deadline)
        parent_digest = hash_file(root / 'manifest.json')
        attempt_jobs = {attempt['job_id'] for attempt in repository.attempts(job.experiment_id)}
        if (manifest.get('job_id') not in attempt_jobs or manifest.get('protocol_sha256') != record['protocol_sha256']
                or manifest.get('environment') != record['environment']):
            raise AppError('PUBLISH_CONFLICT', '研究汇总产物身份不符。', 409)
        rows = repository.units(job.experiment_id)
        if not rows or any(row['status'] != 'succeeded' for row in rows):
            raise AppError('PUBLISH_CONFLICT', '研究仍有未完成单元。', 409)
        if manifest.get('unit_manifests') != {r['execution_key']: r['manifest_sha256'] for r in rows}:
            raise AppError('PUBLISH_CONFLICT', '研究汇总引用的单元不符。', 409)
        check_deadline()
        with self.db.transaction() as connection:
            record = repository.get(job.experiment_id, connection)
            repository.assert_owner(connection, record, job.job_id, job.owner_token)
            jobs = JobRepository(self.db)
            current = jobs.get(job.job_id, connection)
            current.status, current.finished_at, current.training_fraction = 'succeeded', utc_now(), 1.0
            jobs._save(connection, current, current.revision)
            jobs._event(connection, current.job_id, 'state', StateEventPayload(status='succeeded', revision=current.revision))
            record['technical_status'] = 'completed'
            record['parent_manifest_sha256'] = parent_digest
            repository.save(record, connection)
            connection.execute('''INSERT INTO artifact_protocol_registry
                (artifact_id,research_id,execution_key,protocol_fingerprint,manifest_sha256,payload_json)
                VALUES(?,?,NULL,?,?,?)''', (job.experiment_id, job.experiment_id, record['protocol_sha256'],
                parent_digest, canonical_json({'artifact_root': job.experiment_id})))
            connection.execute("UPDATE research_attempts SET status='succeeded' WHERE job_id=?", (job.job_id,))
            parent = ExperimentRepository(self.db).get(job.experiment_id, connection)
            parent.integrity = 'complete'
            ExperimentRepository(self.db).update(parent, connection)

    def recover_published(self, job, *, check=lambda: None):
        # Never auto-resume an optimizer. A complete sealed parent can be verified later via resume.
        return False
