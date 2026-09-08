"""Staged, verified publication with durable intent and recovery fencing."""
import json
import os
from collections.abc import Callable
from pathlib import Path
from uuid import uuid4

from .artifact_protocol import (BASELINES, baseline_aggregate, hash_file, manifest_entries, public_aggregate,
                                public_runs, read_json, report_rows)
from .errors import AppError
from .models import ArtifactStorageRecord, ExperimentRecord, JobRecord, Versions, utc_now
from .settings import AppSettings, confined_path
from .storage.database import Database
from .storage.experiments import ExperimentRepository
from .storage.jobs import JobRepository


def write_json(path: Path, value: dict) -> None:
    with path.open('x', encoding='utf-8', newline='\n') as target:
        json.dump(value, target, ensure_ascii=False, indent=2, allow_nan=False)
        target.write('\n')
        target.flush()
        os.fsync(target.fileno())


def artifact_kind(name: str) -> str:
    return {'summary.json': 'summary', 'request.json': 'request', 'bars.csv': 'bars',
            'manifest.json': 'manifest', 'history.csv': 'history', 'trades.csv': 'trades',
            'model.zip': 'model', 'normalizer.json': 'normalizer', 'training.json': 'training',
            'evaluations.npz': 'validation'}.get(Path(name).name, 'supporting')


def seal_bundle(root: Path, *, job: JobRecord, experiment: ExperimentRecord,
                metadata: dict) -> None:
    """Called by the child after all core writes have closed; no files change afterwards."""
    write_json(root / 'request.json', {
        'request': experiment.request.model_dump(mode='json') if experiment.request else None,
        'request_sha256': experiment.request_sha256,
        'source_experiment_id': experiment.source_experiment_id,
        'purpose': experiment.request.purpose if experiment.request else 'legacy_replay',
    })
    entries = []
    for path in sorted(root.rglob('*')):
        if path.is_symlink():
            raise AppError('ARTIFACT_CORRUPT', '产物不能包含文件链接。', 409)
        if not path.is_file():
            continue
        name = path.relative_to(root).as_posix()
        safe = confined_path(root, name, must_exist=True)
        # Flush closed core files before publishing the manifest that attests to them.
        with safe.open('rb+') as source:
            os.fsync(source.fileno())
        entries.append({'path': name, 'size': safe.stat().st_size,
                        'sha256': hash_file(safe), 'kind': artifact_kind(name)})
    write_json(root / 'manifest.json', {
        'schema_version': 1, 'core_semantics_version': 1, 'metrics_version': 1,
        'experiment_id': experiment.experiment_id, 'job_id': job.job_id,
        'owner_token': job.owner_token, 'kind': experiment.kind,
        'source_experiment_id': experiment.source_experiment_id,
        'request_sha256': experiment.request_sha256,
        'filtered_data_sha256': experiment.filtered_data_sha256,
        'data_label': experiment.data_label, 'is_synthetic': experiment.is_synthetic,
        'dataset': experiment.dataset.model_dump(mode='json') if experiment.dataset else None,
        'legacy_config': experiment.legacy_config.model_dump(mode='json') if experiment.legacy_config else None,
        'created_at': utc_now(), 'files': entries, **metadata,
    })


class ArtifactPublisher:
    def __init__(self, settings: AppSettings, db: Database):
        self.settings, self.db = settings, db
        self.jobs = JobRepository(db)
        self.experiments = ExperimentRepository(db)

    def staged_path(self, job: JobRecord) -> Path:
        return confined_path(self.settings.output_dir, f'.staging/{job.job_id}/bundle', must_exist=True)

    def verify(self, job: JobRecord, root: Path, *, expected_hash: str | None = None,
               check: Callable[[], None] = lambda: None
               ) -> tuple[ExperimentRecord, list[ArtifactStorageRecord], str]:
        manifest, entries = manifest_entries(root, expected_hash=expected_hash, check=check)
        experiment = self.experiments.get(job.experiment_id)
        if (manifest.get('experiment_id') != job.experiment_id or manifest.get('job_id') != job.job_id
                or manifest.get('owner_token') != job.owner_token
                or manifest.get('kind') != job.kind
                or manifest.get('source_experiment_id') != experiment.source_experiment_id
                or manifest.get('request_sha256') != experiment.request_sha256
                or manifest.get('filtered_data_sha256') != experiment.filtered_data_sha256
                or manifest.get('core_semantics_version') != 1 or manifest.get('metrics_version') != 1):
            raise AppError('PUBLISH_CONFLICT', '产物身份或研究协议与任务不一致。', 409)
        summary = read_json(root / 'summary.json')
        saved_request = read_json(root / 'request.json')
        expected_request = experiment.request.model_dump(mode='json') if experiment.request else None
        if (saved_request.get('request') != expected_request
                or saved_request.get('request_sha256') != experiment.request_sha256
                or saved_request.get('source_experiment_id') != experiment.source_experiment_id):
            raise AppError('ARTIFACT_CORRUPT', '保存的请求与已提交配置不一致。', 409)
        runs = public_runs(summary)
        expected_seeds = experiment.request.seeds if experiment.request else [
            run['seed'] for run in self.experiments.get(experiment.source_experiment_id).runs]
        if [run['seed'] for run in runs] != expected_seeds or len(runs) != job.seed_count:
            raise AppError('ARTIFACT_CORRUPT', '产物 seed 不完整或顺序不符。', 409)
        if (summary.get('splits') != {key: value.model_dump() for key, value in experiment.splits.items()}
                or summary.get('data_sha256') != hash_file(root / 'bars.csv', check=check)):
            raise AppError('ARTIFACT_CORRUPT', '产物日期切分或数据指纹与记录不一致。', 409)
        if (job.kind == 'train' and (summary.get('data_sha256') != experiment.filtered_data_sha256
                or summary.get('config') != experiment.request.trading_config.model_dump()
                or summary.get('algorithm') != experiment.request.algorithm
                or summary.get('timesteps') != experiment.request.timesteps)):
            raise AppError('ARTIFACT_CORRUPT', '训练产物的数据或交易配置与请求不一致。', 409)
        required = {'summary.json', 'bars.csv', 'request.json'}
        for seed in expected_seeds:
            prefix = f'seed_{seed}/'
            required.update(prefix + name for name in ('model.zip', 'normalizer.json', 'training.json',
                                                       'history.csv', 'trades.csv'))
            for baseline in BASELINES:
                required.update(prefix + f'baselines/{baseline}/' + name for name in ('history.csv', 'trades.csv'))
        if not required.issubset({entry['path'] for entry in entries}):
            raise AppError('ARTIFACT_CORRUPT', '实验缺少完整模型、基准或重放文件。', 409)
        for entry in entries:
            check()
            if Path(entry['path']).name in ('history.csv', 'trades.csv'):
                for _ in report_rows(confined_path(root, entry['path'], must_exist=True),
                                     history=entry['path'].endswith('/history.csv'), check=check,
                                     interval=experiment.splits['test'].model_dump()):
                    pass
        experiment.output_relative_path = experiment.experiment_id
        experiment.run_id = summary.get('run_id') if isinstance(summary.get('run_id'), str) else None
        experiment.runs, experiment.aggregate = runs, public_aggregate(summary.get('aggregate'), runs=runs)
        experiment.baseline_aggregate = baseline_aggregate(runs)
        experiment.versions = Versions()
        experiment.integrity, experiment.replayable, experiment.replay_block_reason = 'complete', True, None
        manifest_hash = hash_file(root / 'manifest.json', check=check)
        entries = [*entries, {'path': 'manifest.json', 'size': (root / 'manifest.json').stat().st_size,
                             'sha256': manifest_hash, 'kind': 'manifest'}]
        records = []
        for entry in entries:
            artifact_id = str(uuid4())
            records.append(ArtifactStorageRecord(artifact_id=artifact_id, experiment_id=experiment.experiment_id,
                kind=artifact_kind(entry['path']), filename=entry['path'], size=entry['size'],
                sha256=entry['sha256'], download_url=f'/api/v1/artifacts/{artifact_id}/download',
                relative_path=f"{experiment.experiment_id}/{entry['path']}", created_at=utc_now()))
        return experiment, records, manifest_hash

    def publish(self, job: JobRecord, *, check_deadline: Callable[[], None] = lambda: None) -> JobRecord:
        stage = self.staged_path(job)
        experiment, records, fingerprint = self.verify(job, stage, check=check_deadline)
        target = confined_path(self.settings.output_dir, experiment.experiment_id)
        if target.exists():
            raise AppError('PUBLISH_CONFLICT', '目标实验目录已经存在，拒绝覆盖。', 409)
        check_deadline()
        intent = self.jobs.publish_intent(job.job_id, owner_token=job.owner_token, revision=job.revision,
                                         target_relative_path=experiment.experiment_id, manifest_sha256=fingerprint)

        def move() -> None:
            # Windows rename rejects existing destinations; POSIX can replace an empty directory,
            # so check again under the single coordinator's publication transaction.
            if target.exists():
                raise AppError('PUBLISH_CONFLICT', '目标实验目录已经存在，拒绝覆盖。', 409)
            check_deadline()
            stage.rename(target)

        return self.jobs.publish_success(job.job_id, owner_token=intent.owner_token, revision=intent.revision,
                                         experiment=experiment, artifacts=records, move=move)

    def recover_published(self, job: JobRecord, *, check: Callable[[], None] = lambda: None) -> bool:
        """Only a persisted, unchanged intent can repair rename-before-commit failure."""
        if (job.status != 'running' or job.cancel_requested or job.publish_target != job.experiment_id
                or not job.publish_manifest_sha256 or job.publish_owner_token != job.owner_token
                or job.publish_revision != job.revision):
            return False
        target = confined_path(self.settings.output_dir, job.publish_target)
        if not target.is_dir():
            return False
        experiment, records, _ = self.verify(job, target, expected_hash=job.publish_manifest_sha256, check=check)
        self.jobs.publish_success(job.job_id, owner_token=job.owner_token, revision=job.revision,
                                  experiment=experiment, artifacts=records, move=lambda: None)
        return True
