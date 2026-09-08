"""Publication crash/cancellation boundaries, without training or model deserialization."""
import shutil
from uuid import uuid4

import pytest

from stockrl_app.artifact_protocol import BASELINES, METRICS, hash_file
from stockrl_app.artifacts import ArtifactPublisher, seal_bundle, write_json
from stockrl_app.datasets import DatasetService
from stockrl_app.errors import AppError
from stockrl_app.experiments import ExperimentService
from stockrl_app.models import DemoDatasetRequest, ExperimentRequest
from stockrl_app.settings import AppSettings
from stockrl_app.storage.artifacts import ArtifactRepository
from stockrl_app.storage.database import Database
from stockrl_app.storage.experiments import ExperimentRepository
from stockrl_app.storage.jobs import JobRepository
from stockrl_app.storage.migrations import initialize_database


@pytest.fixture
def publication(tmp_path):
    settings = AppSettings(project_root=tmp_path)
    initialize_database(settings.database_path)
    db = Database(settings.database_path)
    dataset = DatasetService(settings, db).create_demo(DemoDatasetRequest(rows=60))
    ExperimentService(settings, db).submit(ExperimentRequest(dataset_id=dataset.dataset_id), str(uuid4()))
    jobs = JobRepository(db)
    job = jobs.claim_next('test-owner')
    job = jobs.update_progress(job.job_id, owner_token=job.owner_token, revision=job.revision,
                                seed=42, seed_index=0, completed_seeds=1, actual_steps=10000)
    experiment = ExperimentRepository(db).get(job.experiment_id)
    stage = settings.staging_dir / job.job_id / 'bundle'
    stage.mkdir(parents=True)
    shutil.copyfile(settings.app_dir / experiment.bars_relative_path, stage / 'bars.csv')
    metrics = {name: None for name in METRICS}
    summary = {'run_id': 'legacy-core-identity', 'algorithm': 'PPO', 'timesteps': 10000,
               'config': experiment.request.trading_config.model_dump(),
               'data_sha256': hash_file(stage / 'bars.csv'),
               'splits': {key: value.model_dump() for key, value in experiment.splits.items()},
               'runs': [{'seed': 42, 'metrics': metrics, 'checkpoint_selection': 'validation_mean_reward',
                         'baselines': {name: {'metrics': metrics} for name in BASELINES}}],
               'aggregate': {name: {'mean': None, 'std': None, 'count': 0} for name in METRICS}}
    write_json(stage / 'summary.json', summary)
    seed = stage / 'seed_42'
    seed.mkdir()
    for name in ('model.zip', 'normalizer.json', 'training.json'):
        (seed / name).write_bytes(b'fixture bytes; never loaded as a model')
    interval = experiment.splits['test']
    dates = DatasetService(settings, db).read_bars(dataset.dataset_id).index[interval.start:interval.end + 1]
    history = ('date,nav,cash,shares,weight,requested_weight,executed_weight,turnover,cost\n'
               + ''.join(f'{date.date().isoformat()},100,100,0,0,0,0,0,0\n' for date in dates))
    trades = 'date,side,shares,price,cost\n'
    (seed / 'history.csv').write_text(history, encoding='utf-8')
    (seed / 'trades.csv').write_text(trades, encoding='utf-8')
    for baseline in BASELINES:
        directory = seed / 'baselines' / baseline
        directory.mkdir(parents=True)
        (directory / 'history.csv').write_text(history, encoding='utf-8')
        (directory / 'trades.csv').write_text(trades, encoding='utf-8')
    seal_bundle(stage, job=job, experiment=experiment, metadata={'seeds': [42]})
    return settings, db, jobs, job, stage


def test_cancel_committing_before_publication_keeps_staging_unpublished(publication):
    settings, db, jobs, job, stage = publication
    jobs.request_cancel(job.job_id)
    with pytest.raises(AppError):
        ArtifactPublisher(settings, db).publish(job)
    assert stage.exists()
    assert not (settings.output_dir / job.experiment_id).exists()
    assert jobs.get(job.job_id).status == 'cancelling'


def test_rename_then_database_failure_recovers_only_persisted_intent(publication, monkeypatch):
    settings, db, jobs, job, stage = publication
    original = ArtifactRepository.insert

    def fail_index(*args, **kwargs):
        raise RuntimeError('Injected after filesystem rename, before SQL commit')

    monkeypatch.setattr(ArtifactRepository, 'insert', fail_index)
    with pytest.raises(RuntimeError):
        ArtifactPublisher(settings, db).publish(job)
    assert not stage.exists()
    pending = jobs.get(job.job_id)
    assert pending.status == 'running'
    assert pending.publish_manifest_sha256
    assert ArtifactRepository(db).list(job.experiment_id) == []
    monkeypatch.setattr(ArtifactRepository, 'insert', original)
    assert ArtifactPublisher(settings, db).recover_published(pending)
    assert jobs.get(job.job_id).status == 'succeeded'
    saved = ExperimentRepository(db).get(job.experiment_id)
    assert saved.run_id == 'legacy-core-identity'
    assert saved.experiment_id != saved.run_id
    assert any(item.filename == 'seed_42/baselines/cash/history.csv' for item in saved.artifacts)


def test_existing_target_is_never_overwritten(publication):
    settings, db, jobs, job, stage = publication
    publisher = ArtifactPublisher(settings, db)
    target = settings.output_dir / job.experiment_id
    target.mkdir()
    sentinel = target / 'user.txt'
    sentinel.write_text('keep', encoding='utf-8')
    with pytest.raises(AppError):
        publisher.publish(job)
    assert sentinel.read_text(encoding='utf-8') == 'keep'
    assert stage.exists()
    assert jobs.get(job.job_id).publish_target is None


def test_cancel_after_rename_prevents_recovery(publication):
    settings, db, jobs, job, stage = publication
    publisher = ArtifactPublisher(settings, db)
    _, _, fingerprint = publisher.verify(job, stage)
    intent = jobs.publish_intent(job.job_id, owner_token=job.owner_token, revision=job.revision,
        target_relative_path=job.experiment_id, manifest_sha256=fingerprint)
    stage.rename(settings.output_dir / job.experiment_id)
    cancelling = jobs.request_cancel(job.job_id)
    assert cancelling.revision > intent.revision
    assert publisher.recover_published(cancelling) is False
    assert jobs.get(job.job_id).status == 'cancelling'
    assert ArtifactRepository(db).list(job.experiment_id) == []


def test_modified_manifest_cannot_repair_publication(publication):
    settings, db, jobs, job, stage = publication
    publisher = ArtifactPublisher(settings, db)
    _, _, fingerprint = publisher.verify(job, stage)
    intent = jobs.publish_intent(job.job_id, owner_token=job.owner_token, revision=job.revision,
        target_relative_path=job.experiment_id, manifest_sha256=fingerprint)
    target = settings.output_dir / job.experiment_id
    stage.rename(target)
    with (target / 'manifest.json').open('a', encoding='utf-8') as output:
        output.write(' ')
    with pytest.raises(AppError) as error:
        publisher.recover_published(intent)
    assert error.value.code == 'ARTIFACT_CORRUPT'
    assert jobs.get(job.job_id).status == 'running'
