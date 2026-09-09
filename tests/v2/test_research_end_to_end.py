from uuid import uuid4

from application_fixtures import research_files, draft_for
from stockrl_app.jobs.worker import Worker
from stockrl_app.market_datasets import MarketDatasetService
from stockrl_app.researches import ResearchService
from stockrl_app.settings import AppSettings
from stockrl_app.storage.database import initialize_database


def test_spawned_worker_publishes_short_synthetic_research(tmp_path, monkeypatch):
    settings = AppSettings(project_root=tmp_path, task_time_limit_seconds=300)
    db = initialize_database(settings.database_path)
    (settings.app_dir / 'locks').mkdir()
    markets = MarketDatasetService(settings, db)
    service = ResearchService(settings, db)
    draft = draft_for(markets.register(research_files())).model_copy(update={'seeds': (43, 42)})
    submitted = service.submit(draft, str(uuid4()))
    worker = Worker(settings)
    job = worker.jobs.claim_next(worker.owner)
    worker.run_job(job)
    finished = worker.jobs.get(job.job_id)
    assert finished.status == 'succeeded', finished.model_dump()
    report = service.get(submitted['research_id'])
    assert report['technical_status'] == 'completed'
    assert report['economic_outcome'] == 'insufficient_evidence'
    assert len(report['units']) == 2
    assert [unit['key']['seed'] for unit in report['units']] == [43, 42]
    unit = report['units'][0]
    assert len(unit['cost_results']) == 3
    assert len(unit['cost_results'][0]['metrics']) == 10
    assert 0 < len(unit['candidates']) <= 2
    assert service.diagnostics(submitted['research_id'], 'SSE:600000', 'fold_1', 42)
    # Simulate a coordinator crash after unit rename but before its DB indexing.
    root = settings.research_dir / submitted['research_id']
    (root / 'manifest.json').unlink()
    (root / 'report.json').unlink()
    with db.transaction() as connection:
        finished.status = 'interrupted'
        connection.execute('UPDATE jobs SET status=?,payload_json=? WHERE job_id=?',
                           ('interrupted', finished.model_dump_json(), finished.job_id))
        connection.execute("UPDATE research_units SET status='interrupted',artifact_root=NULL,manifest_sha256=NULL,payload_json='{}'")
        connection.execute('DELETE FROM artifact_protocol_registry')
    resumed = service.resume(submitted['research_id'], str(uuid4()))
    new_job = worker.jobs.claim_next(worker.owner)
    assert new_job.job_id == resumed['job_id']
    def must_not_retrain(*args, **kwargs):
        raise AssertionError('Complete renamed unit must be reconciled without optimizer restart')
    monkeypatch.setattr('stockrl.research.runner.run_unit', must_not_retrain)
    from stockrl.control import ExecutionControl
    from stockrl_app.research_compute import execute_research, ResearchPublisher
    execute_research(settings, db, new_job, ExecutionControl())
    ResearchPublisher(settings, db).publish(worker.jobs.get(new_job.job_id))
    assert worker.jobs.get(new_job.job_id).status == 'succeeded'
    assert len(service.repository.attempts(submitted['research_id'])) == 2
    # A missing published parent must never retain a completed/candidate verdict.
    (root / 'report.json').write_text('{}', encoding='utf-8')
    import pytest
    from stockrl_app.errors import AppError
    with pytest.raises(AppError) as error:
        service.get(submitted['research_id'])
    assert error.value.code == 'ARTIFACT_CORRUPT'
