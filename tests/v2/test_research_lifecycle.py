from uuid import uuid4

import pytest

from application_fixtures import research_files, draft_for
from stockrl_app.errors import AppError
from stockrl_app.market_datasets import MarketDatasetService
from stockrl_app.researches import ResearchService
from stockrl_app.settings import AppSettings
from stockrl_app.storage.database import initialize_database


@pytest.fixture
def services(tmp_path, monkeypatch):
    settings = AppSettings(project_root=tmp_path)
    db = initialize_database(settings.database_path)
    monkeypatch.setattr('stockrl_app.researches.environment_fingerprint', lambda settings: {'sha256': 'a'*64})
    return MarketDatasetService(settings, db), ResearchService(settings, db)


def test_preview_does_not_queue_submit_idempotent_resume_new_attempt(services):
    markets, researches = services
    dataset = markets.register(research_files())
    assert dataset['qualification'] == 'ready'
    draft = draft_for(dataset)
    preview = researches.preview(draft)
    assert preview['budget']['unit_count'] == 1
    assert researches.jobs.list().items == []
    key = str(uuid4())
    submitted = researches.submit(draft, key)
    assert researches.submit(draft, key) == submitted
    researches.jobs.request_cancel(submitted['job_id'])
    resumed = researches.resume(submitted['research_id'], str(uuid4()))
    assert resumed['job_id'] != submitted['job_id']
    attempts = researches.repository.attempts(submitted['research_id'])
    assert [row['status'] for row in attempts] == ['cancelled', 'queued']
    with pytest.raises(AppError) as error:
        researches.resume(submitted['research_id'], str(uuid4()))
    assert error.value.code == 'STATE_CONFLICT'


def test_snapshot_mutation_rejected_and_owner_fenced(services):
    markets, researches = services
    dataset = markets.register(research_files())
    draft = draft_for(dataset)
    submitted = researches.submit(draft, str(uuid4()))
    job = researches.jobs.claim_next('owner')
    key = {'instrument_id': 'SSE:600000', 'fold_id': 'fold_1', 'seed': 42}
    with pytest.raises(AppError):
        researches.repository.claim_unit(submitted['research_id'], key, job_id=job.job_id, owner_token='wrong')
    claimed = researches.repository.claim_unit(submitted['research_id'], key, job_id=job.job_id, owner_token='owner')
    researches.jobs.request_cancel(job.job_id)
    with pytest.raises(AppError):
        researches.repository.complete_unit(claimed, {}, job_id=job.job_id, artifact_root='unused', manifest_sha256='a'*64)
    markets.paths(dataset['dataset_id'])['bars.csv'].write_bytes(b'changed')
    with pytest.raises(AppError):
        researches.preview(draft)


def test_stressed_costs_fail_before_queue_and_protocol_id_is_not_reused(services):
    import json
    markets, researches = services
    files = research_files()
    profile = json.loads(files['market-profile.json'])
    profile['slippage'] = .4
    files['market-profile.json'] = json.dumps(profile).encode()
    draft = draft_for(markets.register(files))
    assert researches.preview(draft)['blockers'] == ['COST_SCENARIO_INVALID']
    with pytest.raises(AppError):
        researches.submit(draft, str(uuid4()))
    assert not researches.jobs.list().items
    normal = draft_for(markets.register(research_files()))
    researches.submit(normal, str(uuid4()))
    changed = normal.model_copy(update={'hypothesis': 'changed research question'})
    with pytest.raises(AppError) as error:
        researches.submit(changed, str(uuid4()))
    assert error.value.code == 'PROTOCOL_ID_CONFLICT'
