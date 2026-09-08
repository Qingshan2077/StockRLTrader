"""Authored HTTP acceptance checks; not executed during the static-only refactor."""
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from stockrl_app.maintenance import initialize
from stockrl_app.settings import AppSettings


@pytest.fixture
def client(tmp_path):
    settings = AppSettings(project_root=tmp_path)
    initialize(settings)
    with TestClient(create_app(settings), base_url='http://127.0.0.1:8000') as http:
        yield http


def test_liveness_requires_neither_worker_nor_frontend(client):
    assert client.get('/api/v1/health/live').json()['status'] == 'alive'
    response = client.get('/api/v1/health/ready')
    assert response.status_code == 200
    assert response.json()['status'] == 'degraded'
    assert response.json()['components']['worker'] is False


def test_http_submission_idempotence_and_queued_cancellation(client):
    dataset = client.post('/api/v1/datasets/demo', json={'rows': 200}).json()
    payload = {'dataset_id': dataset['dataset_id']}
    headers = {'Idempotency-Key': str(uuid4())}
    first = client.post('/api/v1/experiments', json=payload, headers=headers)
    retry = client.post('/api/v1/experiments', json=payload, headers=headers)
    assert first.status_code == retry.status_code == 202
    assert first.json() == retry.json()
    changed = client.post('/api/v1/experiments', json={**payload, 'timesteps': 2000}, headers=headers)
    assert changed.status_code == 409
    identifier = first.json()['job_id']
    cancelled = client.post(f'/api/v1/jobs/{identifier}/cancel', json={})
    assert cancelled.status_code == 200 and cancelled.json()['status'] == 'cancelled'
    assert 'owner_token' not in cancelled.json()
    events = client.get(f'/api/v1/jobs/{identifier}/events').json()
    assert [event['seq'] for event in events['items']] == [1, 2]


@pytest.mark.parametrize('payload', [{'rows': True}, {'rows': 200, 'output_dir': '../escape'}, {'rows': 100001}])
def test_invalid_body_uses_shared_error_contract(client, payload):
    response = client.post('/api/v1/datasets/demo', json=payload)
    assert response.status_code == 422
    assert set(response.json()) == {'error', 'request_id'}
    assert response.headers['x-request-id'] == response.json()['request_id']


def test_host_origin_and_content_type_are_checked(client):
    assert client.get('/api/v1/health/live', headers={'Host': 'evil.example'}).status_code == 403
    assert client.post('/api/v1/datasets/demo', json={}, headers={'Origin': 'https://evil.example'}).status_code == 403
    assert client.post('/api/v1/datasets/demo', content='{}', headers={'Content-Type': 'text/plain'}).status_code == 422


def test_live_and_openapi_never_start_a_worker(client):
    document = client.get('/api/openapi.json').json()
    assert '/api/v1/jobs/{job_id}/cancel' in document['paths']
    assert client.get('/api/v1/jobs').json()['items'] == []


def test_deep_link_and_missing_asset(tmp_path):
    settings = AppSettings(project_root=tmp_path)
    initialize(settings)
    settings.frontend_dir.mkdir(parents=True)
    (settings.frontend_dir / 'index.html').write_text('<html>research</html>', encoding='utf-8')
    with TestClient(create_app(settings), base_url='http://127.0.0.1:8000') as client:
        assert client.get('/experiments/some-id').text == '<html>research</html>'
        assert client.get('/assets/missing.js').status_code == 404
        assert client.get('/api/missing').status_code == 404
