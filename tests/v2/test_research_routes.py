from uuid import uuid4

from fastapi.testclient import TestClient

from api.main import create_app
from application_fixtures import research_files, draft_for
from stockrl_app.settings import AppSettings
from stockrl_app.storage.database import initialize_database


def test_six_file_import_preview_and_queue(tmp_path):
    settings = AppSettings(project_root=tmp_path)
    initialize_database(settings.database_path)
    client = TestClient(create_app(settings), base_url='http://127.0.0.1')
    fields = {'metadata.json':'metadata','bars.csv':'bars','sessions.csv':'sessions','actions.csv':'actions',
              'tradability.csv':'tradability','market-profile.json':'profile'}
    upload = {fields[name]: (name, data) for name, data in research_files().items()}
    response = client.post('/api/v1/market-datasets/preview', files=upload)
    assert response.status_code == 200, response.text
    assert response.json()['qualification'] == 'ready'
    response = client.post('/api/v1/market-datasets', files=upload)
    assert response.status_code == 201, response.text
    draft = draft_for(response.json()).model_dump(mode='json')
    preview = client.post('/api/v1/research-previews', json=draft)
    assert preview.status_code == 200, preview.text
    assert preview.json()['budget']['requested_total_steps'] == 8
    response = client.post('/api/v1/researches', json=draft, headers={'Idempotency-Key': str(uuid4())})
    assert response.status_code == 202, response.text
    identifier = response.json()['research_id']
    detail = client.get('/api/v1/researches/' + identifier)
    assert detail.status_code == 200, detail.text
    assert detail.json()['economic_outcome'] == 'insufficient_evidence'
    assert detail.json()['qualification'] == 'exploratory'
    assert client.get('/api/v1/experiments').json()['items'] == []
    assert client.get('/api/v1/researches/'+identifier+'/diagnostics', params={
        'instrument_id': 'SSE:600000', 'fold_id':'fold_1', 'seed':42}).status_code == 409


def test_upload_scope_does_not_weaken_json_or_origin_limits(tmp_path):
    settings = AppSettings(project_root=tmp_path)
    initialize_database(settings.database_path)
    client = TestClient(create_app(settings), base_url='http://127.0.0.1')
    assert client.post('/api/v1/research-previews', content=b' ' * 131073,
                       headers={'Content-Type': 'application/json'}).status_code == 413
    assert client.post('/api/v1/market-datasets/preview', files={'bad': ('x', b'x')}).status_code == 422
    assert client.post('/api/v1/market-datasets/preview', files={'metadata':('x',b'{}')},
                       headers={'Origin':'https://example.org'}).status_code == 403
