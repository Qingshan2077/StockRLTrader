from io import BytesIO

import pytest

from stockrl_app.datasets import DatasetService
from stockrl_app.errors import AppError
from stockrl_app.models import DatasetMetadata, DemoDatasetRequest, LocalDatasetRequest
from stockrl_app.settings import AppSettings
from stockrl_app.storage.database import Database
from stockrl_app.storage.migrations import initialize_database


@pytest.fixture
def service(tmp_path):
    settings = AppSettings(project_root=tmp_path)
    initialize_database(settings.database_path)
    return DatasetService(settings, Database(settings.database_path))


def test_demo_is_always_marked_and_preview_is_bounded(service):
    dataset = service.create_demo(DemoDatasetRequest(rows=200))
    assert dataset.is_synthetic and dataset.quote_unit == 'synthetic_unit'
    assert dataset.adjustment == 'unknown'
    assert len(service.preview(dataset.dataset_id).sample) == 100


@pytest.mark.parametrize('content', [b'', b'Date,Open,High,Low,Close,Volume\n2020-01-01,2,1,1,2,10\n', b'Date,Open,High,Low,Close,Volume\n2020-01-01,1,2,1,1,10\n2020-01-01,1,2,1,1,10\n'])
def test_invalid_csv_does_not_register(service, content):
    with pytest.raises(AppError):
        service.create_csv(BytesIO(content), '../../escape.csv', DatasetMetadata())
    assert service.list().items == []


def test_stream_limit_is_enforced_before_registration(service):
    chunks = iter([b'x' * (1024 * 1024)] * 21)
    with pytest.raises(AppError) as error:
        service.create_csv(chunks, 'oversize.csv')
    assert error.value.status_code == 413
    assert service.list().items == []


def test_local_snapshot_survives_source_change(service):
    root = service.settings.local_source_dir
    root.mkdir(parents=True)
    original = root / 'prices_raw.csv'
    original.write_text('Date,Open,High,Low,Close,Volume\n2020-01-01,1,2,1,1,10\n', encoding='utf-8')
    source = service.local_sources()[0]
    dataset = service.create_local(LocalDatasetRequest(source_id=source.source_id))
    original.write_text('changed', encoding='utf-8')
    assert service.read_bars(dataset.dataset_id).iloc[0]['Close'] == 1
    assert dataset.quote_unit == 'unknown'


def test_row_limit_precedes_registration(service):
    body = b'Date,Open,High,Low,Close,Volume\n' + b'2020-01-01,1,2,1,1,10\n' * 100001
    with pytest.raises(AppError) as error:
        service.create_csv(BytesIO(body), 'rows.csv')
    assert error.value.code == 'DATASET_ROW_LIMIT'
    assert service.list().items == []


def test_malicious_upload_name_never_controls_storage_path(service):
    body = b'Date,Open,High,Low,Close,Volume\n2020-01-01,1,2,1,1,10\n'
    dataset = service.create_csv(BytesIO(body), '../../escape.csv')
    assert dataset.display_name == 'escape.csv'
    record = service.repository.get(dataset.dataset_id)
    assert record.snapshot_relative_path == f'datasets/{dataset.dataset_id}/bars.csv'
    assert not (service.settings.project_root / 'escape.csv').exists()


def test_local_symlink_outside_allowlist_is_not_selectable(service, tmp_path):
    outside = tmp_path / 'outside.csv'
    outside.write_text('private', encoding='utf-8')
    root = service.settings.local_source_dir
    root.mkdir(parents=True)
    try:
        (root / 'linked_raw.csv').symlink_to(outside)
    except OSError:
        pytest.skip('Host does not allow symlink creation')
    assert service.local_sources() == []


def test_processed_csv_is_not_a_local_source(service):
    root = service.settings.local_source_dir
    root.mkdir(parents=True)
    (root / 'prices_processed.csv').write_text('Date,Open,High,Low,Close,Volume\n', encoding='utf-8')
    assert service.local_sources() == []


def test_tampered_snapshot_cannot_be_previewed(service):
    dataset = service.create_demo(DemoDatasetRequest(rows=60))
    record = service.repository.get(dataset.dataset_id)
    path = service.settings.app_dir / record.snapshot_relative_path
    path.write_text('tampered', encoding='utf-8')
    with pytest.raises(AppError) as error:
        service.preview(dataset.dataset_id)
    assert error.value.code == 'DATASET_INTEGRITY'
