from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from uuid import uuid4

import pytest
from pydantic import ValidationError

from stockrl_app.datasets import DatasetService
from stockrl_app.errors import AppError
from stockrl_app.experiments import ExperimentService
from stockrl_app.models import DemoDatasetRequest, ExperimentRequest
from stockrl_app.settings import AppSettings
from stockrl_app.storage.database import Database
from stockrl_app.storage.migrations import initialize_database


@pytest.fixture
def service(tmp_path):
    settings = AppSettings(project_root=tmp_path)
    initialize_database(settings.database_path)
    db = Database(settings.database_path)
    dataset = DatasetService(settings, db).create_demo(DemoDatasetRequest(rows=60))
    return ExperimentService(settings, db), dataset


@pytest.mark.parametrize('fields', [{'timesteps': True}, {'seeds': [42, 42]}, {'train_ratio': float('nan')}, {'unknown': 1}, {'purpose': 'research'}, {'trading_config': {'lot_size': True}}, {'seeds': [False]}])
def test_strict_requests(fields):
    with pytest.raises(ValidationError):
        ExperimentRequest(dataset_id=str(uuid4()), **fields)


def test_defaults_and_preview_do_not_submit(service):
    experiments, dataset = service
    request = ExperimentRequest(dataset_id=dataset.dataset_id)
    assert (request.algorithm, request.timesteps, request.seeds, request.episode_length) == ('PPO', 10000, [42], 126)
    preview = experiments.preview(request)
    assert preview.splits['train'].reward_count == 35
    assert experiments.list().items == []


def test_concurrent_key_is_atomic_and_conflict_is_409(service):
    experiments, dataset = service
    request = ExperimentRequest(dataset_id=dataset.dataset_id)
    key = str(uuid4())
    ready = Barrier(2)
    def submit(_):
        ready.wait(timeout=5)
        return ExperimentService(experiments.settings, Database(experiments.db.path)).submit(request, key)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(submit, range(2)))
    assert results[0] == results[1]
    with pytest.raises(AppError) as error:
        experiments.submit(request.model_copy(update={'timesteps': 99}), key)
    assert error.value.status_code == 409


def test_capacity_allows_existing_key(service):
    experiments, dataset = service
    request = ExperimentRequest(dataset_id=dataset.dataset_id)
    key = str(uuid4())
    first = experiments.submit(request, key)
    for _ in range(19):
        experiments.submit(request, str(uuid4()))
    assert experiments.submit(request, key) == first
    with pytest.raises(AppError) as error:
        experiments.submit(request, str(uuid4()))
    assert error.value.status_code == 429


def test_filter_creates_distinct_experiment_bars_without_changing_dataset(service):
    experiments, dataset = service
    bars = experiments.datasets.read_bars(dataset.dataset_id)
    request = ExperimentRequest(dataset_id=dataset.dataset_id, start_date=bars.index[5].date().isoformat(),
                                end_date=bars.index[-5].date().isoformat())
    preview = experiments.preview(request)
    result = experiments.submit(request, str(uuid4()))
    record = experiments.repository.get(result.experiment_id)
    assert preview.splits['train'].observation_start_date == bars.index[5].isoformat()
    assert preview.splits['test'].last_reward_date == bars.index[-5].isoformat()
    assert record.filtered_data_sha256 == preview.filtered_data_sha256
    assert record.filtered_data_sha256 != dataset.snapshot_sha256
    assert len(experiments.datasets.read_bars(dataset.dataset_id)) == 60
    assert (experiments.settings.app_dir / record.bars_relative_path).exists()


def test_filtered_pagination_has_no_duplicates_and_scope_is_checked(service):
    experiments, dataset = service
    expected = set()
    for algorithm in ('PPO', 'SAC', 'PPO', 'SAC', 'PPO'):
        result = experiments.submit(ExperimentRequest(dataset_id=dataset.dataset_id, algorithm=algorithm), str(uuid4()))
        if algorithm == 'PPO':
            expected.add(result.experiment_id)
    first = experiments.list(algorithm='PPO', dataset_id=dataset.dataset_id, status='queued', limit=2)
    second = experiments.list(algorithm='PPO', dataset_id=dataset.dataset_id, status='queued', limit=2, cursor=first.next_cursor)
    assert len(first.items) == 2 and len(second.items) == 1
    assert {item.experiment_id for item in first.items + second.items} == expected
    with pytest.raises(AppError) as error:
        experiments.list(algorithm='SAC', cursor=first.next_cursor)
    assert error.value.code == 'INVALID_CURSOR'


def test_existing_key_is_recovered_even_after_snapshot_is_tampered(service):
    experiments, dataset = service
    request = ExperimentRequest(dataset_id=dataset.dataset_id)
    key = str(uuid4())
    first = experiments.submit(request, key)
    stored = experiments.datasets.repository.get(dataset.dataset_id)
    (experiments.settings.app_dir / stored.snapshot_relative_path).write_text('tampered', encoding='utf-8')
    assert experiments.submit(request, key) == first
    with pytest.raises(AppError):
        experiments.submit(request, str(uuid4()))
