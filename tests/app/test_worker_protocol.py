"""Authored verification cases; execution is deferred by the current user instruction."""
from types import SimpleNamespace

import pytest

from stockrl_app.errors import AppError
from stockrl_app.jobs.compute import DEPENDENCIES, assert_replay_compatible, make_portable
from stockrl_app.jobs.worker import map_event
from stockrl_app.models import JobRecord, utc_now


def test_replay_rejects_any_unknown_dependency_before_loading_models():
    installed = {name: '1.2.3' for name in DEPENDENCIES}
    versions = SimpleNamespace(core_semantics_version=1, metrics_version=1)
    assert_replay_compatible({'versions': installed}, versions, installed)
    for name in DEPENDENCIES:
        changed = {**installed, name: '1.2.4'}
        with pytest.raises(AppError) as error:
            assert_replay_compatible({'versions': changed}, versions, installed)
        assert error.value.code == 'REPLAY_INCOMPATIBLE'
    with pytest.raises(AppError):
        assert_replay_compatible({'versions': installed}, SimpleNamespace(
            core_semantics_version='unknown', metrics_version=1), installed)


def test_progress_preserves_actual_rollout_and_replay_has_no_training_steps():
    job = JobRecord(job_id='job', experiment_id='experiment', kind='train', created_at=utc_now(),
                    seed_count=2, requested_steps_total=20)
    observed = map_event({'event': 'seed_complete', 'seed': 42, 'seed_index': 0, 'timesteps': 2048}, job)
    assert observed == {'seed': 42, 'seed_index': 0, 'completed_seeds': 1, 'actual_steps': 2048}
    job.kind = 'replay'
    assert map_event({'event': 'seed_complete', 'seed': 42, 'seed_index': 0}, job) == {
        'seed': 42, 'seed_index': 0, 'completed_seeds': 1}


def test_portable_summary_preserves_metrics_and_replaces_only_storage_paths(tmp_path):
    summary = {'output_dir': 'C:/old', 'data_path': 'C:/old/bars.csv', 'source_run_dir': 'C:/source',
               'runs': [{'seed': 42, 'metrics': {'total_return': .125}, 'model_path': 'C:/old/model.zip',
                         'normalizer_path': 'C:/old/normalizer.json', 'history_path': 'C:/old/history.csv',
                         'baselines': {'cash': {'metrics': {'total_return': 0.0}}}}]}
    make_portable(tmp_path, summary)
    assert summary['output_dir'] == '.'
    assert 'source_run_dir' not in summary
    assert summary['runs'][0]['metrics'] == {'total_return': .125}
    assert summary['runs'][0]['model_path'] == 'seed_42/model.zip'
    assert summary['runs'][0]['baselines']['cash']['history_path'] == 'seed_42/baselines/cash/history.csv'
