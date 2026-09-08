import json

import pytest

from stockrl_app.artifact_protocol import BASELINES, METRICS, baseline_aggregate, public_aggregate, report_rows
from stockrl_app.errors import AppError
from stockrl_app.legacy import LegacyImporter, known_protocol
from stockrl_app.maintenance import initialize
from stockrl_app.settings import AppSettings


def test_missing_metrics_are_not_undefined_metrics():
    with pytest.raises(AppError):
        public_aggregate({})
    missing = {name: {'mean': None, 'std': None, 'count': 0} for name in METRICS}
    assert public_aggregate(missing)['sharpe']['mean'] is None
    missing['sharpe']['count'] = -1
    with pytest.raises(AppError):
        public_aggregate(missing)


def test_aggregate_counts_match_finite_seed_metrics():
    summary = {name: {'mean': 1.0, 'std': 0.0, 'count': 999} for name in METRICS}
    with pytest.raises(AppError):
        public_aggregate(summary, runs=[{'metrics': {name: 1.0 for name in METRICS}}])
    summary['sharpe']['mean'] = 10 ** 400
    with pytest.raises(AppError):
        public_aggregate(summary)


def test_baseline_summary_uses_every_seed_and_preserves_undefined():
    runs = [{'baselines': {policy: {'metrics': {name: value for name in METRICS}}
                           for policy in BASELINES}} for value in (1.0, 3.0, None)]
    result = baseline_aggregate(runs)
    assert result['cash']['total_return'] == {'mean': 2.0, 'std': 1.0, 'count': 2}


def test_history_csv_validates_unsampled_rows(tmp_path):
    path = tmp_path / 'history.csv'
    path.write_text('date,nav,cash,shares,weight,requested_weight,executed_weight,turnover,cost\n'
                    '2020-01-01,100,100,0,0,0,0,0,0\n'
                    '2020-01-02,NaN,100,0,0,0,0,0,0\n', encoding='utf-8')
    with pytest.raises(AppError):
        list(report_rows(path, history=True))


def test_empty_trades_keep_headers(tmp_path):
    path = tmp_path / 'trades.csv'
    path.write_text('date,side,shares,price,cost\n', encoding='utf-8')
    assert list(report_rows(path, history=False)) == []


def test_bad_legacy_folder_does_not_stop_other_reports(tmp_path):
    settings = AppSettings(project_root=tmp_path)
    db = initialize(settings)
    for name, summary in [('bad', '{'), ('unknown', json.dumps({'runs': [], 'versions': []}))]:
        directory = settings.output_dir / name
        directory.mkdir()
        (directory / 'summary.json').write_text(summary, encoding='utf-8')
    records = LegacyImporter(settings, db).scan()
    assert len(records) == 2
    assert all(record.integrity == 'corrupt' for record in records)
    assert [record.experiment_id for record in LegacyImporter(settings, db).scan()] == [record.experiment_id for record in records]
    assert (settings.output_dir / 'bad' / 'summary.json').read_text(encoding='utf-8') == '{'


def test_legacy_unknown_protocol_is_total_for_malformed_json():
    assert known_protocol({'versions': [], 'runs': [None]}) is False
