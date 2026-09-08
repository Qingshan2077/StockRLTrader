from pathlib import Path

import pytest

from stockrl_app.errors import AppError
from stockrl_app.settings import AppSettings, confined_path


def test_default_database_is_isolated_from_legacy(tmp_path):
    settings = AppSettings(project_root=tmp_path)
    assert settings.database_path == tmp_path / 'outputs' / 'app' / 'state.sqlite3'
    assert not settings.database_path.exists()
    assert settings.output_dir == tmp_path / 'outputs' / 'experiments'


def test_roots_are_fixed_and_escape_is_rejected(tmp_path):
    settings = AppSettings(project_root=tmp_path)
    with pytest.raises((AttributeError, TypeError)):
        settings.app_dir = Path('elsewhere')
    with pytest.raises(AppError):
        confined_path(settings.app_dir, '../escape')


def test_network_root_is_rejected(tmp_path):
    with pytest.raises(AppError):
        AppSettings(project_root=tmp_path, app_dir='//server/share/state')


def test_environment_overrides_fixed_roots_and_same_origin(tmp_path, monkeypatch):
    monkeypatch.setenv('STOCKRL_APP_DIR', 'private/app')
    monkeypatch.setenv('STOCKRL_OUTPUT_DIR', 'private/results')
    monkeypatch.setenv('STOCKRL_PORT', '8123')
    settings = AppSettings.from_env(project_root=tmp_path)
    assert settings.database_path == tmp_path / 'private' / 'app' / 'state.sqlite3'
    assert settings.output_dir == tmp_path / 'private' / 'results'
    assert 'http://127.0.0.1:8123' in settings.allowed_origins
    assert settings.frontend_dir == tmp_path / 'api' / 'static'


def test_symlink_retargeting_of_fixed_root_is_rejected(tmp_path):
    root = tmp_path / 'fixed'
    target = tmp_path / 'other'
    root.mkdir()
    target.mkdir()
    anchored = root.resolve()
    root.rmdir()
    try:
        root.symlink_to(target, target_is_directory=True)
    except OSError:
        pytest.skip('Host does not allow symlink creation')
    with pytest.raises(AppError) as error:
        confined_path(anchored, 'file.csv')
    assert error.value.code == 'ROOT_CHANGED'
