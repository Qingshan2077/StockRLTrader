"""Startup import regressions; runtime execution deferred under static-only scope."""
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize('module', ['stockrl_app.jobs.worker', 'api.main'])
def test_entry_modules_import_without_starting_services(module):
    # A class method named list must not break annotations while loading either entry.
    options = {'creationflags': subprocess.CREATE_NO_WINDOW} if os.name == 'nt' else {}
    result = subprocess.run(
        [sys.executable, '-c', 'import importlib, sys; importlib.import_module(sys.argv[1])', module],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True,
        encoding='utf-8', errors='replace', timeout=30, check=False, **options,
    )
    assert result.returncode == 0, result.stdout + result.stderr
