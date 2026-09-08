"""Launcher contracts; authored for later authorized runtime verification."""

import ast
from pathlib import Path

import pytest

import run


def test_launcher_only_accepts_loopback_hosts():
    with pytest.raises(SystemExit):
        run.parser().parse_args(["--host", "0.0.0.0"])


def test_bad_port_has_a_specific_error():
    with pytest.raises(ValueError, match="端口"):
        run.preflight("127.0.0.1", 65536)


def test_launcher_has_no_automatic_build_or_install():
    tree = ast.parse((Path(__file__).parents[1] / "run.py").read_text(encoding="utf-8"))
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Attribute) and node.func.attr in {"Popen", "call", "run"}]
    for call in calls:
        literals = [node.value for arg in call.args for node in ast.walk(arg)
                    if isinstance(node, ast.Constant) and isinstance(node.value, str)]
        assert not {"npm", "install", "build", "pip"} & set(literals)
