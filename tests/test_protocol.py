"""Chronological invariants for the lightweight shared protocol (not executed here)."""

import ast
from pathlib import Path

import pandas as pd
import pytest

from stockrl.protocol import split_intervals


def test_split_reward_dates_are_disjoint_and_exhaustive():
    bars = pd.DataFrame(index=pd.bdate_range("2024-01-01", periods=81))
    splits = split_intervals(bars)
    assert [(s["start"], s["end"]) for s in splits.values()] == [(0, 48), (48, 64), (64, 80)]
    dates = [set(bars.index[s["start"] + 1:s["end"] + 1]) for s in splits.values()]
    assert set.union(*dates) == set(bars.index[1:])
    assert not (dates[0] & dates[1] or dates[0] & dates[2] or dates[1] & dates[2])


@pytest.mark.parametrize("ratios", [(0, .2), (.9, .2), (.6, 0)])
def test_invalid_partition_is_rejected(ratios):
    with pytest.raises(ValueError):
        split_intervals(pd.DataFrame(index=pd.bdate_range("2024-01-01", periods=81)), *ratios)


def test_protocol_does_not_import_training_frameworks():
    source = Path(__file__).parents[1] / "stockrl" / "protocol.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    imports = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    imports += [alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names]
    assert not any(name and name.split(".")[0] in {"torch", "stable_baselines3"} for name in imports)
