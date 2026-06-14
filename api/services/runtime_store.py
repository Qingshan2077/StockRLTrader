from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import pickle
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd


@dataclass
class SignalRun:
    run_id: str
    ticker: str
    models: list[str]
    created_at: str
    metrics: dict[str, Any] = field(default_factory=dict)
    predictions: dict[str, np.ndarray] = field(default_factory=dict)
    feature_importance: dict[str, Any] = field(default_factory=dict)
    test_features: pd.DataFrame | None = None
    test_labels: np.ndarray | None = None
    ensemble_signal: np.ndarray | None = None


@dataclass
class BacktestRun:
    run_id: str
    ticker: str
    mode: str
    created_at: str
    result: dict[str, Any] = field(default_factory=dict)


class RuntimeStore:
    def __init__(self, base_dir: str = "results/runtime") -> None:
        self.signal_runs: dict[str, SignalRun] = {}
        self.backtest_runs: dict[str, BacktestRun] = {}
        self.base_dir = Path(base_dir)
        self.signal_dir = self.base_dir / "signals"
        self.backtest_dir = self.base_dir / "backtests"
        self.signal_dir.mkdir(parents=True, exist_ok=True)
        self.backtest_dir.mkdir(parents=True, exist_ok=True)
        self._load_existing()

    @staticmethod
    def new_id(prefix: str) -> str:
        return f"{prefix}_{datetime.now():%Y%m%d_%H%M%S}_{uuid4().hex[:8]}"

    def save_signal_run(self, run: SignalRun) -> None:
        self.signal_runs[run.run_id] = run
        self._save_pickle(self.signal_dir / f"{run.run_id}.pkl", run)

    def get_signal_run(self, run_id: str) -> SignalRun:
        if run_id not in self.signal_runs:
            raise KeyError(run_id)
        return self.signal_runs[run_id]

    def list_signal_runs(self) -> list[SignalRun]:
        return sorted(self.signal_runs.values(), key=lambda item: item.created_at, reverse=True)

    def save_backtest_run(self, run: BacktestRun) -> None:
        self.backtest_runs[run.run_id] = run
        self._save_pickle(self.backtest_dir / f"{run.run_id}.pkl", run)

    def list_backtest_runs(self) -> list[BacktestRun]:
        return sorted(self.backtest_runs.values(), key=lambda item: item.created_at, reverse=True)

    def _load_existing(self) -> None:
        for path in self.signal_dir.glob("*.pkl"):
            try:
                run = self._load_pickle(path)
                if isinstance(run, SignalRun):
                    self.signal_runs[run.run_id] = run
            except Exception:
                continue
        for path in self.backtest_dir.glob("*.pkl"):
            try:
                run = self._load_pickle(path)
                if isinstance(run, BacktestRun):
                    self.backtest_runs[run.run_id] = run
            except Exception:
                continue

    @staticmethod
    def _save_pickle(path: Path, value: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            pickle.dump(value, fh)

    @staticmethod
    def _load_pickle(path: Path) -> Any:
        with path.open("rb") as fh:
            return pickle.load(fh)


runtime_store = RuntimeStore()
