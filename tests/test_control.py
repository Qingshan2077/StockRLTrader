"""Behavioral checks for cooperative execution; authored without execution."""

import pytest

from stockrl.control import ExecutionControl, ExperimentCancelled
from stockrl.data import make_demo_data
from stockrl.env import TradingEnv
from stockrl.evaluation import baseline_policy, evaluate_policy


def test_requested_cancel_prevents_a_trade():
    env = TradingEnv(make_demo_data(81))
    try:
        with pytest.raises(ExperimentCancelled):
            evaluate_policy(env, baseline_policy("buy_hold"),
                            control=ExecutionControl(cancel_requested=lambda: True))
        assert env.trades == []
    finally:
        env.close()


def test_evaluation_can_be_cancelled_between_transitions():
    env = TradingEnv(make_demo_data(81))
    try:
        control = ExecutionControl(cancel_requested=lambda: len(env.history) >= 4)
        with pytest.raises(ExperimentCancelled):
            evaluate_policy(env, baseline_policy("half"), control=control)
        assert len(env.history) == 4
    finally:
        env.close()


def test_throttled_progress_does_not_drop_phase_changes():
    events = []
    control = ExecutionControl(event_callback=events.append, progress_interval=3600)
    control.emit({"event": "training", "timesteps": 1}, progress=True)
    control.emit({"event": "training", "timesteps": 2}, progress=True)
    control.phase("evaluating", seed=42)
    assert [event["event"] for event in events] == ["training", "phase"]
