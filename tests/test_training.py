"""Integration checks exercise real SB3 optimization and persisted experiments."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def market(n=81):
    index = pd.date_range("2023-01-02", periods=n, freq="B", name="Date")
    close = 100 * np.exp(np.arange(n) * .001 + .015 * np.sin(np.arange(n) / 3))
    opening = close * (1 + .002 * np.cos(np.arange(n)))
    return pd.DataFrame({"Open": opening, "High": np.maximum(opening, close) * 1.01,
                         "Low": np.minimum(opening, close) * .99, "Close": close,
                         "Volume": np.full(n, 100000)}, index=index)


def test_splits_partition_reward_dates_without_overlap():
    from stockrl.experiments import split_intervals
    bars = market()
    splits = split_intervals(bars, .6, .2)
    assert [(splits[k]["start"], splits[k]["end"]) for k in ("train", "validation", "test")] == [(0, 48), (48, 64), (64, 80)]
    reward_dates = [set(bars.index[s["start"] + 1:s["end"] + 1]) for s in splits.values()]
    assert len(set.union(*reward_dates)) == 80
    assert not (reward_dates[0] & reward_dates[1] or reward_dates[1] & reward_dates[2] or reward_dates[0] & reward_dates[2])


@pytest.mark.parametrize("kwargs", [{}, {"train_ratio": .9, "val_ratio": .2}])
def test_tiny_or_invalid_split_rejected_before_writing(tmp_path, kwargs):
    from stockrl.experiments import run_experiment
    with pytest.raises(ValueError):
        run_experiment(market(12), tmp_path / "unused", timesteps=8, **kwargs)
    assert not (tmp_path / "unused").exists()


@pytest.mark.parametrize("algorithm", ["PPO", "SAC"])
def test_real_training_save_reload_and_validation_selection(tmp_path, algorithm):
    from stockrl.experiments import run_experiment, evaluate_saved_run
    from stockrl.training import load_agent, load_bundle
    events = []
    result = run_experiment(market(), tmp_path, algorithm=algorithm, timesteps=16,
                            seeds=(7,), episode_length=8, data_label="synthetic_demo",
                            progress_callback=events.append)
    run = result["runs"][0]
    model = load_agent(run["model_path"], algorithm)
    # Both algorithms must perform real gradient updates, including tiny SAC runs.
    assert model._n_updates > 0
    assert run["checkpoint_selection"] == "validation_mean_reward"
    evaluations = np.load(run["validation_path"])
    assert len(evaluations["timesteps"]) >= 1
    assert run["best_validation_mean_reward"] == pytest.approx(float(evaluations["results"].max()))
    assert set(run["baselines"]) == {"cash", "buy_hold", "half", "trend"}
    assert run["baselines"]["cash"]["metrics"]["total_return"] == pytest.approx(0)
    history = pd.read_csv(run["history_path"])
    assert pd.Timestamp(history.iloc[0]["date"]) == market().index[64]
    assert pd.Timestamp(history.iloc[-1]["date"]) == market().index[-1]
    assert len(history) == 17
    bundle = load_bundle(run["model_path"])
    assert bundle.config.initial_cash == 10000
    replay = evaluate_saved_run(result["output_dir"], tmp_path / "replay")
    replay_history = pd.read_csv(replay["runs"][0]["history_path"])
    np.testing.assert_allclose(replay_history["nav"], history["nav"], rtol=0, atol=1e-8)
    assert events and events[-1]["event"] == "complete"
    metadata = json.loads((Path(result["output_dir"]) / "summary.json").read_text())
    assert len(metadata["data_sha256"]) == 64
    assert metadata["versions"]["stable-baselines3"]


def test_future_test_changes_cannot_change_training_or_normalization(tmp_path):
    from stockrl.experiments import run_experiment
    from stockrl.training import load_agent
    bars = market()
    altered = bars.copy()
    altered.loc[altered.index[65]:, ["Open", "High", "Low", "Close"]] *= 3
    a = run_experiment(bars, tmp_path / "a", timesteps=8, seeds=(13,), episode_length=8)
    b = run_experiment(altered, tmp_path / "b", timesteps=8, seeds=(13,), episode_length=8)
    ra, rb = a["runs"][0], b["runs"][0]
    assert Path(ra["normalizer_path"]).read_text() == Path(rb["normalizer_path"]).read_text()
    ma, mb = load_agent(ra["model_path"]), load_agent(rb["model_path"])
    for key, value in ma.policy.state_dict().items():
        np.testing.assert_array_equal(value.cpu().numpy(), mb.policy.state_dict()[key].cpu().numpy())
    np.testing.assert_array_equal(np.load(ra["validation_path"])["results"], np.load(rb["validation_path"])["results"])
    altered.loc[altered.index[49]:, ["Open", "High", "Low", "Close"]] *= 2
    c = run_experiment(altered, tmp_path / "c", timesteps=8, seeds=(13,), episode_length=8)
    # Validation may select another checkpoint, but cannot refit preprocessing.
    assert Path(ra["normalizer_path"]).read_text() == Path(c["runs"][0]["normalizer_path"]).read_text()


def test_all_seeds_aggregate_and_artifacts_do_not_choose_test_winner(tmp_path):
    from stockrl.experiments import run_experiment
    result = run_experiment(market(), tmp_path, timesteps=8, seeds=(2, 5, 9), episode_length=8)
    assert [run["seed"] for run in result["runs"]] == [2, 5, 9]
    returns = [run["metrics"]["total_return"] for run in result["runs"]]
    assert result["aggregate"]["total_return"]["mean"] == pytest.approx(np.mean(returns))
    assert result["aggregate"]["total_return"]["std"] == pytest.approx(np.std(returns))
    assert result["aggregate"]["total_return"]["count"] == 3
    assert len({run["model_path"] for run in result["runs"]}) == 3
    assert "best_seed" not in result


def test_replay_rejects_changed_market_snapshot(tmp_path):
    from stockrl.experiments import run_experiment, evaluate_saved_run
    result = run_experiment(market(), tmp_path, timesteps=4, seeds=(7,), episode_length=8)
    data_path = Path(result["data_path"])
    data_path.write_text(data_path.read_text().replace("100000", "200000", 1))
    with pytest.raises(ValueError, match="fingerprint"):
        evaluate_saved_run(result["output_dir"])


def test_saved_bundle_can_move_and_replay_cannot_overwrite_source(tmp_path):
    import shutil
    from stockrl.experiments import run_experiment, evaluate_saved_run
    result = run_experiment(market(), tmp_path / "original", timesteps=4, episode_length=8)
    moved = tmp_path / "moved"
    shutil.move(result["output_dir"], moved)
    original_summary = (moved / "summary.json").read_bytes()
    with pytest.raises(ValueError, match="source"):
        evaluate_saved_run(moved, moved)
    assert (moved / "summary.json").read_bytes() == original_summary
    replay = evaluate_saved_run(moved)
    assert replay["runs"][0]["metrics"] == result["runs"][0]["metrics"]


def test_ppo_budget_with_prime_rollout_above_batch_limit_trains(tmp_path):
    from stockrl.experiments import run_experiment
    from stockrl.training import load_agent
    # 148 // 4 = 37: no divisor in 2..32, previously failed before learning.
    result = run_experiment(market(), tmp_path, timesteps=148, seeds=(7,), episode_length=8)
    run = result["runs"][0]
    assert run["actual_timesteps"] >= 148
    assert run["gradient_updates"] > 0
    assert load_agent(run["model_path"])._n_updates > 0


def test_replay_rejects_another_existing_run_without_changing_artifacts(tmp_path):
    from stockrl.experiments import run_experiment, evaluate_saved_run
    source = run_experiment(market(), tmp_path / "source", timesteps=4, seeds=(7,), episode_length=8)
    other = run_experiment(market(), tmp_path / "other", timesteps=4, seeds=(7,), episode_length=8)
    destination = Path(other["output_dir"])
    before = {path.relative_to(destination): path.read_bytes()
              for path in destination.rglob("*") if path.is_file()}
    with pytest.raises(ValueError, match="exist|fresh"):
        evaluate_saved_run(source["output_dir"], destination)
    after = {path.relative_to(destination): path.read_bytes()
             for path in destination.rglob("*") if path.is_file()}
    assert after == before
