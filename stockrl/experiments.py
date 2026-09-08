"""Chronological experiment orchestration shared by the CLI and dashboard."""
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
from uuid import uuid4

import numpy as np
import pandas as pd

from stockrl.data import load_csv
from stockrl.control import ExecutionControl
from stockrl.env import TradingConfig, TradingEnv
from stockrl.evaluation import baseline_policy, evaluate_policy
from stockrl.features import ObservationNormalizer, build_features
from stockrl.training import algorithm_class, load_bundle, train_agent
from stockrl.protocol import split_intervals as split_intervals


def _json_clean(value):
    if isinstance(value, dict):
        return {str(key): _json_clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_clean(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path, value):
    Path(path).write_text(json.dumps(_json_clean(value), ensure_ascii=False,
                                    indent=2, allow_nan=False), encoding="utf-8")


def _save_evaluation(result, folder):
    folder.mkdir(parents=True, exist_ok=True)
    history = pd.DataFrame(result["history"])
    trades = pd.DataFrame(result["trades"])
    if trades.empty and not len(trades.columns):
        trades = pd.DataFrame(columns=["date", "side", "shares", "price", "cost"])
    history.to_csv(folder / "history.csv", index=False)
    trades.to_csv(folder / "trades.csv", index=False)
    return {"metrics": _json_clean(result["metrics"]),
            "history_path": str((folder / "history.csv").resolve()),
            "trades_path": str((folder / "trades.csv").resolve())}


def _aggregate(runs):
    keys = sorted({key for run in runs for key in run["metrics"]})
    result = {}
    for key in keys:
        values = [run["metrics"].get(key) for run in runs]
        values = [float(v) for v in values if isinstance(v, (int, float)) and np.isfinite(v)]
        result[key] = {"mean": float(np.mean(values)) if values else None,
                       "std": float(np.std(values)) if values else None,
                       "count": len(values)}
    return result


def _test_agent(bars, config, normalizer, split, agent, folder, control=None):
    def new_env():
        return TradingEnv(bars, config=config, normalizer=normalizer,
                          start=split["start"], end=split["end"])

    def policy(observation, env):
        action, _ = agent.predict(observation, deterministic=True)
        return float(np.asarray(action).reshape(-1)[0])

    env = new_env()
    try:
        result = _save_evaluation(evaluate_policy(env, policy, control=control), folder)
    finally:
        env.close()
    result["baselines"] = {}
    for name in ("cash", "buy_hold", "half", "trend"):
        if control:
            control.check()
        env = new_env()
        try:
            result["baselines"][name] = _save_evaluation(
                evaluate_policy(env, baseline_policy(name), control=control), folder / "baselines" / name)
        finally:
            env.close()
    return result


def run_experiment(bars, output_dir, algorithm="PPO", timesteps=10000, seeds=(42,),
                   config=None, train_ratio=.6, val_ratio=.2, data_label="user_csv",
                   episode_length=126, progress_callback=None, control: ExecutionControl | None = None):
    """Train each seed, select on validation, evaluate each selected model once.

    ``output_dir`` is an artifact parent; the returned output_dir is a unique run
    directory. Callback receives dictionaries with event, seed and timesteps.
    Smoke runs verify the software, not investment performance.
    """
    if control:
        control.phase("preparing")
    algorithm = str(algorithm).upper()
    algorithm_class(algorithm)
    if isinstance(timesteps, bool) or not isinstance(timesteps, (int, np.integer)) or timesteps < 2:
        raise ValueError("timesteps must be an integer of at least 2")
    if episode_length is not None and (isinstance(episode_length, bool) or
            not isinstance(episode_length, (int, np.integer)) or episode_length < 1):
        raise ValueError("episode_length must be a positive integer or None")
    seeds = tuple(seeds)
    if not seeds or any(isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or
                        not 0 <= seed < 2 ** 32 for seed in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must contain distinct nonnegative 32-bit integers")
    seeds = tuple(int(seed) for seed in seeds)
    config = config or TradingConfig()
    splits = split_intervals(bars, train_ratio, val_ratio)
    # Build once on original chronology; rolling features can use past context.
    # Fitting is restricted to the training prefix, including its final observation.
    features = build_features(bars)
    normalizer = ObservationNormalizer.fit(features.iloc[:splits["train"]["end"] + 1])
    # Force full input/account validation before creating artifacts.
    TradingEnv(bars, config=config, start=splits["train"]["start"],
               end=splits["train"]["end"], normalizer=normalizer).close()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid4().hex[:8]
    root = Path(output_dir).expanduser().resolve() / run_id
    root.mkdir(parents=True, exist_ok=False)
    data_path = root / "bars.csv"
    bars.to_csv(data_path, index_label="Date", float_format="%.17g")
    fingerprint = hashlib.sha256(data_path.read_bytes()).hexdigest()
    summary = {"run_id": run_id, "algorithm": algorithm, "data_label": str(data_label),
               "created_at": datetime.now(timezone.utc).isoformat(), "splits": splits,
               "runs": [], "aggregate": {}, "output_dir": str(root),
               "data_path": str(data_path), "data_sha256": fingerprint,
               "config": asdict(config), "timesteps": int(timesteps), "seeds": list(seeds),
               "episode_length": episode_length, "train_ratio": train_ratio, "val_ratio": val_ratio,
               "versions": {name: version(name) for name in
                            ("numpy", "pandas", "torch", "gymnasium", "stable-baselines3")},
               "python_version": platform.python_version(), "device": "cpu",
               "normalizer_fit": {"start": 0, "end": splits["train"]["end"]},
               "feature_names": list(features.columns),
               "notes": ["Short training is software verification only.",
                         "Checkpoints selected only by validation mean reward; no test seed selection.",
                         "Annualization uses 252 sessions; Sharpe risk-free rate is zero.",
                         "Aggregate standard deviation uses population ddof=0."]}
    for seed_index, seed in enumerate(seeds):
        if control:
            control.emit({"event": "seed_start", "seed": seed, "seed_index": seed_index,
                          "seed_count": len(seeds), "requested_steps": timesteps})
            control.phase("training", seed=seed)
        folder = root / f"seed_{seed}"
        folder.mkdir()
        normalizer.save(folder / "normalizer.json")

        def training_factory():
            split = splits["train"]
            return TradingEnv(bars, config=config, normalizer=normalizer,
                              start=split["start"], end=split["end"], random_start=True,
                              episode_length=min(episode_length, split["reward_count"])
                              if episode_length is not None else None)

        def validation_factory():
            split = splits["validation"]
            return TradingEnv(bars, config=config, normalizer=normalizer,
                              start=split["start"], end=split["end"])

        model, training = train_agent(training_factory, validation_factory, folder,
                                     algorithm, timesteps, seed, progress_callback, control=control)
        _write_json(folder / "training.json", {"algorithm": algorithm, "seed": seed,
                    "config": asdict(config), "feature_names": list(features.columns),
                    "data_sha256": fingerprint, "splits": splits, **training})
        if control:
            control.phase("evaluating", seed=seed)
        result = _test_agent(bars, config, normalizer, splits["test"], model, folder, control=control)
        result.update({"seed": seed, "model_path": str(folder / "model.zip"),
                       "normalizer_path": str(folder / "normalizer.json"),
                       "validation_path": str(folder / "evaluations.npz"), **training})
        summary["runs"].append(result)
        if control:
            control.emit({"event": "seed_complete", "seed": seed, "seed_index": seed_index,
                          "timesteps": training["actual_timesteps"]})
        if progress_callback:
            progress_callback({"event": "seed_complete", "seed": seed, "timesteps": training["actual_timesteps"]})
    if control:
        control.phase("publishing")
    summary["aggregate"] = _aggregate(summary["runs"])
    summary = _json_clean(summary)
    _write_json(root / "summary.json", summary)
    if progress_callback:
        progress_callback({"event": "complete", "run_id": run_id, "output_dir": str(root)})
    return summary


def evaluate_saved_run(run_dir, output_dir=None, *, control: ExecutionControl | None = None):
    """Replay saved test intervals without training, tuning or refitting anything.

    A run folder can be moved: models, data and normalizers are resolved relative
    to it, independently of the original absolute artifact paths in its summary.
    An explicit output directory must be fresh; existing artifacts are preserved.
    """
    if control:
        control.phase("preparing")
    root = Path(run_dir).expanduser().resolve()
    if root.name == "summary.json":
        root = root.parent
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    data_path = root / "bars.csv"
    if hashlib.sha256(data_path.read_bytes()).hexdigest() != summary["data_sha256"]:
        raise ValueError("Saved data fingerprint does not match the experiment")
    bars = load_csv(data_path)
    destination = Path(output_dir).expanduser().resolve() if output_dir else root / ("evaluation-" + uuid4().hex[:8])
    if destination == root:
        raise ValueError("Evaluation output must differ from the source run directory")
    try:
        destination.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise ValueError("Evaluation output already exists; choose a fresh destination") from exc
    runs = []
    for seed_index, saved in enumerate(summary["runs"]):
        if control:
            control.emit({"event": "seed_start", "seed": saved["seed"], "seed_index": seed_index,
                          "seed_count": len(summary["runs"])})
            control.phase("evaluating", seed=saved["seed"])
        folder = root / f"seed_{saved['seed']}"
        bundle = load_bundle(folder / "model.zip")
        result = _test_agent(bars, bundle.config, bundle.normalizer, summary["splits"]["test"],
                             bundle.agent, destination / folder.name, control=control)
        result.update({"seed": saved["seed"], "model_path": str(folder / "model.zip"),
                       "normalizer_path": str(folder / "normalizer.json"),
                       "checkpoint_selection": saved["checkpoint_selection"]})
        runs.append(result)
        if control:
            control.emit({"event": "seed_complete", "seed": saved["seed"], "seed_index": seed_index})
    if control:
        control.phase("publishing")
    replay = {**summary, "source_run_dir": str(root), "output_dir": str(destination),
              "data_path": str(data_path), "evaluation_only": True,
              "runs": runs, "aggregate": _aggregate(runs)}
    _write_json(destination / "summary.json", replay)
    return replay
