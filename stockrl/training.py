"""Small, reproducible SB3 learners and independently reloadable agent bundles."""
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Callable

import numpy as np
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
import torch

from stockrl.env import TradingConfig
from stockrl.features import ObservationNormalizer


def algorithm_class(algorithm):
    name = str(algorithm).upper()
    if name not in {"PPO", "SAC"}:
        raise ValueError("algorithm must be PPO or SAC")
    return {"PPO": PPO, "SAC": SAC}[name]


def load_agent(model_path, algorithm="PPO"):
    """Load a checkpoint on CPU; supply its saved normalizer to TradingEnv."""
    return algorithm_class(algorithm).load(str(Path(model_path).resolve()), device="cpu")


@dataclass
class AgentBundle:
    agent: object
    normalizer: ObservationNormalizer
    config: TradingConfig
    metadata: dict

    def policy(self, observation, env=None):
        action, _ = self.agent.predict(observation, deterministic=True)
        return float(np.asarray(action).reshape(-1)[0])


def load_bundle(model_path):
    """Load model + preprocessing + account rules from a movable seed folder."""
    path = Path(model_path).resolve()
    if path.is_dir():
        path = path / "model.zip"
    metadata = json.loads((path.parent / "training.json").read_text(encoding="utf-8"))
    return AgentBundle(load_agent(path, metadata["algorithm"]),
                       ObservationNormalizer.load(path.parent / "normalizer.json"),
                       TradingConfig(**metadata["config"]), metadata)


class ValidationCheckpoint(EvalCallback):
    """Select using fixed validation episodes, including the final updated policy.

    Standard EvalCallback may save only an untrained policy when a smoke run ends
    at its first rollout. Skip pre-optimization checks and always evaluate after
    the final optimization; validation is the sole checkpoint criterion.
    """
    def __init__(self, *args, progress_callback=None, seed=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.progress_callback = progress_callback
        self.experiment_seed = seed

    def _on_step(self):
        if self.progress_callback and self.n_calls % self.eval_freq == 0:
            self.progress_callback({"event": "training", "seed": self.experiment_seed,
                                    "timesteps": self.num_timesteps})
        if self.model._n_updates == 0:
            return True
        return super()._on_step()

    def _on_training_end(self):
        self.n_calls += (-self.n_calls) % self.eval_freq
        super()._on_step()


def train_agent(train_env_factory: Callable, validation_env_factory: Callable,
                seed_dir, algorithm, timesteps, seed, progress_callback=None):
    """Return the validation-selected policy; never receives a test environment."""
    learner = algorithm_class(algorithm)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    set_random_seed(seed, using_cuda=False)
    seed_dir = Path(seed_dir)
    train_env = Monitor(train_env_factory())
    validation_env = Monitor(validation_env_factory())
    common = dict(policy="MlpPolicy", env=train_env, seed=seed, device="cpu",
                  verbose=0, policy_kwargs={"net_arch": [32, 32]})
    if learner is PPO:
        rollout = max(2, min(128, timesteps // 4))
        batch = max(size for size in range(2, min(32, rollout) + 1) if rollout % size == 0)
        model = learner(**common, n_steps=rollout, batch_size=batch, n_epochs=4)
    else:
        model = learner(**common, buffer_size=max(1000, min(1000000, timesteps)),
                        learning_starts=min(100, timesteps // 4),
                        batch_size=max(2, min(32, timesteps // 4)),
                        train_freq=1, gradient_steps=1)
    callback = ValidationCheckpoint(validation_env, best_model_save_path=str(seed_dir),
                                    log_path=str(seed_dir), eval_freq=max(1, min(2048, timesteps // 4)),
                                    n_eval_episodes=1, deterministic=True, verbose=0,
                                    progress_callback=progress_callback, seed=seed)
    try:
        model.learn(total_timesteps=timesteps, callback=callback)
        best = seed_dir / "best_model.zip"
        if not best.exists():
            raise RuntimeError("Training did not produce a validation-selected checkpoint")
        best.replace(seed_dir / "model.zip")
        return load_agent(seed_dir / "model.zip", algorithm), {
            "actual_timesteps": int(model.num_timesteps),
            "gradient_updates": int(model._n_updates),
            "best_validation_mean_reward": float(callback.best_mean_reward),
            "checkpoint_selection": "validation_mean_reward",
            "hyperparameters": {key: getattr(model, key) for key in
                                (("n_steps", "batch_size", "n_epochs") if learner is PPO else
                                 ("learning_starts", "buffer_size", "batch_size", "gradient_steps"))},
        }
    finally:
        train_env.close()
        validation_env.close()
