"""
RL 训练器 — 统一 PPO/SAC 训练循环
"""
import numpy as np
from pathlib import Path
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback


class TrainingCallback(BaseCallback):
    """训练进度回调"""

    def __init__(self, update_freq: int = 100):
        super().__init__(0)
        self.update_freq = update_freq
        self.episode_rewards: list[float] = []
        self.portfolio_values: list[float] = []
        self.actions: list[float] = []
        self._ep_reward: float = 0.0

    def _on_step(self) -> bool:
        if "rewards" in self.locals and len(self.locals["rewards"]) > 0:
            self._ep_reward += float(self.locals["rewards"][0])

        if "actions" in self.locals:
            self.actions.append(float(self.locals["actions"][0]))

        if self.locals.get("dones", [False])[0]:
            self.episode_rewards.append(self._ep_reward)
            self._ep_reward = 0.0
            env = self.training_env.envs[0]
            self.portfolio_values.append(float(getattr(env, "portfolio_value", 0)))
        return True


class RLTrainer:
    """统一 RL 训练器 (PPO / SAC)"""

    def __init__(self, algorithm: str = "PPO", config: dict = None):
        self.algorithm = algorithm.upper()
        self.config = config or {}
        self.model = None
        self.callback: TrainingCallback | None = None

    def train(self, env, total_timesteps: int = 100000) -> dict:
        """训练并返回指标"""
        if self.algorithm == "PPO":
            self.model = self._create_ppo(env)
        elif self.algorithm == "SAC":
            self.model = self._create_sac(env)
        else:
            raise ValueError(f"不支持的算法: {self.algorithm}")

        self.callback = TrainingCallback(update_freq=100)
        self.model.learn(total_timesteps=total_timesteps,
                          callback=self.callback, progress_bar=False)

        return {
            "algorithm": self.algorithm,
            "total_timesteps": total_timesteps,
            "episode_rewards": self.callback.episode_rewards,
            "final_portfolio_value": (
                self.callback.portfolio_values[-1]
                if self.callback.portfolio_values else 0
            ),
            "action_mean": float(np.mean(self.callback.actions))
            if self.callback.actions else 0.0,
        }

    def _create_ppo(self, env):
        ppo_cfg = self.config.get("ppo", {})
        arch = ppo_cfg.get("policy_arch", {"pi": [128, 64], "vf": [128, 64]})
        return PPO(
            "MlpPolicy", env,
            learning_rate=ppo_cfg.get("learning_rate", 0.0003),
            n_steps=ppo_cfg.get("n_steps", 2048),
            batch_size=ppo_cfg.get("batch_size", 64),
            n_epochs=ppo_cfg.get("n_epochs", 10),
            gamma=ppo_cfg.get("gamma", 0.99),
            gae_lambda=ppo_cfg.get("gae_lambda", 0.95),
            clip_range=ppo_cfg.get("clip_range", 0.2),
            ent_coef=ppo_cfg.get("ent_coef", 0.01),
            vf_coef=ppo_cfg.get("vf_coef", 0.5),
            max_grad_norm=ppo_cfg.get("max_grad_norm", 0.5),
            policy_kwargs=dict(net_arch=[dict(pi=arch["pi"], vf=arch["vf"])]),
            verbose=0,
        )

    def _create_sac(self, env):
        sac_cfg = self.config.get("sac", {})
        arch = sac_cfg.get("policy_arch", [128, 64])
        return SAC(
            "MlpPolicy", env,
            learning_rate=sac_cfg.get("learning_rate", 0.0003),
            buffer_size=sac_cfg.get("buffer_size", 100000),
            batch_size=sac_cfg.get("batch_size", 256),
            tau=sac_cfg.get("tau", 0.005),
            gamma=sac_cfg.get("gamma", 0.99),
            ent_coef=sac_cfg.get("ent_coef", "auto"),
            policy_kwargs=dict(net_arch=dict(pi=arch, qf=arch)),
            verbose=0,
        )

    def predict(self, observation: np.ndarray) -> float:
        """单步推理"""
        if self.model is None:
            return 0.0
        action, _ = self.model.predict(observation, deterministic=True)
        return float(np.clip(action[0], 0.0, 1.0))

    def save(self, path: str) -> None:
        if self.model:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            self.model.save(str(path))

    def load(self, path: str) -> None:
        if self.algorithm == "PPO":
            self.model = PPO.load(str(path))
        elif self.algorithm == "SAC":
            self.model = SAC.load(str(path))
