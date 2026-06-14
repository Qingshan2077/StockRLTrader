from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from api.core.config import get_config, get_config_dict
from api.services.data_service import DataService
from api.services.job_manager import Job
from api.services.runtime_store import runtime_store
from api.services.serialization import json_safe
from layers.env.execution_env import ExecutionEnv
from layers.rl.trainer import RLTrainer


@dataclass
class RLConfig:
    initial_balance: float = 10000.0
    commission: float = 0.001
    slippage: float = 0.0005
    drawdown_threshold: float = 0.2
    lambda_cost: float = 0.1
    lambda_turnover: float = 0.05
    lambda_drawdown: float = 0.1


class RLService:
    def train(
        self,
        job: Job,
        ticker: str,
        algorithm: str,
        timesteps: int,
        pnl_weight: float,
        cost_weight: float,
        turnover_weight: float,
        drawdown_weight: float,
        signal_run_id: str | None = None,
    ) -> dict[str, Any]:
        symbol = ticker.upper()
        job.log("preparing RL execution environment", 0.15)

        if signal_run_id:
            signal_run = runtime_store.get_signal_run(signal_run_id)
            data = signal_run.test_features
            signals = signal_run.ensemble_signal
        else:
            data, _ = DataService().load_candles(symbol, processed=True, limit=None)
            signals = data["Close"].pct_change().rolling(5).mean().fillna(0.0).values

        if data is None or signals is None:
            raise ValueError("No data or signal available for RL training")

        prices = data["Close"].values.astype(np.float64)
        volumes = data["Volume"].values.astype(np.float64) if "Volume" in data.columns else np.ones(len(data)) * 1e6
        signals = np.asarray(signals, dtype=np.float64)[: len(prices)]
        target_positions = np.clip(signals * 10.0, -1.0, 1.0)
        volatilities = self._rolling_volatility(prices)

        cfg = RLConfig(
            initial_balance=float(get_config("backtest.initial_balance", 10000.0)),
            commission=float(get_config("backtest.commission", 0.001)),
            slippage=float(get_config("backtest.slippage", 0.0005)),
            lambda_cost=cost_weight,
            lambda_turnover=turnover_weight,
            lambda_drawdown=drawdown_weight,
        )

        env = ExecutionEnv(
            signal_scores=signals,
            target_positions=target_positions,
            prices=prices,
            volumes=volumes,
            volatilities=volatilities,
            config=cfg,
            window_size=int(get_config("rl.env.window_size", 5)),
        )

        job.log("training RL model", 0.35)
        trainer = RLTrainer(algorithm=algorithm, config=get_config_dict().get("rl", {}))
        metrics = trainer.train(env, total_timesteps=timesteps)

        model_dir = Path(get_config("system.model_dir", "models")) / "rl"
        model_path = model_dir / f"{symbol}_{algorithm.upper()}_{job.job_id}.zip"
        trainer.save(str(model_path))

        job.log("RL training completed", 1.0)
        return {
            "ticker": symbol,
            "algorithm": algorithm.upper(),
            "timesteps": timesteps,
            "model_path": str(model_path),
            "weights": {
                "pnl": pnl_weight,
                "cost": cost_weight,
                "turnover": turnover_weight,
                "drawdown": drawdown_weight,
            },
            "metrics": json_safe(metrics),
        }

    @staticmethod
    def _rolling_volatility(prices: np.ndarray, window: int = 20) -> np.ndarray:
        returns = np.diff(prices) / np.maximum(prices[:-1], 1e-12)
        vol = np.full(len(prices), 0.2, dtype=np.float64)
        for i in range(window, len(prices)):
            vol[i] = np.std(returns[i - window:i]) * np.sqrt(252)
        return np.nan_to_num(vol, nan=0.2)
