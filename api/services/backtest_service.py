from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np

from api.services.data_service import DataService
from api.services.runtime_store import BacktestRun, runtime_store
from api.services.serialization import json_safe
from layers.backtest.backtest_engine import BacktestConfig, BacktestEngine
from layers.risk.risk_engine import RiskEngine


class BacktestService:
    def __init__(self) -> None:
        self.data_service = DataService()

    def list_runs(self) -> list[dict[str, Any]]:
        return [
            {
                "run_id": run.run_id,
                "ticker": run.ticker,
                "mode": run.mode,
                "created_at": run.created_at,
                "metrics": json_safe(run.result.get("metrics", {})),
            }
            for run in runtime_store.list_backtest_runs()
        ]

    def run(
        self,
        ticker: str,
        mode: str,
        model_run_id: str | None,
        initial_balance: float,
        commission: float,
        slippage: float,
        max_position: float,
    ) -> dict[str, Any]:
        if model_run_id:
            signal_run = runtime_store.get_signal_run(model_run_id)
            data = signal_run.test_features
            signals = signal_run.ensemble_signal
            if signals is None and signal_run.predictions:
                signals = np.mean(np.vstack(list(signal_run.predictions.values())), axis=0)
        else:
            data, _ = self.data_service.load_candles(ticker, processed=True, limit=None)
            returns = data["Close"].pct_change().fillna(0.0)
            signals = returns.rolling(5).mean().fillna(0.0).values

        if data is None or signals is None:
            raise ValueError("No data or signal available for backtest")

        config = BacktestConfig(
            initial_balance=initial_balance,
            commission=commission,
            slippage=slippage,
            max_position=max_position,
        )
        engine = BacktestEngine(config)
        risk_manager = RiskEngine() if mode in {"signal_risk", "full"} else None
        rl_executor = _ImmediateExecutor() if mode == "full" else None
        result = engine.run(mode, data, np.asarray(signals), risk_manager=risk_manager, rl_executor=rl_executor)
        safe_result = json_safe(result)

        run_id = runtime_store.new_id("bt")
        runtime_store.save_backtest_run(
            BacktestRun(
                run_id=run_id,
                ticker=ticker.upper(),
                mode=mode,
                created_at=datetime.now().isoformat(),
                result=safe_result,
            )
        )

        return {"run_id": run_id, "ticker": ticker.upper(), "mode": mode, "result": safe_result}


class _ImmediateExecutor:
    def predict_execution_ratio(self, observation):
        return 1.0
