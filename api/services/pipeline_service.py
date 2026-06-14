from __future__ import annotations

from api.core.config import get_config_dict
from api.services.backtest_service import BacktestService
from api.services.experiment_service import ExperimentService
from api.services.job_manager import Job
from api.services.signal_service import SignalService


class PipelineService:
    def run(
        self,
        job: Job,
        ticker: str,
        skip_rl: bool = True,
        walk_forward: bool = False,
        multi_model: bool = False,
        baseline_only: bool = False,
    ) -> dict:
        symbol = ticker.upper()
        models = ["lightgbm", "ridge"] if multi_model else ["lightgbm"]

        job.log("training signal models", 0.2)
        signal_result = SignalService().train(
            ticker=symbol,
            models=models,
            horizon=5,
            label_type="regression",
            use_cache=True,
        )

        job.log("building ensemble signal", 0.45)
        ensemble_result = SignalService().build_ensemble("weighted", [signal_result["run_id"]])

        modes = ["signal_only"] if baseline_only else ["signal_only", "signal_risk"]
        backtests = {}
        for index, mode in enumerate(modes):
            job.log(f"running backtest {mode}", 0.55 + index * 0.15)
            backtests[mode] = BacktestService().run(
                ticker=symbol,
                mode=mode,
                model_run_id=signal_result["run_id"],
                initial_balance=10000.0,
                commission=0.001,
                slippage=0.0005,
                max_position=1.0,
            )

        job.log("saving experiment", 0.9)
        exp_service = ExperimentService()
        exp_id = exp_service.create(
            name=f"pipeline_{symbol}",
            config={
                "ticker": symbol,
                "models": models,
                "skip_rl": skip_rl,
                "walk_forward": walk_forward,
                "multi_model": multi_model,
                "baseline_only": baseline_only,
                "source": "react-fastapi",
                "developer": "zlh",
                "system_config": get_config_dict(),
            },
            tags=[symbol, "pipeline", "react", "zlh"],
        )

        final_metrics = {}
        for mode, result in backtests.items():
            metrics = result.get("result", {}).get("metrics", {})
            for key, value in metrics.items():
                final_metrics[f"{mode}_{key}"] = value
        exp_service.manager.complete_experiment(exp_id, final_metrics=final_metrics)

        return {
            "ticker": symbol,
            "signal": signal_result,
            "ensemble": ensemble_result,
            "backtests": backtests,
            "experiment_id": exp_id,
            "rl_skipped": skip_rl,
            "walk_forward_requested": walk_forward,
        }
