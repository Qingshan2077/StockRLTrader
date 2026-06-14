from __future__ import annotations

from typing import Any

import numpy as np

from api.services.runtime_store import runtime_store
from api.services.serialization import json_safe
from layers.evaluation.alpha_eval import AlphaEvaluator


class EvaluationService:
    def alpha_report(self, model_run_id: str, window: int = 21) -> dict[str, Any]:
        run = runtime_store.get_signal_run(model_run_id)
        if run.ensemble_signal is not None:
            signal = run.ensemble_signal
        elif run.predictions:
            signal = np.mean(np.vstack(list(run.predictions.values())), axis=0)
        else:
            raise ValueError("No signal is available for this run")
        if run.test_labels is None:
            raise ValueError("No labels are available for this run")

        evaluator = AlphaEvaluator(signal, run.test_labels)
        report = evaluator.full_report()
        report["ic_series"] = evaluator.compute_ic_series(window=window)
        return {
            "run_id": model_run_id,
            "ticker": run.ticker,
            "report": json_safe(report),
        }
