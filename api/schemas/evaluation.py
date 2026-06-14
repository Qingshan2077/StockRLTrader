from typing import Any

from pydantic import BaseModel


class AlphaEvaluationRequest(BaseModel):
    model_run_id: str
    window: int = 21


class AlphaEvaluationResponse(BaseModel):
    run_id: str
    ticker: str
    report: dict[str, Any]
