from fastapi import APIRouter, HTTPException

from api.schemas.evaluation import AlphaEvaluationRequest, AlphaEvaluationResponse
from api.services.evaluation_service import EvaluationService


router = APIRouter(prefix="/evaluation", tags=["evaluation"])


@router.post("/alpha", response_model=AlphaEvaluationResponse)
def alpha_evaluation(body: AlphaEvaluationRequest) -> AlphaEvaluationResponse:
    try:
        return AlphaEvaluationResponse(**EvaluationService().alpha_report(body.model_run_id, window=body.window))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Signal run not found: {body.model_run_id}") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
