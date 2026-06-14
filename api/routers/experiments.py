from fastapi import APIRouter

from api.schemas.experiments import (
    ExperimentCompareRequest,
    ExperimentCreateRequest,
    ExperimentListResponse,
    ExperimentResponse,
)
from api.services.experiment_service import ExperimentService


router = APIRouter(prefix="/experiments", tags=["experiments"])


@router.get("", response_model=ExperimentListResponse)
def list_experiments(status: str | None = None, tag: str | None = None) -> ExperimentListResponse:
    experiments = [ExperimentResponse(**exp) for exp in ExperimentService().list(status=status, tag=tag)]
    return ExperimentListResponse(experiments=experiments)


@router.post("", response_model=ExperimentResponse)
def create_experiment(body: ExperimentCreateRequest) -> ExperimentResponse:
    service = ExperimentService()
    exp_id = service.create(body.name, body.config, body.tags)
    exp = next(exp for exp in service.list() if exp["exp_id"] == exp_id)
    return ExperimentResponse(**exp)


@router.post("/compare")
def compare_experiments(body: ExperimentCompareRequest) -> list[dict]:
    return ExperimentService().compare(body.exp_ids, body.metrics)
