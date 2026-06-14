from fastapi import APIRouter

from api.schemas.rl import RLRunResponse, RLTrainRequest
from api.services.job_manager import Job, job_manager
from api.services.rl_service import RLService


router = APIRouter(prefix="/rl", tags=["rl"])


@router.post("/train", response_model=RLRunResponse)
def train_rl(body: RLTrainRequest) -> RLRunResponse:
    def run(job: Job) -> dict:
        return RLService().train(
            job=job,
            ticker=body.ticker,
            algorithm=body.algorithm,
            timesteps=body.timesteps,
            pnl_weight=body.pnl_weight,
            cost_weight=body.cost_weight,
            turnover_weight=body.turnover_weight,
            drawdown_weight=body.drawdown_weight,
            signal_run_id=body.signal_run_id,
        )

    job = job_manager.submit(f"rl:{body.ticker.upper()}:{body.algorithm}", run)
    return RLRunResponse(job_id=job.job_id, status=job.status)
