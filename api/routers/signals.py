from fastapi import APIRouter

from api.schemas.signals import EnsembleRequest, SignalRunListResponse, SignalRunResponse, SignalTrainRequest
from api.services.job_manager import Job, job_manager
from api.services.signal_service import SignalService


router = APIRouter(prefix="/signals", tags=["signals"])


@router.get("/runs", response_model=SignalRunListResponse)
def list_signal_runs() -> SignalRunListResponse:
    return SignalRunListResponse(runs=SignalService().list_runs())


@router.post("/train", response_model=SignalRunResponse)
def train_signals(body: SignalTrainRequest) -> SignalRunResponse:
    def run(job: Job) -> dict:
        job.log("building features", 0.15)
        result = SignalService().train(
            ticker=body.ticker,
            models=body.models,
            horizon=body.horizon,
            label_type=body.label_type,
            use_cache=body.use_cache,
        )
        job.log("signal training completed", 1.0)
        return result

    job = job_manager.submit(f"signals:{body.ticker.upper()}", run)
    return SignalRunResponse(job_id=job.job_id, status=job.status)


@router.post("/ensemble", response_model=SignalRunResponse)
def build_ensemble(body: EnsembleRequest) -> SignalRunResponse:
    result = SignalService().build_ensemble(body.method, body.model_run_ids)
    return SignalRunResponse(run_id=result["run_id"], status="completed", result=result)
