from fastapi import APIRouter, HTTPException

from api.schemas.jobs import JobListResponse, JobResponse, JobResultResponse, PipelineJobRequest
from api.services.job_manager import Job, job_manager
from api.services.pipeline_service import PipelineService


router = APIRouter(prefix="/jobs", tags=["jobs"])


def _placeholder_pipeline(body: PipelineJobRequest):
    def run(job: Job) -> dict:
        return PipelineService().run(
            job=job,
            ticker=body.ticker,
            skip_rl=body.skip_rl,
            walk_forward=body.walk_forward,
            multi_model=body.multi_model,
            baseline_only=body.baseline_only,
        )

    return run


@router.get("", response_model=JobListResponse)
def list_jobs() -> JobListResponse:
    return JobListResponse(jobs=[JobResponse(**job.public_dict()) for job in job_manager.list()])


@router.post("/pipeline", response_model=JobResponse)
def create_pipeline_job(body: PipelineJobRequest) -> JobResponse:
    job = job_manager.submit(f"pipeline:{body.ticker.upper()}", _placeholder_pipeline(body))
    return JobResponse(**job.public_dict())


@router.get("/{job_id}", response_model=JobResponse)
def get_job(job_id: str) -> JobResponse:
    try:
        return JobResponse(**job_manager.get(job_id).public_dict())
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}") from exc


@router.get("/{job_id}/result", response_model=JobResultResponse)
def get_job_result(job_id: str) -> JobResultResponse:
    try:
        job = job_manager.get(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}") from exc
    return JobResultResponse(job_id=job.job_id, status=job.status, result=job.result)


@router.get("/{job_id}/logs")
def get_job_logs(job_id: str) -> dict:
    try:
        job = job_manager.get(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}") from exc
    return {"job_id": job.job_id, "logs": job.logs}


@router.delete("/{job_id}", response_model=JobResponse)
def cancel_job(job_id: str) -> JobResponse:
    try:
        return JobResponse(**job_manager.cancel(job_id).public_dict())
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}") from exc
