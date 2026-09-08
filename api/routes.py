from typing import Annotated, Literal
from uuid import UUID
import os
import tempfile

from fastapi import APIRouter, Depends, File, Form, Header, Query, Request, Response, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from stockrl_app.datasets import DatasetService
from stockrl_app.experiments import ExperimentService
from stockrl_app.models import (ArtifactRecord, Capabilities, Dataset, DatasetMetadata, DatasetPreview,
    DemoDatasetRequest, EventPage, ExperimentDetail, ExperimentPreview, ExperimentRequest, Integrity,
    JobDetail, JobKind, JobStatus, LocalDatasetRequest, LocalSource, Page, SubmissionResult)
from stockrl_app.replays import ReplayService
from stockrl_app.result_models import ComparisonResponse, EmptyRequest, ErrorEnvelope, Policy, SeriesResponse, TradesResponse
from stockrl_app.results import ResultService
from stockrl_app.storage.jobs import JobRepository

from . import dependencies as deps

router = APIRouter(prefix='/api/v1', responses={
    code: {'model': ErrorEnvelope, 'description': '统一错误响应'}
    for code in (403, 404, 409, 413, 422, 429, 500, 503)})
DatasetDep = Annotated[DatasetService, Depends(deps.datasets)]
ExperimentDep = Annotated[ExperimentService, Depends(deps.experiments)]
ResultDep = Annotated[ResultService, Depends(deps.results)]
ReplayDep = Annotated[ReplayService, Depends(deps.replays)]
JobDep = Annotated[JobRepository, Depends(deps.jobs)]
Limit = Annotated[int, Query(ge=1, le=100)]
Key = Annotated[str, Header(alias='Idempotency-Key', min_length=36, max_length=36)]


@router.get('/health/live', operation_id='healthLive')
def live() -> dict:
    return {'status': 'alive', 'api_version': 1}


@router.get('/health/ready', operation_id='healthReady')
def ready(request: Request, jobs: JobDep) -> Response:
    settings, db = request.app.state.settings, request.app.state.database
    components = {'database': False, 'storage': False, 'frontend': (settings.frontend_dir / 'index.html').is_file(),
                  'worker': False}
    try:
        with db.connection():
            components['database'] = True
        # A write probe is temporary and never creates or migrates an application database.
        for root in (settings.app_dir, settings.output_dir):
            with tempfile.TemporaryFile(dir=root) as probe:
                probe.write(b'ready')
                probe.flush()
                os.fsync(probe.fileno())
        components['storage'] = True
        components['worker'] = jobs.worker_available()
    except Exception as exc:
        # Detailed exception text belongs in server logs, never in the readiness response.
        import logging
        logging.getLogger(__name__).warning('Readiness unavailable: %s', type(exc).__name__)
    critical = components['database'] and components['storage']
    status = 'ready' if all(components.values()) else ('degraded' if critical else 'unavailable')
    return JSONResponse({'status': status, 'components': components}, status_code=200 if critical else 503)


@router.get('/capabilities', response_model=Capabilities, operation_id='getCapabilities')
def capabilities(service: ExperimentDep):
    return service.capabilities()


@router.get('/local-sources', response_model=list[LocalSource], operation_id='listLocalSources')
def local_sources(service: DatasetDep):
    return service.local_sources()


@router.get('/datasets', response_model=Page[Dataset], operation_id='listDatasets')
def list_datasets(service: DatasetDep, limit: Limit = 20, cursor: str | None = None):
    return service.list(limit=limit, cursor=cursor)


@router.post('/datasets/csv', response_model=Dataset, status_code=201, operation_id='uploadDataset')
def upload_dataset(service: DatasetDep, file: Annotated[UploadFile, File()],
    display_name: Annotated[str | None, Form(max_length=200)] = None,
    source_description: Annotated[str, Form(max_length=10000)] = '',
    adjustment: Annotated[str, Form(min_length=1, max_length=100)] = 'unknown',
    quote_unit: Annotated[str, Form(min_length=1, max_length=100)] = 'unknown'):
    metadata = DatasetMetadata(display_name=display_name, source_description=source_description,
                               adjustment=adjustment, quote_unit=quote_unit)
    return service.create_csv(file.file, file.filename or 'upload.csv', metadata)


@router.post('/datasets/demo', response_model=Dataset, status_code=201, operation_id='createDemoDataset')
def create_demo(body: DemoDatasetRequest, service: DatasetDep):
    return service.create_demo(body)


@router.post('/datasets/local', response_model=Dataset, status_code=201, operation_id='createLocalDataset')
def create_local(body: LocalDatasetRequest, service: DatasetDep):
    return service.create_local(body)


@router.get('/datasets/{dataset_id}', response_model=Dataset, operation_id='getDataset')
def get_dataset(dataset_id: UUID, service: DatasetDep):
    return service.get(str(dataset_id))


@router.get('/datasets/{dataset_id}/preview', response_model=DatasetPreview, operation_id='previewDataset')
def preview_dataset(dataset_id: UUID, service: DatasetDep, limit: Limit = 100):
    return service.preview(str(dataset_id), limit=limit)


@router.post('/experiment-previews', response_model=ExperimentPreview, operation_id='previewExperiment')
def preview_experiment(body: ExperimentRequest, service: ExperimentDep):
    return service.preview(body)


@router.post('/experiments', response_model=SubmissionResult, status_code=202, operation_id='submitExperiment')
def submit_experiment(body: ExperimentRequest, service: ExperimentDep, key: Key):
    return service.submit(body, key)


@router.get('/experiments', response_model=Page[ExperimentDetail], operation_id='listExperiments')
def list_experiments(service: ExperimentDep, limit: Limit = 20, cursor: str | None = None,
    kind: JobKind | None = None, integrity: Integrity | None = None, algorithm: Literal['PPO', 'SAC'] | None = None,
    dataset_id: UUID | None = None, status: JobStatus | None = None):
    return service.list(limit=limit, cursor=cursor, kind=kind, integrity=integrity, algorithm=algorithm,
                        dataset_id=str(dataset_id) if dataset_id else None, status=status)


@router.get('/experiments/{experiment_id}', response_model=ExperimentDetail, operation_id='getExperiment')
def get_experiment(experiment_id: UUID, service: ResultDep):
    return service.detail(str(experiment_id))


@router.get('/experiments/{experiment_id}/series', response_model=SeriesResponse, operation_id='getSeries')
def series(experiment_id: UUID, service: ResultDep, seed: Annotated[int, Query(ge=0, le=4294967295)],
    policy: Policy = 'rl', max_points: Annotated[int, Query(ge=100, le=5000)] = 5000):
    return service.series(str(experiment_id), seed, policy, max_points)


@router.get('/experiments/{experiment_id}/trades', response_model=TradesResponse, operation_id='getTrades')
def trades(experiment_id: UUID, service: ResultDep, seed: Annotated[int, Query(ge=0, le=4294967295)],
    policy: Policy = 'rl', limit: Limit = 20, cursor: str | None = None):
    return service.trades(str(experiment_id), seed, policy, limit=limit, cursor=cursor)


@router.get('/experiments/{experiment_id}/artifacts', response_model=list[ArtifactRecord], operation_id='listArtifacts')
def artifacts(experiment_id: UUID, service: ResultDep):
    service.experiments.get(str(experiment_id))
    return [item.public() for item in service.artifacts.list(str(experiment_id))]


@router.get('/artifacts/{artifact_id}/download', operation_id='downloadArtifact')
def download(artifact_id: UUID, service: ResultDep) -> FileResponse:
    path, record = service.download(str(artifact_id))
    return FileResponse(path, media_type='application/octet-stream', filename=record.filename.replace('/', '__'),
                        headers={'Cache-Control': 'no-store'})


@router.post('/experiments/{experiment_id}/replays', response_model=SubmissionResult, status_code=202, operation_id='submitReplay')
def replay(experiment_id: UUID, body: EmptyRequest, service: ReplayDep, key: Key):
    return service.submit(str(experiment_id), key)


@router.get('/jobs', response_model=Page[JobDetail], operation_id='listJobs')
def list_jobs(service: JobDep, limit: Limit = 20, cursor: str | None = None,
              status: JobStatus | None = None, kind: JobKind | None = None):
    return service.list_details(limit=limit, cursor=cursor, status=status, kind=kind)


@router.get('/jobs/{job_id}', response_model=JobDetail, operation_id='getJob')
def get_job(job_id: UUID, service: JobDep):
    return service.detail(str(job_id))


@router.post('/jobs/{job_id}/cancel', response_model=JobDetail, operation_id='cancelJob')
def cancel(job_id: UUID, body: EmptyRequest, response: Response, service: JobDep):
    record = service.request_cancel(str(job_id))
    response.status_code = 202 if record.status == 'cancelling' else 200
    return service.detail(str(job_id))


@router.get('/jobs/{job_id}/events', response_model=EventPage, operation_id='getJobEvents')
def events(job_id: UUID, service: JobDep, after_seq: Annotated[int, Query(ge=0)] = 0,
           limit: Annotated[int, Query(ge=1, le=500)] = 100):
    return service.events(str(job_id), after_seq=after_seq, limit=limit)


@router.get('/comparison', response_model=ComparisonResponse, operation_id='compareExperiments')
def comparison(service: ResultDep, ids: Annotated[list[UUID], Query(min_length=1, max_length=4)]):
    return service.compare([str(value) for value in ids])
