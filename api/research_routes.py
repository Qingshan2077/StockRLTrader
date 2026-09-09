"""Bounded research endpoints; orchestration remains in the independent worker."""
from fastapi import APIRouter, Header, Query, Request
from starlette.datastructures import UploadFile
from pydantic import BaseModel, ConfigDict

from stockrl.research.contracts import ResearchProtocolDraft, DiagnosticsReport
from stockrl_app.research_models import (MarketPreviewResponse, MarketDatasetResponse, MarketDatasetList,
    ResearchPreviewResponse, ResearchSubmission, ResearchResumeResponse, ResearchListResponse, ResearchDetailResponse)
from stockrl_app.errors import AppError
from stockrl_app.market_datasets import MarketDatasetService
from stockrl_app.researches import ResearchService

router = APIRouter(prefix='/api/v1', tags=['research'])
FIELDS = {'metadata': 'metadata.json', 'bars': 'bars.csv', 'sessions': 'sessions.csv',
          'actions': 'actions.csv', 'tradability': 'tradability.csv', 'profile': 'market-profile.json'}


def service(request: Request):
    return ResearchService(request.app.state.settings, request.app.state.database)


async def uploaded_bundle(request: Request) -> dict[str, bytes]:
    total = 0
    files = {}
    async with request.form(max_files=6, max_fields=0, max_part_size=64 * 1024 * 1024) as form:
        for field, upload in form.multi_items():
            if field not in FIELDS or field in files or not isinstance(upload, UploadFile):
                raise AppError('DATA_INVALID', '数据包只接受六个固定命名文件，不能重复字段。', 422)
            chunks, size = [], 0
            while chunk := await upload.read(1024 * 1024):
                total += len(chunk)
                size += len(chunk)
                if size > 64 * 1024 * 1024 or total > 100 * 1024 * 1024:
                    raise AppError('REQUEST_TOO_LARGE', '数据包合计上限 100 MiB，单文件上限 64 MiB。', 413)
                chunks.append(chunk)
            files[field] = b''.join(chunks)
    return {FIELDS[key]: value for key, value in files.items()}


@router.post('/market-datasets/preview', response_model=MarketPreviewResponse)
async def preview_market(request: Request):
    return MarketDatasetService(request.app.state.settings, request.app.state.database).preview(await uploaded_bundle(request))


@router.post('/market-datasets', status_code=201, response_model=MarketDatasetResponse)
async def register_market(request: Request):
    return MarketDatasetService(request.app.state.settings, request.app.state.database).register(await uploaded_bundle(request))


@router.get('/market-datasets', response_model=MarketDatasetList)
def market_datasets(request: Request):
    return MarketDatasetService(request.app.state.settings, request.app.state.database).list()


@router.post('/research-previews', response_model=ResearchPreviewResponse)
def preview_research(draft: ResearchProtocolDraft, request: Request):
    return service(request).preview(draft)


@router.post('/researches', status_code=202, response_model=ResearchSubmission)
def submit_research(draft: ResearchProtocolDraft, request: Request, idempotency_key: str = Header(alias='Idempotency-Key')):
    return service(request).submit(draft, idempotency_key)


@router.get('/researches', response_model=ResearchListResponse)
def researches(request: Request, limit: int = Query(20, ge=1, le=100), cursor: str | None = None):
    return service(request).list(limit=limit, cursor=cursor)


@router.get('/researches/{research_id}', response_model=ResearchDetailResponse)
def research(research_id: str, request: Request):
    return service(request).get(research_id)


@router.get('/researches/{research_id}/diagnostics', response_model=DiagnosticsReport)
def diagnostics(research_id: str, request: Request, instrument_id: str, fold_id: str,
                seed: int = Query(ge=0, le=4294967295)):
    return service(request).diagnostics(research_id, instrument_id, fold_id, seed)


class EmptyRequest(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)


@router.post('/researches/{research_id}/resume', status_code=202, response_model=ResearchResumeResponse)
def resume(research_id: str, body: EmptyRequest, request: Request,
           idempotency_key: str = Header(alias='Idempotency-Key')):
    return service(request).resume(research_id, idempotency_key)
