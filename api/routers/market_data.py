from fastapi import APIRouter, HTTPException, Query

from api.schemas.market_data import (
    BatchDownloadRequest,
    BatchDownloadResponse,
    CandleResponse,
    CustomNameRequest,
    DeleteTickerResponse,
    DownloadRequest,
    DownloadResponse,
    TickerInfo,
    TickerListResponse,
)
from api.services.data_service import DataService
from api.services.serialization import dataframe_records


router = APIRouter(prefix="/market", tags=["market"])


@router.get("/tickers", response_model=TickerListResponse)
def list_tickers() -> TickerListResponse:
    return TickerListResponse(tickers=DataService().list_tickers())


@router.get("/{ticker}/candles", response_model=CandleResponse)
def get_candles(
    ticker: str,
    processed: bool = Query(default=True),
    limit: int = Query(default=500, ge=1, le=5000),
) -> CandleResponse:
    try:
        df, source = DataService().load_candles(ticker, processed=processed, limit=limit)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return CandleResponse(
        ticker=ticker.upper(),
        source=source,
        row_count=len(df),
        columns=[str(c) for c in df.columns],
        records=dataframe_records(df),
    )


@router.post("/{ticker}/download", response_model=DownloadResponse)
def download_ticker(ticker: str, body: DownloadRequest) -> DownloadResponse:
    try:
        result = DataService(start_date=body.start_date).download(ticker, force_update=body.force_update)
        return DownloadResponse(**result)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/batch-download", response_model=BatchDownloadResponse)
def batch_download(body: BatchDownloadRequest) -> BatchDownloadResponse:
    results = DataService(start_date=body.start_date).batch_download(
        body.tickers,
        force_update=body.force_update,
        start_date=body.start_date,
    )
    return BatchDownloadResponse(results=[DownloadResponse(**item) for item in results])


@router.post("/{ticker}/custom-name", response_model=TickerInfo)
def set_custom_name(ticker: str, body: CustomNameRequest) -> TickerInfo:
    return DataService().set_custom_name(ticker, body.custom_name)


@router.delete("/{ticker}", response_model=DeleteTickerResponse)
def delete_ticker(ticker: str) -> DeleteTickerResponse:
    return DeleteTickerResponse(**DataService().delete(ticker))
