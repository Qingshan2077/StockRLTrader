from fastapi import APIRouter

from api.schemas.backtest import BacktestRunListResponse, BacktestRunRequest, BacktestRunResponse
from api.services.backtest_service import BacktestService


router = APIRouter(prefix="/backtest", tags=["backtest"])


@router.get("/runs", response_model=BacktestRunListResponse)
def list_backtest_runs() -> BacktestRunListResponse:
    return BacktestRunListResponse(runs=BacktestService().list_runs())


@router.post("/run", response_model=BacktestRunResponse)
def run_backtest(body: BacktestRunRequest) -> BacktestRunResponse:
    result = BacktestService().run(
        ticker=body.ticker,
        mode=body.mode,
        model_run_id=body.model_run_id,
        initial_balance=body.initial_balance,
        commission=body.commission,
        slippage=body.slippage,
        max_position=body.max_position,
    )
    return BacktestRunResponse(run_id=result["run_id"], status="completed", result=result)
