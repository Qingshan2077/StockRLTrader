from fastapi import APIRouter, HTTPException

from api.schemas.strategy import MacdBacktestRequest, MacdBacktestResponse
from api.services.strategy_service import StrategyService


router = APIRouter(prefix="/strategy", tags=["strategy"])


@router.post("/macd/backtest", response_model=MacdBacktestResponse)
def macd_backtest(body: MacdBacktestRequest) -> MacdBacktestResponse:
    try:
        return MacdBacktestResponse(
            **StrategyService().macd_backtest(
                body.ticker,
                initial_balance=body.initial_balance,
                commission=body.commission,
                smooth_threshold=body.smooth_threshold,
            )
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
