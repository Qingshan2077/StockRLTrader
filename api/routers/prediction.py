from fastapi import APIRouter, HTTPException

from api.schemas.prediction import (
    AdvancedForecastRequest,
    AdvancedForecastResponse,
    ProbabilityPredictionRequest,
    ProbabilityPredictionResponse,
)
from api.services.prediction_service import PredictionService


router = APIRouter(prefix="/prediction", tags=["prediction"])


@router.post("/probability", response_model=ProbabilityPredictionResponse)
def probability_prediction(body: ProbabilityPredictionRequest) -> ProbabilityPredictionResponse:
    try:
        return ProbabilityPredictionResponse(**PredictionService().probability(body.ticker, body.horizons))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/advanced/forecast", response_model=AdvancedForecastResponse)
def advanced_forecast(body: AdvancedForecastRequest) -> AdvancedForecastResponse:
    try:
        return AdvancedForecastResponse(
            **PredictionService().advanced_forecast(
                body.ticker,
                days=body.days,
                price_horizon=body.price_horizon,
                trend_horizons=body.trend_horizons,
                method=body.method,
            )
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
