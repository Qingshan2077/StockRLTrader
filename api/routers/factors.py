from fastapi import APIRouter, HTTPException

from api.schemas.features import FeatureBuildRequest, FeatureSummaryResponse
from api.services.feature_service import FeatureService


router = APIRouter(prefix="/factors", tags=["factors"])


@router.post("/{ticker}/summary", response_model=FeatureSummaryResponse)
def factor_summary(ticker: str, body: FeatureBuildRequest) -> FeatureSummaryResponse:
    try:
        return FeatureSummaryResponse(
            **FeatureService().build_summary(
                ticker,
                use_cache=body.use_cache,
                horizon=body.horizon,
                label_type=body.label_type,
            )
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/{ticker}/analysis")
def factor_analysis(ticker: str, body: FeatureBuildRequest) -> dict:
    try:
        return FeatureService().analyze_single_asset(
            ticker,
            use_cache=body.use_cache,
            horizon=body.horizon,
            label_type=body.label_type,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
