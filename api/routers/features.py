from fastapi import APIRouter, HTTPException

from api.schemas.features import FeatureBuildRequest, FeatureSummaryResponse
from api.services.feature_service import FeatureService


router = APIRouter(prefix="/features", tags=["features"])


@router.post("/{ticker}/build", response_model=FeatureSummaryResponse)
def build_features(ticker: str, body: FeatureBuildRequest) -> FeatureSummaryResponse:
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
