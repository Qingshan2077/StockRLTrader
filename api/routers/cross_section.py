from fastapi import APIRouter, Query

from api.schemas.jobs import JobResponse
from api.services.job_manager import Job, job_manager
from api.services.cross_section_service import CrossSectionService


router = APIRouter(prefix="/cross-section", tags=["cross-section"])


@router.get("/universe")
def get_cross_section_universe(market: str = "a_share", limit: int = 300) -> dict:
    if market != "a_share":
        return {"market": market, "tickers": []}

    try:
        from layers.data.ashare_universe import get_universe

        universe = get_universe(size=limit)
        if hasattr(universe, "to_dict") and "code" in universe:
            tickers = universe["code"].astype(str).tolist()
        else:
            tickers = list(universe)[:limit]
    except Exception:
        tickers = []

    return {"market": market, "tickers": tickers, "count": len(tickers)}


@router.get("/local-rank")
def get_local_cross_section_rank(limit: int = 100) -> dict:
    return CrossSectionService().local_rank(limit=limit)


@router.post("/train", response_model=JobResponse)
def train_cross_section_model(
    limit: int = Query(default=100, ge=5, le=500),
    model_type: str = Query(default="lightgbm"),
    horizon: int = Query(default=5, ge=1, le=60),
    force_rebuild: bool = Query(default=False),
) -> JobResponse:
    def run(job: Job) -> dict:
        job.log("building full cross-section factor panel", 0.2)
        result = CrossSectionService().train(
            limit=limit,
            model_type=model_type,
            horizon=horizon,
            force_rebuild=force_rebuild,
        )
        job.log("cross-section model training completed", 1.0)
        return result

    job = job_manager.submit(f"cross-section:{model_type}:{limit}", run)
    return JobResponse(**job.public_dict())


@router.post("/local-backtest")
def run_local_cross_section_backtest(limit: int = 100, long_n: int = 10, short_n: int = 10) -> dict:
    return CrossSectionService().local_backtest(limit=limit, long_n=long_n, short_n=short_n)


@router.get("/factor-report")
def get_local_cross_section_factor_report(limit: int = 100) -> dict:
    return CrossSectionService().factor_report(limit=limit)
