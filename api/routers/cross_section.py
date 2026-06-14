from fastapi import APIRouter

from api.services.cross_section_service import CrossSectionService


router = APIRouter(prefix="/cross-section", tags=["cross-section"])


@router.get("/universe")
def get_cross_section_universe(market: str = "a_share", limit: int = 300) -> dict:
    if market != "a_share":
        return {"market": market, "tickers": []}

    try:
        from layers.data.ashare_universe import get_universe

        tickers = get_universe()[:limit]
    except Exception:
        tickers = []

    return {"market": market, "tickers": tickers, "count": len(tickers)}


@router.get("/local-rank")
def get_local_cross_section_rank(limit: int = 100) -> dict:
    return CrossSectionService().local_rank(limit=limit)


@router.post("/local-backtest")
def run_local_cross_section_backtest(limit: int = 100, long_n: int = 10, short_n: int = 10) -> dict:
    return CrossSectionService().local_backtest(limit=limit, long_n=long_n, short_n=short_n)


@router.get("/factor-report")
def get_local_cross_section_factor_report(limit: int = 100) -> dict:
    return CrossSectionService().factor_report(limit=limit)
