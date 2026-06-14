from fastapi import APIRouter

from api.core.config import get_config, get_config_dict
from api.schemas.common import HealthResponse, SystemSummary
from api.services.data_service import DataService


router = APIRouter(tags=["system"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse()


@router.get("/system/summary", response_model=SystemSummary)
def system_summary() -> SystemSummary:
    data_service = DataService()
    return SystemSummary(
        name=get_config("system.name", "StockTrader"),
        version=get_config("system.version", "3.0.0"),
        stage=get_config("stage", "single_asset"),
        data_dir=get_config("system.data_dir", "stock_data"),
        model_dir=get_config("system.model_dir", "models"),
        log_dir=get_config("system.log_dir", "logs"),
        available_tickers=len(data_service.list_tickers()),
    )


@router.get("/system/config")
def system_config() -> dict:
    return get_config_dict()
