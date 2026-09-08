from fastapi import Request

from stockrl_app.datasets import DatasetService
from stockrl_app.experiments import ExperimentService
from stockrl_app.replays import ReplayService
from stockrl_app.results import ResultService
from stockrl_app.storage.database import Database
from stockrl_app.storage.jobs import JobRepository


def database(request: Request) -> Database:
    return request.app.state.database


def datasets(request: Request) -> DatasetService:
    return DatasetService(request.app.state.settings, database(request))


def experiments(request: Request) -> ExperimentService:
    return ExperimentService(request.app.state.settings, database(request))


def results(request: Request) -> ResultService:
    return ResultService(request.app.state.settings, database(request))


def replays(request: Request) -> ReplayService:
    return ReplayService(request.app.state.settings, database(request))


def jobs(request: Request) -> JobRepository:
    return JobRepository(database(request), request.app.state.settings.worker_unavailable_seconds)
