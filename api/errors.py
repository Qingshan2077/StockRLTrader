"""Stable, non-leaking HTTP error responses."""

import logging
import sqlite3
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException
from starlette.responses import JSONResponse

from stockrl_app.errors import AppError
from stockrl.research.contracts import ResearchError

logger = logging.getLogger("stockrl.api")


def request_id(request: Request) -> str:
    return getattr(request.state, "request_id", str(uuid4()))


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(ResearchError)
    async def research_error(request: Request, exc: ResearchError):
        failure = AppError(exc.code, str(exc), 422)
        return JSONResponse(failure.response(request_id(request)), status_code=422)

    @app.exception_handler(AppError)
    async def application_error(request: Request, exc: AppError):
        return JSONResponse(exc.response(request_id(request)), status_code=exc.status_code)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError):
        # Do not echo file bodies, raw submitted values or non-JSON exceptions.
        details = [{"field": ".".join(map(str, error["loc"])), "reason": error["msg"]}
                   for error in exc.errors()]
        failure = AppError("INVALID_REQUEST", "请检查输入的参数。", 422, details)
        return JSONResponse(failure.response(request_id(request)), status_code=422)

    @app.exception_handler(HTTPException)
    async def http_error(request: Request, exc: HTTPException):
        messages = {404: "资源不存在。", 405: "此操作不支持该请求方式。"}
        failure = AppError("HTTP_ERROR", messages.get(exc.status_code, "请求无法完成。"), exc.status_code)
        return JSONResponse(failure.response(request_id(request)), status_code=exc.status_code)

    @app.exception_handler(sqlite3.Error)
    async def storage_error(request: Request, exc: sqlite3.Error):
        logger.error("Storage request %s failed", request_id(request), exc_info=exc)
        failure = AppError("STORAGE_UNAVAILABLE", "存储暂时不可用，请稍后重试。", 503)
        return JSONResponse(failure.response(request_id(request)), status_code=503)

    @app.exception_handler(Exception)
    async def unexpected_error(request: Request, exc: Exception):
        logger.error("Request %s failed", request_id(request), exc_info=exc)
        failure = AppError("INTERNAL_ERROR", "操作未完成，请使用请求编号查看本地日志。", 500)
        return JSONResponse(failure.response(request_id(request)), status_code=500)
