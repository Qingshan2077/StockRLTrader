"""Loopback application boundaries and bounded HTTP input, independent of CORS."""

from urllib.parse import urlsplit
from uuid import UUID, uuid4

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from stockrl_app.errors import AppError
from stockrl_app.settings import AppSettings


class LocalRequestMiddleware:
    def __init__(self, app: ASGIApp, settings: AppSettings):
        self.app = app
        self.settings = settings

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = scope.get("headers", [])
        values = {key.lower(): value.decode("latin-1") for key, value in headers}
        try:
            identifier = str(UUID(values.get(b"x-request-id", "")))
        except ValueError:
            identifier = str(uuid4())
        scope.setdefault("state", {})["request_id"] = identifier
        received = 0
        started = False
        upload = scope["path"] == "/api/v1/datasets/csv"
        limit = self.settings.upload_limit_bytes + 65536 if upload else 131072

        async def bounded_receive():
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > limit:
                    raise AppError("REQUEST_TOO_LARGE", "上传内容超过允许大小。", 413)
            return message

        async def response_send(message):
            nonlocal started
            if message["type"] == "http.response.start":
                started = True
                message.setdefault("headers", []).extend([
                    (b"x-request-id", identifier.encode("ascii")),
                    (b"x-content-type-options", b"nosniff"),
                    (b"referrer-policy", b"same-origin"),
                ])
            await send(message)

        try:
            host_values = [value for key, value in headers if key.lower() == b"host"]
            try:
                parsed = urlsplit("http://" + values.get(b"host", ""))
                host = parsed.hostname
                if (parsed.username is not None or parsed.password is not None or parsed.path
                        or parsed.query or parsed.fragment or (parsed.port is not None and not 1 <= parsed.port <= 65535)):
                    host = None
            except ValueError:
                host = None
            allowed = {value.strip("[]").lower() for value in self.settings.allowed_hosts}
            if len(host_values) != 1 or host not in allowed:
                raise AppError("HOST_NOT_ALLOWED", "只允许本机应用地址。", 403)
            origin = values.get(b"origin")
            if origin and origin not in self.settings.allowed_origins:
                raise AppError("ORIGIN_NOT_ALLOWED", "请求来源不在允许范围内。", 403)
            if scope["method"] in {"POST", "PUT", "PATCH"}:
                media = values.get(b"content-type", "").split(";", 1)[0].strip().lower()
                expected = "multipart/form-data" if upload else "application/json"
                if media != expected:
                    raise AppError("CONTENT_TYPE_REQUIRED", "请求内容格式不正确。", 422)
            length = values.get(b"content-length")
            if length is not None:
                try:
                    count = int(length)
                except ValueError as exc:
                    raise AppError("INVALID_REQUEST", "请求大小无效。", 422) from exc
                if count < 0:
                    raise AppError("INVALID_REQUEST", "请求大小无效。", 422)
                if count > limit:
                    raise AppError("REQUEST_TOO_LARGE", "上传内容超过允许大小。", 413)
            await self.app(scope, bounded_receive, response_send)
        except AppError as exc:
            if started:
                raise
            response = JSONResponse(exc.response(identifier), status_code=exc.status_code)
            await response(scope, receive, response_send)
