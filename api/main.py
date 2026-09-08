"""ASGI factory. Importing this module starts no worker and does not migrate storage."""
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from starlette.middleware.cors import CORSMiddleware

from stockrl_app.errors import AppError
from stockrl_app.settings import AppSettings, confined_path
from stockrl_app.storage.database import Database

from .errors import install_error_handlers
from .routes import router
from .security import LocalRequestMiddleware


def create_app(settings: AppSettings | None = None) -> FastAPI:
    settings = settings or AppSettings.from_env()
    app = FastAPI(title='StockRL Research API', version='1.0.0', docs_url='/api/docs',
                  openapi_url='/api/openapi.json', redoc_url=None)
    app.state.settings = settings
    app.state.database = Database(settings.database_path, busy_timeout_ms=settings.busy_timeout_ms)
    install_error_handlers(app)
    app.include_router(router)
    app.add_middleware(CORSMiddleware, allow_origins=list(settings.allowed_origins),
                       allow_methods=['GET', 'POST'], allow_headers=['Content-Type', 'Idempotency-Key', 'X-Request-Id'],
                       expose_headers=['X-Request-Id'], allow_credentials=False)
    app.add_middleware(LocalRequestMiddleware, settings=settings)

    @app.get('/{path:path}', include_in_schema=False)
    def frontend(path: str):
        if path == 'api' or path.startswith('api/'):
            raise HTTPException(404)
        if path:
            try:
                candidate = confined_path(settings.frontend_dir, path)
            except AppError as exc:
                raise HTTPException(404) from exc
            if candidate.is_file():
                return FileResponse(candidate)
            # Static asset misses must not receive index.html and an incorrect 200.
            if path.startswith('assets/') or '.' in path.rsplit('/', 1)[-1]:
                raise HTTPException(404)
        index = settings.frontend_dir / 'index.html'
        if not index.is_file():
            raise AppError('FRONTEND_UNAVAILABLE', '前端静态文件尚未准备，请按本地开发说明构建。', 503)
        return FileResponse(index, headers={'Cache-Control': 'no-cache'})

    return app
