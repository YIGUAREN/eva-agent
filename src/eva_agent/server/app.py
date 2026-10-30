"""FastAPI application factory."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import cast

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from eva_agent.server.errors import ApiError
from eva_agent.server.state import AppState


def get_state(request: Request) -> AppState:
    return cast(AppState, request.app.state.eva)


async def _api_error(_request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, ApiError)
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": {"code": exc.code, "message": exc.message}},
    )


async def _http_error(_request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, StarletteHTTPException)
    code = "not_found" if exc.status_code == 404 else "http_error"
    message = "not found" if exc.status_code == 404 else "request failed"
    return JSONResponse(status_code=exc.status_code, content={"error": {"code": code, "message": message}})


async def _validation_error(_request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, RequestValidationError)
    fields = [".".join(str(part) for part in error["loc"]) for error in exc.errors()]
    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "code": "invalid_request",
                "message": "request validation failed",
                "fields": fields,
            }
        },
    )


@asynccontextmanager
async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
    yield


def create_app(state: AppState) -> FastAPI:
    app = FastAPI(
        title="Eva Agent",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=_lifespan,
    )
    app.state.eva = state
    app.add_exception_handler(ApiError, _api_error)
    app.add_exception_handler(StarletteHTTPException, _http_error)
    app.add_exception_handler(RequestValidationError, _validation_error)

    from eva_agent.server.routes.auth import router as auth_router
    from eva_agent.server.routes.health import router as health_router
    from eva_agent.server.routes.runs import router as runs_router

    app.include_router(health_router)
    app.include_router(auth_router)
    app.include_router(runs_router)
    _mount_spa(app, state.index_html)
    return app


def _mount_spa(app: FastAPI, index_html: Path) -> None:
    assets = index_html.parent / "assets"
    if assets.is_dir():
        app.mount("/assets", StaticFiles(directory=assets), name="assets")

    @app.get("/{path:path}", include_in_schema=False, response_model=None)
    async def spa(path: str) -> FileResponse | JSONResponse:
        if path == "api" or path.startswith("api/"):
            raise ApiError(404, "not_found", "not found")
        root = index_html.parent.resolve()
        candidate = (root / path).resolve()
        if path and candidate.is_file() and root in candidate.parents:
            return FileResponse(candidate)
        if index_html.is_file():
            return FileResponse(index_html)
        return JSONResponse(status_code=404, content={"error": {"code": "not_found", "message": "not found"}})
