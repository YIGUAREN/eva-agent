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
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from eva_agent.server.errors import ApiError
from eva_agent.server.state import AppState

_UPLOAD_PATH = "/api/uploads"
_MULTIPART_ALLOWANCE = 1024 * 1024


class _BodyTooLarge(BaseException):
    """Sentinel raised from ``receive`` once the upload body exceeds the cap.

    It deliberately derives from ``BaseException``: FastAPI wraps multipart
    parsing in ``except Exception`` and would otherwise turn this abort into a
    400 response before the middleware can answer with 413.
    """


class UploadSizeLimitMiddleware:
    """Reject oversized upload bodies before multipart parsing buffers them.

    FastAPI parses ``UploadFile`` parts before the endpoint runs, so a handler
    only sees the declared size after Starlette has already spooled the whole
    body. This middleware enforces the cap on the raw request: a too-large
    ``Content-Length`` is rejected up front, and streamed bodies are counted as
    chunks arrive.
    """

    def __init__(self, app: ASGIApp, *, max_bytes: int) -> None:
        self._app = app
        self._limit = max_bytes + _MULTIPART_ALLOWANCE

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope.get("method") != "POST" or scope.get("path") != _UPLOAD_PATH:
            await self._app(scope, receive, send)
            return

        declared = _content_length(scope)
        if declared is not None and declared > self._limit:
            await _too_large_response()(scope, receive, send)
            return

        received = 0
        response_started = False

        async def limited_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self._limit:
                    raise _BodyTooLarge
            return message

        async def tracking_send(message: Message) -> None:
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self._app(scope, limited_receive, tracking_send)
        except _BodyTooLarge:
            if not response_started:
                await _too_large_response()(scope, receive, send)


def _content_length(scope: Scope) -> int | None:
    for name, value in scope.get("headers", []):
        if name.lower() == b"content-length":
            try:
                return int(value)
            except ValueError:
                return None
    return None


def _too_large_response() -> JSONResponse:
    return JSONResponse(
        status_code=413,
        content={"error": {"code": "upload_rejected", "message": "上传包超过大小限制"}},
    )


def get_state(request: Request) -> AppState:
    return cast(AppState, request.app.state.eva)


async def _api_error(_request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, ApiError)
    body: dict[str, object] = {"code": exc.code, "message": exc.message}
    if exc.details is not None:
        body["details"] = exc.details
    return JSONResponse(status_code=exc.status_code, content={"error": body})


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
    app.add_middleware(UploadSizeLimitMiddleware, max_bytes=state.config.web.upload_max_bytes)

    from eva_agent.server.routes.auth import router as auth_router
    from eva_agent.server.routes.events import router as events_router
    from eva_agent.server.routes.health import router as health_router
    from eva_agent.server.routes.packages import router as packages_router
    from eva_agent.server.routes.reports import router as reports_router
    from eva_agent.server.routes.runs import router as runs_router
    from eva_agent.server.routes.uploads import router as uploads_router

    app.include_router(health_router)
    app.include_router(auth_router)
    app.include_router(runs_router)
    app.include_router(reports_router)
    app.include_router(events_router)
    app.include_router(uploads_router)
    app.include_router(packages_router)
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
