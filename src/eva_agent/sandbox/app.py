"""Token-protected sandbox-runner HTTP service."""

from __future__ import annotations

import asyncio
import secrets
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from eva_agent.config.settings import UploadsSettings
from eva_agent.runtime.uploads import upload_paths, validate_upload_id
from eva_agent.sandbox.extract import ExtractionError, ExtractionLimits, extract_archive


class ExtractRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    upload_id: str = Field(min_length=1, max_length=64)
    format: Literal["zip", "tar"]
    relative_path: str = Field(default="package.bin", max_length=255)


def _rejected(message: str) -> HTTPException:
    return HTTPException(status_code=400, detail={"code": "upload_rejected", "message": message})


def create_sandbox_app(*, data_dir: str, token: str, uploads: UploadsSettings) -> FastAPI:
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/extract")
    async def extract(payload: ExtractRequest, request: Request) -> dict[str, object]:
        supplied = request.headers.get("X-Sandbox-Token", "")
        if not secrets.compare_digest(supplied, token):
            raise HTTPException(status_code=401, detail={"code": "forbidden", "message": "invalid sandbox token"})
        if payload.relative_path != "package.bin":
            raise _rejected("invalid package path")
        try:
            validate_upload_id(payload.upload_id)
        except ValueError:
            raise _rejected("invalid upload id") from None
        paths = upload_paths(data_dir, payload.upload_id)
        if not paths.package.is_file():
            raise _rejected("package not found")
        limits = ExtractionLimits(
            max_total_bytes=uploads.max_extracted_bytes,
            max_files=uploads.max_files,
            max_depth=uploads.max_depth,
        )
        try:
            manifest = await asyncio.to_thread(
                extract_archive, paths.package, paths.extracted, limits=limits
            )
        except ExtractionError as exc:
            raise _rejected(exc.reason) from None
        return {"upload_id": payload.upload_id, "format": manifest["format"], "manifest": manifest}

    return app
