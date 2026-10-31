"""Read-only evidence access confined to one directory."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from eva_agent.tools.registry import ToolSpec


class EvidenceAccessError(ValueError):
    """Raised when a tool tries to leave the evidence root or read a file."""

    def __init__(self, message: str, *, reason: str = "unavailable") -> None:
        super().__init__(message)
        # Callers translate this instead of surfacing the path-bearing message.
        self.reason = reason


class ReadEvidenceInput(BaseModel):
    root: str = Field(min_length=1)
    path: str = Field(min_length=1)
    max_bytes: int | None = Field(default=None, gt=0)
    run_id: str | None = None


def resolve_evidence_path(root: str, relative: str) -> Path:
    root_path = Path(root).resolve()
    candidate = (root_path / relative).resolve()
    if not candidate.is_relative_to(root_path):
        raise EvidenceAccessError(f"path '{relative}' escapes evidence root", reason="invalid")
    if not candidate.is_file():
        raise EvidenceAccessError(f"path '{relative}' is not a file", reason="not_found")
    return candidate


def build_read_evidence_tool(default_max_bytes: int) -> ToolSpec:
    async def handler(payload: ReadEvidenceInput) -> str:
        path = resolve_evidence_path(payload.root, payload.path)
        limit = payload.max_bytes or default_max_bytes
        size = path.stat().st_size
        if size > limit:
            raise EvidenceAccessError(f"file '{payload.path}' exceeds {limit} bytes", reason="too_large")
        try:
            return path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise EvidenceAccessError(f"file '{payload.path}' is not valid UTF-8", reason="undecodable") from exc

    return ToolSpec(
        name="read_evidence",
        description="Read one UTF-8 evidence file; the path must stay inside the evidence root.",
        input_model=ReadEvidenceInput,
        output_model=None,
        permission="internal",
        timeout_s=10.0,
        side_effects=False,
        idempotent=True,
        handler=handler,
    )
