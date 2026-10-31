"""Prepare grader rows from paste, dataset, upload, or target evidence."""

from __future__ import annotations

import json
from pathlib import Path

from eva_agent.domain.evidence import EvidenceBundle
from eva_agent.domain.run import InputKind
from eva_agent.tools.dataset import SampleDatasetInput
from eva_agent.tools.evidence import EvidenceAccessError, ReadEvidenceInput
from eva_agent.tools.registry import ToolRegistry
from eva_agent.tools.target import ReadTargetProbeInput

PASTE_READ_LIMIT = 1_000_000
UPLOAD_QA_READ_LIMIT = 8 * 1024 * 1024
_QA_MISSING = "upload evidence has no qa.jsonl"
_QA_TOO_LARGE = "qa.jsonl is too large"
_QA_MALFORMED = "qa.jsonl is malformed"


def _paste_row(payload: dict) -> dict[str, str]:
    return {
        "query": str(payload.get("query", "")),
        "response": str(payload.get("response") or payload.get("transcript") or ""),
        "context": str(payload.get("context") or ""),
        "reference": "",
    }


def _qa_row(line: str) -> dict[str, str]:
    try:
        payload = json.loads(line)
    except json.JSONDecodeError:
        raise ValueError(_QA_MALFORMED) from None
    if not isinstance(payload, dict):
        raise ValueError(_QA_MALFORMED)
    query = payload.get("query")
    response = payload.get("response")
    context = payload.get("context", "")
    reference = payload.get("reference", "")
    if not isinstance(query, str) or not isinstance(response, str):
        raise ValueError(_QA_MALFORMED)
    if not isinstance(context, str) or not isinstance(reference, str):
        raise ValueError(_QA_MALFORMED)
    return {"query": query, "response": response, "context": context, "reference": reference}


def parse_qa_jsonl(text: str) -> list[dict[str, str]]:
    """Parse one JSON object per non-empty line of an upload ``qa.jsonl``."""
    return [_qa_row(line) for line in text.splitlines() if line.strip()]


async def read_qa_rows(
    tools: ToolRegistry,
    evidence: EvidenceBundle,
    *,
    run_id: str | None = None,
) -> list[dict[str, str]]:
    """Read ``qa.jsonl`` through the evidence tool, hiding the tool's path errors."""
    if evidence.root is None:
        raise ValueError(_QA_MISSING)
    try:
        raw = await tools.call(
            "read_evidence",
            ReadEvidenceInput(
                root=evidence.root,
                path="qa.jsonl",
                max_bytes=UPLOAD_QA_READ_LIMIT,
                run_id=run_id,
            ),
        )
    except EvidenceAccessError as exc:
        # The tool's message embeds the evidence root; never surface it, but
        # keep the reason precise so callers can distinguish failure modes.
        if exc.reason == "too_large":
            raise ValueError(_QA_TOO_LARGE) from None
        if exc.reason == "undecodable":
            raise ValueError(_QA_MALFORMED) from None
        raise ValueError(_QA_MISSING) from None
    return parse_qa_jsonl(raw)


async def _prepare_with_indexes(
    tools: ToolRegistry,
    evidence: EvidenceBundle,
    *,
    run_id: str,
    limit: int,
) -> tuple[list[dict[str, str]], list[int | None]]:
    indexes: list[int | None]
    if evidence.kind is InputKind.DATASET:
        if evidence.dataset is None or evidence.root is None:
            raise ValueError("dataset evidence is incomplete")
        sample = await tools.call(
            "sample_dataset",
            SampleDatasetInput(
                root=evidence.root,
                path=evidence.dataset.path,
                limit=limit,
                column_map=dict(evidence.dataset.column_map),
                run_id=run_id,
            ),
        )
        rows = [dict(row) for row in sample.rows]
        indexes = [None] * len(rows)
    elif evidence.kind is InputKind.PASTE:
        if evidence.root is None:
            raise ValueError("paste evidence has no workspace")
        raw = await tools.call(
            "read_evidence",
            ReadEvidenceInput(
                root=evidence.root,
                path="paste.json",
                max_bytes=PASTE_READ_LIMIT,
                run_id=run_id,
            ),
        )
        rows = [_paste_row(json.loads(raw))]
        indexes = [None]
    elif evidence.kind is InputKind.UPLOAD:
        rows = await read_qa_rows(tools, evidence, run_id=run_id)
        # Cite the parsed-row index the shared usability filter will keep.
        indexes = list(range(len(rows)))
    elif evidence.kind is InputKind.TARGET:
        if evidence.root is None:
            raise ValueError("target evidence has no workspace")
        # target_probe.jsonl lives under the run root, next to the evidence directory.
        run_root = Path(evidence.root).parent
        probe = await tools.call("read_target_probe", ReadTargetProbeInput(root=str(run_root), run_id=run_id))
        records = [
            record
            for record in probe.records
            if record.status_code is not None and 200 <= record.status_code < 300 and record.response_text
        ]
        rows = [
            {
                "query": record.prompt,
                "response": record.response_text,
                "context": "",
                "reference": "",
            }
            for record in records
        ]
        indexes = [record.index for record in records]
    else:
        raise ValueError(f"unsupported input kind '{evidence.kind.value}'")

    usable = [
        (row, source_index)
        for row, source_index in zip(rows, indexes, strict=True)
        if row.get("query") and row.get("response")
    ][:limit]
    return [row for row, _ in usable], [source_index for _, source_index in usable]


async def prepare_rows(
    tools: ToolRegistry,
    evidence: EvidenceBundle,
    *,
    run_id: str,
    limit: int,
) -> list[dict[str, str]]:
    rows, _ = await _prepare_with_indexes(tools, evidence, run_id=run_id, limit=limit)
    return rows


async def prepare_rows_with_indexes(
    tools: ToolRegistry,
    evidence: EvidenceBundle,
    *,
    run_id: str,
    limit: int,
) -> tuple[list[dict[str, str]], list[int | None]]:
    return await _prepare_with_indexes(tools, evidence, run_id=run_id, limit=limit)
