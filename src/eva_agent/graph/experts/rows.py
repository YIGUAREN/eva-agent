"""Prepare grader rows from paste, dataset, or target evidence."""

from __future__ import annotations

import json
from pathlib import Path

from eva_agent.domain.evidence import EvidenceBundle
from eva_agent.domain.run import InputKind
from eva_agent.tools.dataset import SampleDatasetInput
from eva_agent.tools.evidence import ReadEvidenceInput
from eva_agent.tools.registry import ToolRegistry
from eva_agent.tools.target import ReadTargetProbeInput

PASTE_READ_LIMIT = 1_000_000


def _paste_row(payload: dict) -> dict[str, str]:
    return {
        "query": str(payload.get("query", "")),
        "response": str(payload.get("response") or payload.get("transcript") or ""),
        "context": str(payload.get("context") or ""),
        "reference": "",
    }


async def prepare_rows(
    tools: ToolRegistry,
    evidence: EvidenceBundle,
    *,
    run_id: str,
    limit: int,
) -> list[dict[str, str]]:
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
    elif evidence.kind is InputKind.TARGET:
        if evidence.root is None:
            raise ValueError("target evidence has no workspace")
        # target_probe.jsonl lives under the run root, next to the evidence directory.
        run_root = Path(evidence.root).parent
        probe = await tools.call("read_target_probe", ReadTargetProbeInput(root=str(run_root), run_id=run_id))
        rows = [
            {
                "query": record.prompt,
                "response": record.response_text,
                "context": "",
                "reference": "",
            }
            for record in probe.records
            if record.status_code is not None
            and 200 <= record.status_code < 300
            and record.response_text
        ]
    else:
        raise ValueError(f"unsupported input kind '{evidence.kind.value}'")

    usable = [row for row in rows if row.get("query") and row.get("response")]
    return usable[:limit]
