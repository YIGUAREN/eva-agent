"""Sample a dataset that already lives in the evidence workspace."""

from __future__ import annotations

from pydantic import BaseModel, Field

from eva_agent.adapters.inputs.dataset import load_dataset
from eva_agent.domain.dataset import DatasetSample
from eva_agent.domain.run import DatasetRef
from eva_agent.tools.evidence import resolve_evidence_path
from eva_agent.tools.registry import ToolSpec


class SampleDatasetInput(BaseModel):
    root: str = Field(min_length=1)
    path: str = Field(min_length=1)
    limit: int = Field(gt=0)
    column_map: dict[str, str] = Field(default_factory=dict)
    run_id: str | None = None


def build_sample_dataset_tool(*, max_bytes: int, max_rows: int) -> ToolSpec:
    async def handler(payload: SampleDatasetInput) -> DatasetSample:
        path = resolve_evidence_path(payload.root, payload.path)
        suffix = path.suffix.lower()
        if suffix not in {".csv", ".jsonl"}:
            raise ValueError("dataset must be csv or jsonl")
        rows = load_dataset(
            DatasetRef(
                path=str(path),
                format="csv" if suffix == ".csv" else "jsonl",
                column_map=dict(payload.column_map),
            ),
            max_bytes=max_bytes,
            max_rows=max_rows,
        )
        return DatasetSample(columns=rows.columns, rows=rows.rows[: payload.limit])

    return ToolSpec(
        name="sample_dataset",
        description="Return the dataset columns and at most N rows from the evidence workspace.",
        input_model=SampleDatasetInput,
        output_model=DatasetSample,
        permission="internal",
        timeout_s=30.0,
        side_effects=False,
        idempotent=True,
        handler=handler,
    )
