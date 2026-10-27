"""Load CSV/JSONL datasets into canonical query/response/reference rows."""

from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass
from pathlib import Path

from eva_agent.adapters.inputs.manifest import sha256_bytes, write_manifest
from eva_agent.domain.evidence import EvidenceBundle, EvidenceFile
from eva_agent.domain.run import DatasetRef, InputKind

CANONICAL_FIELDS = ("query", "response", "reference")


class DatasetError(ValueError):
    """Raised when a dataset cannot be imported without guessing."""


@dataclass(frozen=True)
class DatasetRows:
    columns: list[str]
    rows: list[dict[str, str]]


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError as exc:
        raise DatasetError(f"{path}: dataset must be UTF-8") from exc


def _csv_rows(text: str) -> tuple[list[str], list[dict[str, str]]]:
    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames:
        raise DatasetError("CSV dataset has no header row")
    columns = [str(name) for name in reader.fieldnames]
    rows: list[dict[str, str]] = []
    for row in reader:
        if None in row:
            raise DatasetError(f"line {reader.line_num}: too many fields")
        rows.append({key: "" if value is None else value for key, value in row.items()})
    return columns, rows


def _jsonl_rows(text: str) -> tuple[list[str], list[dict[str, str]]]:
    rows: list[dict[str, str]] = []
    columns: set[str] = set()
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DatasetError(f"line {line_number}: invalid JSON: {exc.msg}") from exc
        if not isinstance(value, dict):
            raise DatasetError(f"line {line_number}: each JSONL line must be an object")
        row = {str(key): "" if item is None else str(item) for key, item in value.items()}
        rows.append(row)
        columns.update(row)
    return sorted(columns), rows


def _canonicalize(row: dict[str, str], column_map: dict[str, str], row_number: int) -> dict[str, str]:
    result: dict[str, str] = {}
    for field in CANONICAL_FIELDS:
        source = column_map.get(field, field)
        if source not in row and field == "reference":
            result[field] = ""
            continue
        if source not in row:
            raise DatasetError(f"row {row_number}: missing column '{source}'")
        result[field] = row[source].strip()
    if not result["query"]:
        raise DatasetError(f"row {row_number}: query must not be empty")
    if not result["response"]:
        raise DatasetError(f"row {row_number}: response must not be empty")
    return result


def load_dataset(ref: DatasetRef, *, max_bytes: int, max_rows: int) -> DatasetRows:
    """Read and validate a dataset. Rejects instead of truncating."""
    path = Path(ref.path)
    if not path.is_file():
        raise DatasetError(f"dataset not found: {ref.path}")
    size = path.stat().st_size
    if size > max_bytes:
        raise DatasetError(f"dataset is {size} bytes; max_bytes is {max_bytes}")

    unknown = set(ref.column_map) - set(CANONICAL_FIELDS)
    if unknown:
        raise DatasetError(f"unknown canonical field(s) in column_map: {', '.join(sorted(unknown))}")

    text = _read_text(path)
    if ref.format == "csv":
        columns, raw_rows = _csv_rows(text)
    else:
        columns, raw_rows = _jsonl_rows(text)

    if not raw_rows:
        raise DatasetError("dataset has no rows")
    if len(raw_rows) > max_rows:
        raise DatasetError(f"dataset has {len(raw_rows)} rows; max_rows is {max_rows}")

    rows = [_canonicalize(raw, ref.column_map, index) for index, raw in enumerate(raw_rows, start=1)]
    return DatasetRows(columns=columns, rows=rows)


def write_dataset_workspace(
    run_id: str,
    ref: DatasetRef,
    data_dir: str | Path,
    *,
    max_bytes: int,
    max_rows: int,
) -> EvidenceBundle:
    """Copy a dataset into the evidence workspace and validate the copy."""
    source = Path(ref.path)
    if not source.is_file():
        raise DatasetError(f"dataset not found: {ref.path}")
    if source.stat().st_size > max_bytes:
        raise DatasetError(f"dataset is {source.stat().st_size} bytes; max_bytes is {max_bytes}")

    evidence_dir = Path(data_dir) / "runs" / run_id / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    target = evidence_dir / f"dataset.{ref.format}"

    try:
        target.write_bytes(source.read_bytes())
        copied_ref = DatasetRef(path=str(target), format=ref.format, column_map=dict(ref.column_map))
        load_dataset(copied_ref, max_bytes=max_bytes, max_rows=max_rows)
        data = target.read_bytes()
        file = EvidenceFile(path=target.name, size=len(data), sha256=sha256_bytes(data))
        bundle_ref = DatasetRef(path=target.name, format=ref.format, column_map=dict(ref.column_map))
        digest = write_manifest(
            evidence_dir,
            kind=InputKind.DATASET,
            files=[file],
            pasted=None,
            dataset=bundle_ref,
        )
    except Exception:
        target.unlink(missing_ok=True)
        raise

    return EvidenceBundle(
        kind=InputKind.DATASET,
        root=str(evidence_dir),
        files=[file],
        dataset=bundle_ref,
        manifest_sha256=digest,
    )
