"""Append-only storage for target probe records."""

from __future__ import annotations

from pathlib import Path

from pydantic import ValidationError

from eva_agent.domain.probe import ProbeRecord, ProbeSummary


class ProbeStoreError(ValueError):
    """Raised when a probe file cannot be read or parsed."""


class ProbeStore:
    def __init__(self, path: Path) -> None:
        self._path = path

    def reset(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_bytes(b"")

    def append(self, record: ProbeRecord) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(record.model_dump_json())
            handle.write("\n")

    def read(self) -> list[ProbeRecord]:
        if not self._path.is_file():
            return []
        records: list[ProbeRecord] = []
        for line_number, line in enumerate(self._path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                records.append(ProbeRecord.model_validate_json(line))
            except ValidationError as exc:
                raise ProbeStoreError(
                    f"invalid probe record on line {line_number}: {exc.error_count()} error(s)"
                ) from None
        return records

    def summary(self) -> ProbeSummary:
        records = self.read()
        succeeded = sum(1 for record in records if _succeeded(record))
        return ProbeSummary(
            path=str(self._path),
            total=len(records),
            succeeded=succeeded,
            failed=len(records) - succeeded,
        )


def _succeeded(record: ProbeRecord) -> bool:
    return record.error is None and record.status_code is not None and 200 <= record.status_code < 300
