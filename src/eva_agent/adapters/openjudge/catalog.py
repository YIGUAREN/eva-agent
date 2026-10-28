"""Load the grader catalog; catalog entries are the single source of truth."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator


class CatalogError(ValueError):
    """Raised when the grader catalog is missing or invalid."""


class GraderEntry(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    id: str = Field(min_length=1)
    klass: str = Field(alias="class", min_length=1)
    mode: Literal["pointwise", "listwise"]
    inputs: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    needs_model: bool = True
    score_min: float
    score_max: float
    cost: str = "low"

    @model_validator(mode="after")
    def _score_range_is_valid(self) -> GraderEntry:
        if self.score_max <= self.score_min:
            raise ValueError(f"score_max ({self.score_max}) must be greater than score_min ({self.score_min})")
        return self


def load_catalog(path: Path) -> dict[str, GraderEntry]:
    if not path.is_file():
        raise CatalogError(f"grader catalog not found: {path}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise CatalogError(f"invalid YAML in {path}: {exc}") from exc
    if not isinstance(raw, dict) or not isinstance(raw.get("graders"), list):
        raise CatalogError(f"catalog root must contain a 'graders' list: {path}")

    entries: dict[str, GraderEntry] = {}
    for item in raw["graders"]:
        try:
            entry = GraderEntry.model_validate(item)
        except ValidationError as exc:
            raise CatalogError(f"invalid grader entry: {exc.errors(include_input=False)}") from None
        if entry.id in entries:
            raise CatalogError(f"duplicate grader id: {entry.id}")
        entries[entry.id] = entry
    return entries


def load_class(path: str) -> type:
    module_name, _, class_name = path.rpartition(".")
    if not module_name or not class_name:
        raise ImportError(f"invalid class path: {path}")
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def summarize(entries: dict[str, GraderEntry]) -> list[dict[str, Any]]:
    return [
        {
            "id": entry.id,
            "mode": entry.mode,
            "inputs": entry.inputs,
            "tags": entry.tags,
            "needs_model": entry.needs_model,
            "cost": entry.cost,
        }
        for entry in entries.values()
    ]
