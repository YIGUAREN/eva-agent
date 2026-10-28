"""Validation for declarative grader specs (no eval/exec, fail closed)."""

from __future__ import annotations

import re
from typing import Any

from eva_agent.domain.grader import CheckSpec, GraderSpec, ScoreSpec
from eva_agent.graders_dsl.paths import PathError, parse_path

__all__ = ["CheckSpec", "DeclarativeSpecError", "ScoreSpec", "validate_declarative_spec"]

_ALLOWED_SCHEMA_KEYS = frozenset({"type", "required", "properties", "items", "enum"})
_ALLOWED_SCHEMA_TYPES = frozenset({"object", "array", "string", "number", "integer", "boolean", "null"})
_MAX_SCHEMA_DEPTH = 16


class DeclarativeSpecError(ValueError):
    """Raised when a declarative grader spec is invalid."""


def validate_declarative_spec(spec: GraderSpec) -> None:
    if spec.level != "declarative":
        raise DeclarativeSpecError("only declarative graders are supported here")
    checks = spec.checks
    if not checks:
        raise DeclarativeSpecError("declarative grader requires checks")
    if not spec.input_map:
        raise DeclarativeSpecError("declarative grader requires input_map")

    for key, path in spec.input_map.items():
        try:
            parse_path(path)
        except PathError as exc:
            raise DeclarativeSpecError(f"invalid input_map path for '{key}': {exc}") from None

    names = [check.effective_name for check in checks]
    if len(names) != len(set(names)):
        raise DeclarativeSpecError("duplicate check name")

    for check in checks:
        if check.field not in spec.input_map:
            raise DeclarativeSpecError(f"field '{check.field}' is not declared in input_map")
        if check.type == "json_schema" and check.schema is not None:
            validate_schema_subset(check.schema)
        if check.type == "regex" and check.pattern is not None:
            try:
                re.compile(check.pattern)
            except re.error as exc:
                raise DeclarativeSpecError(f"invalid regex: {exc}") from None

    if spec.score is not None:
        unknown = set(spec.score.weights) - set(names)
        if unknown:
            raise DeclarativeSpecError(f"unknown weight key: {sorted(unknown)[0]}")


def validate_schema_subset(schema: Any, *, depth: int = 0) -> None:
    if depth > _MAX_SCHEMA_DEPTH:
        raise DeclarativeSpecError("schema exceeds maximum depth")
    if not isinstance(schema, dict):
        raise DeclarativeSpecError("schema must be an object")
    unknown = set(schema) - _ALLOWED_SCHEMA_KEYS
    if unknown:
        raise DeclarativeSpecError(f"unknown schema key: {sorted(unknown)[0]}")

    if "type" in schema and schema["type"] not in _ALLOWED_SCHEMA_TYPES:
        raise DeclarativeSpecError(f"unknown schema type: {schema['type']!r}")
    if "enum" in schema and (not isinstance(schema["enum"], list) or not schema["enum"]):
        raise DeclarativeSpecError("enum must be a non-empty list")
    if "required" in schema:
        required = schema["required"]
        if not isinstance(required, list) or not all(isinstance(item, str) for item in required):
            raise DeclarativeSpecError("required must be a list of strings")
    if "properties" in schema:
        properties = schema["properties"]
        if not isinstance(properties, dict):
            raise DeclarativeSpecError("properties must be an object")
        for name, subschema in properties.items():
            if not isinstance(name, str):
                raise DeclarativeSpecError("property names must be strings")
            validate_schema_subset(subschema, depth=depth + 1)
    if "items" in schema:
        validate_schema_subset(schema["items"], depth=depth + 1)
