"""The four L2 check implementations."""

from __future__ import annotations

import json
import re
from typing import Any

from eva_agent.domain.grader import CheckSpec

MISSING = object()


def run_check(check: CheckSpec, value: Any) -> bool:
    if value is MISSING:
        return False
    try:
        if check.type == "json_parse":
            if not isinstance(value, str):
                return False
            json.loads(value)
            return True
        if check.type == "json_schema":
            if not isinstance(value, str) or check.schema is None:
                return False
            return matches_schema(json.loads(value), check.schema)
        if check.type == "regex":
            if check.pattern is None:
                return False
            return re.fullmatch(check.pattern, str(value)) is not None
        if check.op is None or check.value is None:
            return False
        return compare_numeric(value, check.op, check.value)
    except (json.JSONDecodeError, TypeError, ValueError):
        return False


def compare_numeric(value: Any, op: str, threshold: float) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    if op == "<":
        return value < threshold
    if op == "<=":
        return value <= threshold
    if op == ">":
        return value > threshold
    if op == ">=":
        return value >= threshold
    return value == threshold


def matches_schema(value: Any, schema: dict[str, Any]) -> bool:
    if "type" in schema and not matches_type(value, schema["type"]):
        return False
    if "enum" in schema and value not in schema["enum"]:
        return False
    if "required" in schema and (not isinstance(value, dict) or any(key not in value for key in schema["required"])):
        return False
    if "properties" in schema and isinstance(value, dict):
        for key, subschema in schema["properties"].items():
            if key in value and not matches_schema(value[key], subschema):
                return False
    if "items" in schema:
        if not isinstance(value, list):
            return False
        if not all(matches_schema(item, schema["items"]) for item in value):
            return False
    return True


def matches_type(value: Any, expected: str) -> bool:
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, list)
    if expected == "string":
        return isinstance(value, str)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "null":
        return value is None
    if isinstance(value, bool):
        return False
    if expected == "integer":
        return isinstance(value, int)
    if expected == "number":
        return isinstance(value, (int, float))
    return False
