"""Skill-package discovery and threat-grader row preparation.

A skill package is any directory in the evidence workspace that contains a
``SKILL.md``, ``skill.yaml``, or ``skill.yml``. The installed
``SkillThreatAnalysisGrader`` consumes one package per row: the manifest, the
instruction body, and the contents of separate script and reference files.
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath
from typing import Any

from eva_agent.domain.evidence import SKILL_MANIFEST_NAMES, EvidenceBundle
from eva_agent.domain.grader import GraderSpec
from eva_agent.tools.evidence import EvidenceAccessError, ReadEvidenceInput
from eva_agent.tools.registry import ToolRegistry

GRADER_ID = "threat_analysis"
GRADER_NAME = "threat_analysis"
SKILL_FILE_READ_LIMIT = 128 * 1024
MAX_PACKAGE_CHARS = 200_000
MAX_PACKAGES = 50
MAX_SCRIPTS = 20
MAX_REFERENCES = 20
_SCRIPT_SUFFIXES = frozenset({".py", ".sh", ".js", ".mjs", ".ts", ".ps1", ".bat", ".rb", ".pl"})
_MANIFEST_PRIORITY = ("skill.yaml", "skill.yml", "SKILL.md")


def skill_spec() -> GraderSpec:
    return GraderSpec(level="builtin", name=GRADER_NAME, builtin_id=GRADER_ID)


def _package_roots(paths: list[str]) -> list[str]:
    return sorted(
        {
            PurePosixPath(path).parent.as_posix()
            for path in paths
            if PurePosixPath(path).name in SKILL_MANIFEST_NAMES
        },
        key=lambda item: (len(PurePosixPath(item).parts), item),
    )


def _owner(path: str, roots: frozenset[str]) -> str | None:
    """Return the deepest package root that contains ``path``.

    Walking the path's ancestors against a precomputed root set keeps ownership
    linear in path depth instead of rescanning every root for every path.
    """
    parent = PurePosixPath(path).parent
    while True:
        key = parent.as_posix()
        if key in roots:
            return key
        if parent == PurePosixPath("."):
            return None
        parent = parent.parent


def _relative(path: str, root: str) -> str:
    return path if root == "." else path[len(root) + 1 :]


def _package_name(root: str, *, evidence_root: str | None) -> str:
    if root != ".":
        return PurePosixPath(root).name
    # The evidence root itself is the package directory; its basename is the
    # only stable name available for it.
    return (Path(evidence_root).name if evidence_root else "") or "uploaded-skill"


def _is_script(relative: str) -> bool:
    path = PurePosixPath(relative)
    return path.suffix.lower() in _SCRIPT_SUFFIXES or (bool(path.parts) and path.parts[0] == "scripts")


async def _read(
    tools: ToolRegistry,
    evidence: EvidenceBundle,
    path: str,
    *,
    run_id: str,
) -> str | None:
    if not evidence.root:
        # No readable root: fail the dimension rather than the whole run.
        return None
    try:
        return await tools.call(
            "read_evidence",
            ReadEvidenceInput(
                root=evidence.root,
                path=path,
                max_bytes=SKILL_FILE_READ_LIMIT,
                run_id=run_id,
            ),
        )
    except EvidenceAccessError as exc:
        if exc.reason != "too_large":
            # Binary or unreadable members carry no graded text; skip them.
            return None
    try:
        # The member was cut at the read limit, so a split multi-byte tail is
        # expected; only this truncated case decodes leniently.
        return await tools.call(
            "read_evidence",
            ReadEvidenceInput(
                root=evidence.root,
                path=path,
                max_bytes=SKILL_FILE_READ_LIMIT,
                truncate=True,
                run_id=run_id,
            ),
        )
    except EvidenceAccessError:
        # Binary or unreadable members carry no graded text; skip them.
        return None


async def build_skill_rows(
    tools: ToolRegistry,
    evidence: EvidenceBundle,
    *,
    run_id: str,
) -> list[dict[str, Any]]:
    """Return one package-level row per skill package in the evidence workspace."""
    if not evidence.root:
        # No readable root: fail the dimension rather than the whole run.
        return []
    paths = [item.path for item in evidence.files]
    roots = _package_roots(paths)
    if len(roots) > MAX_PACKAGES:
        raise EvidenceAccessError(
            f"too many skill packages: {len(roots)} exceeds MAX_PACKAGES={MAX_PACKAGES}",
            reason="too_many_packages",
        )
    root_set = frozenset(roots)
    owned: dict[str, list[str]] = {root: [] for root in roots}
    for path in paths:
        owner = _owner(path, root_set)
        if owner is not None:
            owned[owner].append(path)
    rows: list[dict[str, Any]] = []
    for root in roots:
        members = sorted(owned[root])
        manifest_path: str | None = None
        for candidate in _MANIFEST_PRIORITY:
            manifest_path = next(
                (path for path in members if _relative(path, root) == candidate),
                None,
            )
            if manifest_path is not None:
                break
        instruction_path = next(
            (path for path in members if _relative(path, root) == "SKILL.md"),
            None,
        )
        manifest = await _read(tools, evidence, manifest_path, run_id=run_id) if manifest_path else ""
        instruction = await _read(tools, evidence, instruction_path, run_id=run_id) if instruction_path else ""
        scripts: list[str] = []
        references: list[str] = []
        budget = MAX_PACKAGE_CHARS
        for path in members:
            if path in {manifest_path, instruction_path}:
                continue
            relative = _relative(path, root)
            if relative in SKILL_MANIFEST_NAMES:
                # Only one manifest is graded; a second root-level variant is
                # not a reference document.
                continue
            if _is_script(relative):
                if len(scripts) >= MAX_SCRIPTS:
                    continue
                target = scripts
            else:
                if len(references) >= MAX_REFERENCES:
                    continue
                target = references
            if budget <= 0:
                break
            text = await _read(tools, evidence, path, run_id=run_id)
            if text is None:
                continue
            text = text[:budget]
            budget -= len(text)
            target.append(text)
        rows.append(
            {
                "skill_name": _package_name(root, evidence_root=evidence.root),
                "skill_manifest": manifest or "",
                "instruction_body": instruction or "",
                "script_contents": scripts,
                "reference_contents": references,
            }
        )
    return rows
