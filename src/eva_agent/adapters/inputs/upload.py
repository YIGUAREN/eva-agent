"""Evidence workspace built from a sandbox-extracted upload."""

from __future__ import annotations

from pathlib import Path

from eva_agent.adapters.inputs.manifest import sha256_bytes, write_manifest
from eva_agent.domain.evidence import EvidenceBundle, EvidenceFile
from eva_agent.domain.run import InputKind, validate_run_id
from eva_agent.runtime.artifacts import run_paths


def write_upload_workspace(run_id: str, data_dir: str | Path) -> EvidenceBundle:
    """Index the files already copied into the run evidence directory."""
    validate_run_id(run_id)
    evidence_dir = run_paths(data_dir, run_id).evidence
    if not evidence_dir.is_dir():
        raise ValueError("upload evidence directory is missing or empty")
    root = evidence_dir.resolve()
    files: list[EvidenceFile] = []
    for path in sorted(evidence_dir.rglob("*")):
        if path.is_symlink():
            raise ValueError("upload evidence contains an unsafe path")
        if not path.is_file() or path == evidence_dir / "manifest.json":
            continue
        resolved = path.resolve()
        if root not in resolved.parents:
            raise ValueError("upload evidence contains an unsafe path")
        data = path.read_bytes()
        files.append(
            EvidenceFile(
                path=resolved.relative_to(root).as_posix(),
                size=len(data),
                sha256=sha256_bytes(data),
            )
        )
    if not files:
        raise ValueError("upload evidence is empty")
    files.sort(key=lambda item: item.path)
    digest = write_manifest(
        evidence_dir,
        kind=InputKind.UPLOAD,
        files=files,
        pasted=None,
        dataset=None,
    )
    return EvidenceBundle(
        kind=InputKind.UPLOAD,
        root=str(evidence_dir),
        files=files,
        manifest_sha256=digest,
    )
