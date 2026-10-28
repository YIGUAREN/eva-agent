"""Write pasted text into an evidence workspace."""

from __future__ import annotations

from pathlib import Path

from eva_agent.adapters.inputs.manifest import canonical_json, sha256_bytes, write_manifest
from eva_agent.domain.evidence import EvidenceBundle, EvidenceFile
from eva_agent.domain.run import InputKind, PastedInput, validate_run_id


def write_paste_workspace(run_id: str, pasted: PastedInput, data_dir: str | Path) -> EvidenceBundle:
    validate_run_id(run_id)
    evidence_dir = Path(data_dir) / "runs" / run_id / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)

    payload = pasted.model_dump(exclude_none=True)
    data = canonical_json(payload)
    target = evidence_dir / "paste.json"
    target.write_bytes(data)

    file = EvidenceFile(path=target.name, size=len(data), sha256=sha256_bytes(data))
    digest = write_manifest(
        evidence_dir,
        kind=InputKind.PASTE,
        files=[file],
        pasted=payload,
        dataset=None,
    )
    return EvidenceBundle(
        kind=InputKind.PASTE,
        root=str(evidence_dir),
        files=[file],
        pasted=payload,
        manifest_sha256=digest,
    )
