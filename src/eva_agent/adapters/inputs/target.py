"""Write a target definition and its prompts into an evidence workspace."""

from __future__ import annotations

from pathlib import Path

from eva_agent.adapters.inputs.manifest import canonical_json, sha256_bytes, write_manifest
from eva_agent.domain.evidence import EvidenceBundle, EvidenceFile
from eva_agent.domain.run import InputKind, validate_run_id
from eva_agent.domain.target import TargetSpec


def write_target_workspace(
    run_id: str,
    target: TargetSpec,
    prompts: list[str],
    data_dir: str | Path,
) -> EvidenceBundle:
    validate_run_id(run_id)
    evidence_dir = Path(data_dir) / "runs" / run_id / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)

    payload = {"target": target.model_dump(mode="json"), "prompts": list(prompts)}
    data = canonical_json(payload)
    path = evidence_dir / "target.json"
    path.write_bytes(data)

    file = EvidenceFile(path=path.name, size=len(data), sha256=sha256_bytes(data))
    digest = write_manifest(
        evidence_dir,
        kind=InputKind.TARGET,
        files=[file],
        pasted=None,
        dataset=None,
        target=target,
        target_prompts=list(prompts),
    )
    return EvidenceBundle(
        kind=InputKind.TARGET,
        root=str(evidence_dir),
        files=[file],
        target=target,
        target_prompts=list(prompts),
        manifest_sha256=digest,
    )
