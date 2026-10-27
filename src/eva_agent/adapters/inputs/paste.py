"""Turn pasted text into an evidence bundle."""

import hashlib
import json

from eva_agent.domain.evidence import EvidenceBundle
from eva_agent.domain.run import InputKind, PastedInput


def build_paste_evidence(pasted: PastedInput) -> EvidenceBundle:
    """Build a deterministic evidence bundle from pasted text."""
    payload = pasted.model_dump(exclude_none=True)
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return EvidenceBundle(kind=InputKind.PASTE, pasted=payload, manifest_sha256=digest)
