"""Evidence bundle produced by input adapters."""

from __future__ import annotations

from pathlib import PurePosixPath

from pydantic import BaseModel, ConfigDict, Field, model_validator

from eva_agent.domain.run import DatasetRef, InputKind
from eva_agent.domain.target import TargetSpec

# A manifest anywhere in the evidence workspace makes the upload a skill
# package, not only one at the root: zipped packages commonly keep SKILL.md
# under a wrapper directory. A root-only check would let that layout bypass
# skill scanning, and detection must match the package rule in
# security/skill_package.py, which treats every manifest-bearing directory as
# a package.
SKILL_MANIFEST_NAMES = ("SKILL.md", "skill.yaml", "skill.yml")


class EvidenceFile(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    path: str = Field(min_length=1)
    size: int = Field(ge=0)
    sha256: str = Field(min_length=64, max_length=64)


class EvidenceBundle(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: InputKind
    root: str | None = None
    files: list[EvidenceFile] = Field(default_factory=list)
    pasted: dict[str, str] | None = None
    dataset: DatasetRef | None = None
    target: TargetSpec | None = None
    target_prompts: list[str] = Field(default_factory=list)
    manifest_sha256: str = Field(min_length=64, max_length=64)

    @model_validator(mode="after")
    def _target_matches_kind(self) -> EvidenceBundle:
        if self.kind is InputKind.TARGET and self.target is None:
            raise ValueError("target evidence requires a target")
        if self.kind is not InputKind.TARGET and self.target is not None:
            raise ValueError("target is only valid for target evidence")
        return self

    def has_skill_manifest(self) -> bool:
        return any(PurePosixPath(item.path).name in SKILL_MANIFEST_NAMES for item in self.files)
