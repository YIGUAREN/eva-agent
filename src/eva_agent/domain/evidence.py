"""Evidence bundle produced by input adapters."""

from pydantic import BaseModel, ConfigDict, Field

from eva_agent.domain.run import DatasetRef, InputKind


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
    manifest_sha256: str = Field(min_length=64, max_length=64)
