"""Target adapter factory."""

from eva_agent.adapters.target.base import TargetAdapter
from eva_agent.adapters.target.http_json import HttpJsonTargetAdapter
from eva_agent.adapters.target.openai import OpenAITargetAdapter
from eva_agent.domain.target import TargetKind, TargetSpec


def build_adapter(target: TargetSpec) -> TargetAdapter:
    if target.kind is TargetKind.OPENAI_COMPATIBLE:
        return OpenAITargetAdapter()
    return HttpJsonTargetAdapter()
