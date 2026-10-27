"""Normalize the run input into an evidence bundle."""

from eva_agent.adapters.inputs.paste import build_paste_evidence
from eva_agent.domain.run import InputKind, RunStatus
from eva_agent.graph.state import EvalState


async def intake_node(state: EvalState) -> dict:
    run_input = state["input"]

    if run_input.kind is not InputKind.PASTE:
        return {
            "status": RunStatus.FAILED,
            "errors": [f"阶段 0 暂不支持输入类型: {run_input.kind.value}"],
        }

    if run_input.paste is None:
        return {"status": RunStatus.FAILED, "errors": ["paste 输入缺少内容"]}

    return {"evidence": build_paste_evidence(run_input.paste), "status": RunStatus.RUNNING}
