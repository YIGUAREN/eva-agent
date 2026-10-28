"""Normalize the run input into an evidence workspace."""

from eva_agent.adapters.inputs.dataset import write_dataset_workspace
from eva_agent.adapters.inputs.paste import write_paste_workspace
from eva_agent.adapters.inputs.target import write_target_workspace
from eva_agent.adapters.target.secrets import validate_target_env
from eva_agent.domain.run import InputKind, RunStatus
from eva_agent.graph.dependencies import GraphDependencies
from eva_agent.graph.state import EvalState


async def intake_node(state: EvalState, deps: GraphDependencies) -> dict:
    run_input = state["input"]
    run_id = state["run_id"]

    try:
        if run_input.kind is InputKind.PASTE:
            if run_input.paste is None:
                raise ValueError("paste input is missing its payload")
            evidence = write_paste_workspace(run_id, run_input.paste, deps.data_dir)
        elif run_input.kind is InputKind.DATASET:
            if run_input.dataset is None:
                raise ValueError("dataset input is missing a dataset reference")
            evidence = write_dataset_workspace(
                run_id,
                run_input.dataset,
                deps.data_dir,
                max_bytes=deps.max_input_bytes,
                max_rows=deps.max_dataset_rows,
            )
        elif run_input.kind is InputKind.TARGET:
            if run_input.target_ref is None:
                raise ValueError("target input is missing a target reference")
            target = deps.targets.get(run_input.target_ref)
            if target is None:
                raise ValueError(f"unknown target '{run_input.target_ref}'")
            if deps.policy is None:
                raise ValueError("target input requires a security policy")
            validate_target_env(target, deps.env)
            validated = await deps.policy.validate(target)
            await deps.event_sink(
                run_id,
                "target.validated",
                {"target": target.name, "kind": target.kind.value, "demo": validated.demo},
            )
            evidence = write_target_workspace(
                run_id,
                target,
                run_input.target_prompts,
                deps.data_dir,
            )
        else:
            raise ValueError(f"input kind '{run_input.kind.value}' is not supported in phase 1")
    except Exception as exc:
        await deps.event_sink(run_id, "evidence.failed", {"error": str(exc)})
        return {"status": RunStatus.FAILED, "errors": [f"准备证据失败: {exc}"]}

    await deps.event_sink(
        run_id,
        "evidence.ready",
        {"kind": evidence.kind.value, "manifest_sha256": evidence.manifest_sha256},
    )
    return {"evidence": evidence, "status": RunStatus.RUNNING}
