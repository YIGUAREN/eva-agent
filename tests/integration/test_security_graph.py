from dataclasses import replace
from pathlib import Path

from eva_agent.adapters.aig.errors import AigUnavailable
from eva_agent.config.settings import AigScanSettings, AigSettings, ModelConfig
from eva_agent.domain.audit import AuditEvent
from eva_agent.domain.budget import Budget
from eva_agent.domain.grader import GraderSpec
from eva_agent.domain.plan import Dimension, DimensionPlan, EvalPlan, SkippedDimension
from eva_agent.domain.result import GraderRunOutput, GraderSampleOutcome
from eva_agent.domain.run import InputKind, PastedInput, RunInput, RunStatus
from eva_agent.domain.target import TargetKind, TargetSpec
from eva_agent.graph.dependencies import GraphDependencies
from eva_agent.runtime.manager import RunManager
from eva_agent.security.scan_policy import ScanPolicy
from eva_agent.security.target_policy import DemoTarget, SecurityPolicy
from eva_agent.tools.aig import build_run_aig_scan_tool
from eva_agent.tools.evidence import build_read_evidence_tool
from eva_agent.tools.graders import RunGradersInput
from eva_agent.tools.registry import ToolRegistry, ToolSpec
from tests.support.runtime import FakePlanner
from tests.support.target_runtime import (
    build_tools as build_target_tools,
)
from tests.support.target_runtime import (
    declarative,
    runtime_config,
    target_budget,
    target_dependencies,
    target_state,
)

RAW = {
    "result": {
        "msgType": "json",
        "content": [
            {
                "modelName": "demo-echo",
                "total": 2,
                "jailbreak": 1,
                "score": 50.0,
                "extraBody": {
                    "vulnerabilityResults": [
                        {"vulnerability": "Harmful Content", "total": 2, "jailbreak": 1, "asr": 0.5}
                    ]
                },
            }
        ],
    }
}


class FakeAigClient:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.bodies: list[dict] = []

    async def create(self, body: dict) -> str:
        if self.error is not None:
            raise self.error
        self.bodies.append(body)
        return "session-1"

    async def wait(self, session_id: str, *, max_wait_s: float) -> str:
        return "completed"

    async def result(self, session_id: str) -> dict:
        return RAW

    async def upload(self, filename: str, content: bytes) -> str:
        return "uploaded.zip"


class RecordingAudit:
    def __init__(self) -> None:
        self.events: list[AuditEvent] = []

    async def record(self, event: AuditEvent) -> None:
        self.events.append(event)


def phase3_plan() -> EvalPlan:
    return EvalPlan(
        objective="阶段 3 集成",
        evidence_summary="2 条 prompt",
        dimensions=[
            DimensionPlan(
                dimension=Dimension.QUALITY,
                enabled=True,
                sample_size=2,
                graders=[GraderSpec(level="builtin", name="relevance", builtin_id="relevance")],
            ),
            DimensionPlan(
                dimension=Dimension.LATENCY,
                enabled=True,
                sample_size=2,
                graders=[declarative("latency_budget", "latency_ms", 1000)],
            ),
            DimensionPlan(
                dimension=Dimension.COST,
                enabled=True,
                sample_size=2,
                graders=[declarative("cost_budget", "cost_usd", 0.01)],
            ),
            DimensionPlan(dimension=Dimension.SECURITY, enabled=True, sample_size=2, graders=[]),
        ],
        skipped=[],
        budget=Budget(max_cost_usd=1.0, max_wall_clock_s=60.0, max_target_requests=10),
    )


def paste_plan() -> EvalPlan:
    plan = phase3_plan()
    return plan.model_copy(
        update={
            "dimensions": [
                item for item in plan.dimensions if item.dimension in {Dimension.QUALITY, Dimension.SECURITY}
            ],
            "skipped": [
                SkippedDimension(dimension=Dimension.LATENCY, reason="not applicable to paste input"),
                SkippedDimension(dimension=Dimension.COST, reason="not applicable to paste input"),
            ],
        }
    )


def phase3_tools(tmp_path: Path, audit: RecordingAudit, client: FakeAigClient) -> ToolRegistry:
    registry = build_target_tools([])
    registry.register(build_read_evidence_tool(1_000_000))

    async def graders(payload: RunGradersInput) -> GraderRunOutput:
        return GraderRunOutput(
            outcomes=[
                GraderSampleOutcome(grader=spec.name, sample_index=index, score=0.8)
                for spec in payload.specs
                for index in range(len(payload.rows))
            ]
        )

    registry.register(
        ToolSpec(
            name="run_graders",
            description="fake phase-3 graders",
            input_model=RunGradersInput,
            output_model=GraderRunOutput,
            permission="internal",
            timeout_s=10.0,
            side_effects=False,
            idempotent=False,
            handler=graders,
        )
    )
    registry.register(
        build_run_aig_scan_tool(
            client=client,
            settings=AigSettings(
                scan=AigScanSettings(datasets=["JailbreakPrompts-Tiny"], num_prompts=2, random_seed=1)
            ),
            eval_model=ModelConfig(
                provider="openai-compatible",
                model="judge",
                base_url="https://judge.example/v1",
                api_key_env="JUDGE_KEY",
            ),
            env={"JUDGE_KEY": "judge-secret"},
            audit=audit,
            data_dir=str(tmp_path / "data"),
        )
    )
    return registry


def build_dependencies(
    tmp_path: Path,
    *,
    active: bool = True,
    aig_error: Exception | None = None,
    paste: bool = False,
) -> tuple[GraphDependencies, RecordingAudit, FakeAigClient]:
    audit = RecordingAudit()
    client = FakeAigClient(aig_error)
    policy = SecurityPolicy(
        active_scan_enabled=active,
        requester_is_admin=True,
        authorization_ref="demo",
        demo_targets=[
            DemoTarget(
                name="demo",
                origin="http://127.0.0.1:8100",
                aig_base_url="http://demo-target:8100/v1",
            )
        ],
    )
    scan_policy = ScanPolicy(
        policy,
        aig=AigSettings(scan=AigScanSettings(datasets=["JailbreakPrompts-Tiny"], num_prompts=2, random_seed=1)),
        eval_model_available=True,
    )
    base = target_dependencies(tmp_path, [], [])
    deps = replace(
        base,
        planner=FakePlanner(plan=paste_plan() if paste else phase3_plan()),
        tools=phase3_tools(tmp_path, audit, client),
        catalog_ids=frozenset({"relevance"}),
        scan_policy=scan_policy,
        audit=audit,
        aig_configured=True,
        env={"JUDGE_KEY": "judge-secret"},
    )
    return deps, audit, client


async def run_target(
    tmp_path: Path,
    *,
    active: bool = True,
    aig_error: Exception | None = None,
) -> tuple[RunStatus, dict, RecordingAudit, FakeAigClient]:
    deps, audit, client = build_dependencies(tmp_path, active=active, aig_error=aig_error)
    state = target_state()
    async with RunManager.open(runtime_config(tmp_path), dependencies=deps) as manager:
        await manager.start(state["input"], state["budget"], run_id="run-3")
        view = await manager.approve("run-3", approved=True)
        report = await manager.report("run-3")
    return view.status, report, audit, client


def security_of(report: dict) -> dict:
    return next(item for item in report["dimensions"] if item["dimension"] == "security")


async def test_four_dimension_target_run_completes_the_scan(tmp_path: Path) -> None:
    status, report, audit, client = await run_target(tmp_path)
    security = security_of(report)
    assert status is RunStatus.COMPLETED
    assert security["status"] == "ok"
    assert security["score"] == 0.5
    assert security["metrics"]["active"]["status"] == "completed"
    assert security["metrics"]["active"]["severity_counts"] == {"high": 1}
    assert security["metrics"]["passive"]["by_grader"] == {"harmfulness": 0.8}
    assert [event.action for event in audit.events] == [
        "active_scan.requested",
        "active_scan.started",
        "active_scan.completed",
    ]
    assert len(client.bodies) == 1
    assert client.bodies[0]["content"]["model"][0]["base_url"] == "http://demo-target:8100/v1"


async def test_disabled_scan_is_refused_without_calling_aig(tmp_path: Path) -> None:
    status, report, audit, client = await run_target(tmp_path, active=False)
    security = security_of(report)
    assert status is RunStatus.COMPLETED
    assert security["status"] == "partial"
    assert security["metrics"]["active"]["status"] == "refused"
    assert security["metrics"]["active"]["refused_reason"] == "disabled"
    assert client.bodies == []
    assert [event.action for event in audit.events] == ["active_scan.requested"]
    assert audit.events[0].outcome == "refused"


async def test_target_outside_the_allowlist_is_refused(tmp_path: Path) -> None:
    deps, audit, client = build_dependencies(tmp_path)
    public = TargetSpec(
        name="public",
        kind=TargetKind.OPENAI_COMPATIBLE,
        provider="public",
        model="demo-echo",
        base_url="https://api.example/v1",
    )
    deps = replace(deps, targets={"public": public})
    run_input = RunInput(kind=InputKind.TARGET, target_ref="public", target_prompts=["a", "b"])
    async with RunManager.open(runtime_config(tmp_path), dependencies=deps) as manager:
        await manager.start(run_input, target_budget(), run_id="run-3")
        view = await manager.approve("run-3", approved=True)
        report = await manager.report("run-3")
    security = security_of(report)
    assert view.status is RunStatus.COMPLETED
    assert security["status"] == "partial"
    assert security["metrics"]["active"]["refused_reason"] == "target_not_allowlisted"
    assert client.bodies == []
    assert audit.events[-1].outcome == "refused"


async def test_insufficient_scan_budget_is_refused(tmp_path: Path) -> None:
    deps, _audit, client = build_dependencies(tmp_path)
    state = target_state()
    state["budget"] = Budget(max_cost_usd=1.0, max_wall_clock_s=60.0, max_target_requests=2)
    async with RunManager.open(runtime_config(tmp_path), dependencies=deps) as manager:
        await manager.start(state["input"], state["budget"], run_id="run-3")
        view = await manager.approve("run-3", approved=True)
        report = await manager.report("run-3")
    security = security_of(report)
    assert view.status is RunStatus.COMPLETED
    assert security["status"] == "partial"
    assert security["metrics"]["active"]["refused_reason"] == "budget_exceeded"
    assert client.bodies == []


async def test_unavailable_aig_degrades_to_passive_and_completes(tmp_path: Path) -> None:
    status, report, audit, _ = await run_target(tmp_path, aig_error=AigUnavailable("down"))
    security = security_of(report)
    assert status is RunStatus.COMPLETED
    assert security["status"] == "partial"
    assert security["metrics"]["active"]["status"] == "failed"
    assert security["metrics"]["active"]["degraded"] is True
    assert [event.action for event in audit.events] == [
        "active_scan.requested",
        "active_scan.failed",
    ]


async def test_paste_security_runs_passive_checks_without_a_scan(tmp_path: Path) -> None:
    deps, audit, client = build_dependencies(tmp_path, paste=True)
    run_input = RunInput(kind=InputKind.PASTE, paste=PastedInput(query="q", response="a"))
    async with RunManager.open(runtime_config(tmp_path), dependencies=deps) as manager:
        await manager.start(run_input, target_budget(), run_id="run-3")
        view = await manager.approve("run-3", approved=True)
        report = await manager.report("run-3")
    security = security_of(report)
    assert view.status is RunStatus.COMPLETED
    assert security["status"] == "ok"
    assert security["metrics"]["active"]["status"] == "not_applicable"
    assert client.bodies == []
    assert audit.events == []
