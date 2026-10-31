from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from eva_agent.adapters.aig.models import AigAgentReport, AigReport, AigSkillReport
from eva_agent.config.settings import AigScanSettings, AigSettings
from eva_agent.domain.budget import Budget
from eva_agent.domain.evidence import EvidenceBundle, EvidenceFile
from eva_agent.domain.plan import Dimension, DimensionPlan
from eva_agent.domain.probe import ProbeRecord
from eva_agent.domain.result import GraderRunOutput, GraderSampleOutcome
from eva_agent.domain.run import InputKind
from eva_agent.domain.target import TargetKind, TargetSpec
from eva_agent.graph.dependencies import GraphDependencies
from eva_agent.graph.events import EventSink, noop_event_sink
from eva_agent.graph.experts.security import SecurityExpert
from eva_agent.security.scan_policy import ScanDecision, ScanPolicy
from eva_agent.security.target_policy import SecurityPolicy
from eva_agent.tools.aig import (
    AigAgentScanInput,
    AigAgentScanOutput,
    AigScanInput,
    AigScanOutput,
    AigSkillScanInput,
    AigSkillScanOutput,
)
from eva_agent.tools.evidence import build_read_evidence_tool
from eva_agent.tools.graders import RunGradersInput
from eva_agent.tools.registry import ToolRegistry, ToolSpec
from eva_agent.tools.target import ReadTargetProbeInput, ReadTargetProbeOutput
from tests.support.runtime import FakePlanner

_DIGEST = "0" * 64


class RecordingPolicy(ScanPolicy):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.calls: list[dict] = []

    def decide(self, **kwargs: Any) -> ScanDecision:
        self.calls.append(kwargs)
        return super().decide(**kwargs)


class FakeScanTool:
    def __init__(self, output) -> None:
        self.output = output
        self.calls: list[Any] = []
        self.spec = None

    async def __call__(self, payload):
        self.calls.append(payload)
        return self.output


class RecordingAudit:
    def __init__(self) -> None:
        self.events = []

    async def record(self, event) -> None:
        self.events.append(event)


def register(
    registry: ToolRegistry,
    name: str,
    input_model: type[BaseModel],
    output_model: type[BaseModel],
    handler: Callable[[Any], Awaitable[Any]],
) -> None:
    registry.register(
        ToolSpec(
            name=name,
            description=f"fake {name}",
            input_model=input_model,
            output_model=output_model,
            permission="admin",
            timeout_s=10.0,
            side_effects=True,
            idempotent=False,
            handler=handler,
        )
    )


def register_graders(registry: ToolRegistry, *, fail_on: str | None = None) -> None:
    async def graders(payload: RunGradersInput) -> GraderRunOutput:
        names = {spec.name for spec in payload.specs}
        if fail_on == "threat_analysis" and "threat_analysis" in names:
            raise RuntimeError("threat grader failed")
        if fail_on == "passive" and names - {"threat_analysis"}:
            raise RuntimeError("passive grader failed")
        return GraderRunOutput(
            outcomes=[
                GraderSampleOutcome(
                    grader=spec.name,
                    sample_index=index,
                    score=0.0 if spec.name == "threat_analysis" else 0.8,
                )
                for spec in payload.specs
                for index in range(len(payload.rows))
            ]
        )

    registry.register(
        ToolSpec(
            name="run_graders",
            description="fake graders",
            input_model=RunGradersInput,
            output_model=GraderRunOutput,
            permission="internal",
            timeout_s=10.0,
            side_effects=False,
            idempotent=False,
            handler=graders,
        )
    )


def register_target_probe(registry: ToolRegistry) -> None:
    async def probe(_payload: ReadTargetProbeInput) -> ReadTargetProbeOutput:
        return ReadTargetProbeOutput(
            records=[
                ProbeRecord(
                    index=0,
                    prompt="q",
                    started_at=datetime.now(UTC),
                    status_code=200,
                    latency_ms=5,
                    response_text="safe answer",
                )
            ]
        )

    registry.register(
        ToolSpec(
            name="read_target_probe",
            description="fake target probe",
            input_model=ReadTargetProbeInput,
            output_model=ReadTargetProbeOutput,
            permission="public_read",
            timeout_s=10.0,
            side_effects=False,
            idempotent=True,
            handler=probe,
        )
    )


def security_plan() -> DimensionPlan:
    return DimensionPlan(dimension=Dimension.SECURITY, enabled=True, sample_size=2)


def security_budget() -> Budget:
    return Budget(max_cost_usd=1.0, max_wall_clock_s=60.0, max_target_requests=10)


def scan_policy() -> RecordingPolicy:
    return RecordingPolicy(
        SecurityPolicy(
            active_scan_enabled=True,
            requester_is_admin=False,
            authorization_ref="policy-static-ref",
            demo_targets=[],
            allowlist=["http://127.0.0.1:8100"],
        ),
        aig=AigSettings(
            scan=AigScanSettings(datasets=["d"], num_prompts=2, random_seed=1),
        ),
        eval_model_available=True,
    )


def dependencies(
    tmp_path: Path,
    *,
    tools: ToolRegistry,
    policy: ScanPolicy,
    audit: RecordingAudit | None = None,
    aig_configured: bool = True,
    event_sink: EventSink = noop_event_sink,
) -> GraphDependencies:
    return GraphDependencies(
        planner=FakePlanner(),
        tools=tools,
        catalog_ids=frozenset(),
        data_dir=str(tmp_path / "data"),
        max_input_bytes=1_000_000,
        max_dataset_rows=100,
        default_sample_size=5,
        scan_policy=policy,
        audit=audit if audit is not None else RecordingAudit(),
        aig_configured=aig_configured,
        event_sink=event_sink,
    )


def agent_target() -> TargetSpec:
    return TargetSpec(
        name="agent",
        kind=TargetKind.AGENT_HTTP,
        provider="demo",
        model="agent",
        url="http://127.0.0.1:8100/run",
        method="POST",
        request_template={"prompt": "{{prompt}}"},
        response_path="$.output",
    )


def model_target() -> TargetSpec:
    return TargetSpec(
        name="demo",
        kind=TargetKind.OPENAI_COMPATIBLE,
        provider="demo",
        model="demo-echo",
        base_url="http://127.0.0.1:8100/v1",
    )


def target_evidence(tmp_path: Path, target: TargetSpec) -> EvidenceBundle:
    return EvidenceBundle(
        kind=InputKind.TARGET,
        root=str(tmp_path / "evidence"),
        target=target,
        target_prompts=["q"],
        manifest_sha256=_DIGEST,
    )


def upload_evidence(root: Path, *paths: str) -> EvidenceBundle:
    return EvidenceBundle(
        kind=InputKind.UPLOAD,
        root=str(root),
        files=[EvidenceFile(path=path, size=(root / path).stat().st_size, sha256=_DIGEST) for path in paths],
        manifest_sha256=_DIGEST,
    )


def agent_output(*, score: int = 40) -> AigAgentScanOutput:
    report = AigAgentReport.model_validate(
        {
            "schema_version": "agent-security-report@1",
            "agent_name": "agent",
            "score": score,
            "risk_type": "high",
            "total_tests": 4,
            "vulnerable_tests": 1,
            "results": [
                {
                    "id": "f-1",
                    "type": "tool_abuse",
                    "title": "Tool abuse",
                    "description": "untrusted prose",
                    "level": "HIGH",
                    "owasp": ["ASI02"],
                    "suggestion": "untrusted prose",
                }
            ],
            "owasp_agentic_2026_top10": [],
        }
    )
    return AigAgentScanOutput(
        session_id="sess-agent",
        report=report,
        raw_result_path="/runs/run-1/aig/sess-agent.json",
    )


def zero_test_agent_output() -> AigAgentScanOutput:
    report = AigAgentReport.model_validate(
        {
            "schema_version": "agent-security-report@1",
            "agent_name": "agent",
            "score": 100,
            "risk_type": "safe",
            "total_tests": 0,
            "vulnerable_tests": 0,
            "results": [],
            "owasp_agentic_2026_top10": [],
        }
    )
    return AigAgentScanOutput(
        session_id="sess-agent",
        report=report,
        raw_result_path="/runs/run-1/aig/sess-agent.json",
    )


def skill_output() -> AigSkillScanOutput:
    report = AigSkillReport.model_validate(
        {
            "score": 30,
            "language": "Markdown",
            "readme": "untrusted prose",
            "results": [
                {
                    "title": "Suspicious script",
                    "description": "untrusted prose",
                    "risk_type": "T04: Embedded Malicious Code",
                    "level": "High",
                    "suggestion": "untrusted prose",
                    "file": "scripts/run.py",
                    "line_start": 1,
                }
            ],
        }
    )
    return AigSkillScanOutput(
        session_id="sess-skill",
        report=report,
        raw_result_path="/runs/run-1/aig/sess-skill.json",
    )


def model_output() -> AigScanOutput:
    report = AigReport.model_validate(
        {
            "modelName": "demo-echo",
            "total": 2,
            "jailbreak": 1,
            "score": 50.0,
            "extraBody": {
                "vulnerabilityResults": [{"vulnerability": "Harmful Content", "total": 2, "jailbreak": 1, "asr": 0.5}]
            },
        }
    )
    return AigScanOutput(
        session_id="sess-model",
        reports=[report],
        datasets=["d"],
        num_prompts=2,
        raw_result_path="/runs/run-1/aig/sess-model.json",
    )


async def test_agent_target_runs_agent_scan_with_the_run_authorization(tmp_path: Path) -> None:
    tools = ToolRegistry()
    tools.register(build_read_evidence_tool(1_000_000))
    agent_tool = FakeScanTool(agent_output())
    model_tool = FakeScanTool(model_output())
    register(tools, "run_aig_agent_scan", AigAgentScanInput, AigAgentScanOutput, agent_tool)
    register(tools, "run_aig_scan", AigScanInput, AigScanOutput, model_tool)
    policy = scan_policy()
    audit = RecordingAudit()
    deps = dependencies(tmp_path, tools=tools, policy=policy, audit=audit)

    result = await SecurityExpert(deps).run(
        run_id="run-1",
        plan=security_plan(),
        evidence=target_evidence(tmp_path, agent_target()),
        probe=None,
        budget=security_budget(),
        authorization_ref="ticket-42",
        requester_is_admin=True,
    )

    assert result.status == "ok"
    assert result.metrics["active"]["kind"] == "agent_scan"
    call = policy.calls[0]
    assert call["scan_kind"] == "agent_scan"
    assert call["requester_is_admin"] is True
    assert call["authorization_ref"] == "ticket-42"
    assert call["probe_count"] == 0
    assert model_tool.calls == []
    assert agent_tool.calls[0].authorization_ref == "ticket-42"
    requested = next(event for event in audit.events if event.action == "active_scan.requested")
    assert requested.scan_kind == "agent_scan"
    assert requested.authorization_ref == "ticket-42"


async def test_agent_scan_high_finding_caps_the_score(tmp_path: Path) -> None:
    tools = ToolRegistry()
    tools.register(build_read_evidence_tool(1_000_000))
    agent_tool = FakeScanTool(agent_output(score=95))
    register(tools, "run_aig_agent_scan", AigAgentScanInput, AigAgentScanOutput, agent_tool)
    deps = dependencies(tmp_path, tools=tools, policy=scan_policy())

    result = await SecurityExpert(deps).run(
        run_id="run-1",
        plan=security_plan(),
        evidence=target_evidence(tmp_path, agent_target()),
        probe=None,
        budget=security_budget(),
        authorization_ref="ticket-42",
        requester_is_admin=True,
    )

    assert result.metrics["active"]["severity_counts"] == {"high": 1}
    assert result.metrics["score_cap"] == 0.5
    assert result.score == 0.5
    assert result.status == "ok"


async def test_zero_test_agent_scan_does_not_end_ok(tmp_path: Path) -> None:
    tools = ToolRegistry()
    tools.register(build_read_evidence_tool(1_000_000))
    agent_tool = FakeScanTool(zero_test_agent_output())
    register(tools, "run_aig_agent_scan", AigAgentScanInput, AigAgentScanOutput, agent_tool)
    deps = dependencies(tmp_path, tools=tools, policy=scan_policy())

    result = await SecurityExpert(deps).run(
        run_id="run-1",
        plan=security_plan(),
        evidence=target_evidence(tmp_path, agent_target()),
        probe=None,
        budget=security_budget(),
        authorization_ref="ticket-42",
        requester_is_admin=True,
    )

    assert result.status == "failed"
    assert result.score is None
    assert result.confidence == 0.0
    assert result.error == "no usable security score"


def skill_upload(tmp_path: Path, *, qa: str | None = None) -> tuple[Path, EvidenceBundle]:
    root = tmp_path / "evidence"
    root.mkdir()
    (root / "SKILL.md").write_text("# Greeter\n\nSay hello.\n", encoding="utf-8")
    (root / "scripts").mkdir()
    (root / "scripts" / "run.py").write_text("print('hi')\n", encoding="utf-8")
    paths = ["SKILL.md", "scripts/run.py"]
    if qa is not None:
        (root / "qa.jsonl").write_text(qa, encoding="utf-8")
        paths.append("qa.jsonl")
    return root, upload_evidence(root, *paths)


def wrapped_skill_upload(tmp_path: Path) -> tuple[Path, EvidenceBundle]:
    """A package zipped with a wrapper directory, alongside a root qa.jsonl."""
    root = tmp_path / "evidence"
    package = root / "ssh-collector"
    (package / "scripts").mkdir(parents=True)
    (package / "SKILL.md").write_text("# Collector\n\nCollect facts.\n", encoding="utf-8")
    (package / "scripts" / "run.py").write_text("print('hi')\n", encoding="utf-8")
    (root / "qa.jsonl").write_text('{"query": "q", "response": "safe answer"}\n', encoding="utf-8")
    paths = ["ssh-collector/SKILL.md", "ssh-collector/scripts/run.py", "qa.jsonl"]
    return root, upload_evidence(root, *paths)


def skill_tools(*, fail_on: str | None = None) -> tuple[ToolRegistry, FakeScanTool]:
    tools = ToolRegistry()
    tools.register(build_read_evidence_tool(1_000_000))
    register_graders(tools, fail_on=fail_on)
    skill_tool = FakeScanTool(skill_output())
    register(tools, "run_aig_skill_scan", AigSkillScanInput, AigSkillScanOutput, skill_tool)
    return tools, skill_tool


async def test_skill_upload_runs_threat_analysis_and_a_skill_scan(tmp_path: Path) -> None:
    _, bundle = skill_upload(tmp_path)

    tools, skill_tool = skill_tools()
    policy = scan_policy()
    audit = RecordingAudit()
    deps = dependencies(tmp_path, tools=tools, policy=policy, audit=audit)

    result = await SecurityExpert(deps).run(
        run_id="run-1",
        plan=security_plan(),
        evidence=bundle,
        probe=None,
        budget=security_budget(),
        authorization_ref="ticket-42",
        requester_is_admin=True,
    )

    assert result.metrics["passive"]["by_grader"]["threat_analysis"] == 0.0
    assert result.metrics["active"]["kind"] == "skill_scan"
    assert policy.calls[0]["scan_kind"] == "skill_scan"
    assert skill_tool.calls[0].files == [item.path for item in bundle.files]
    requested = next(event for event in audit.events if event.action == "active_scan.requested")
    assert requested.scan_kind == "skill_scan"
    assert requested.authorization_ref == "ticket-42"


async def test_wrapped_skill_upload_takes_the_skill_scan_path(tmp_path: Path) -> None:
    _, bundle = wrapped_skill_upload(tmp_path)
    tools, skill_tool = skill_tools()
    policy = scan_policy()
    deps = dependencies(tmp_path, tools=tools, policy=policy)
    expert = SecurityExpert(deps)

    assert expert._scan_kind(bundle) == "skill_scan"

    result = await expert.run(
        run_id="run-1",
        plan=security_plan(),
        evidence=bundle,
        probe=None,
        budget=security_budget(),
        authorization_ref="ticket-42",
        requester_is_admin=True,
    )

    # The active skill scan must run, not just the passive qa grading.
    assert policy.calls[0]["scan_kind"] == "skill_scan"
    assert result.metrics["active"]["kind"] == "skill_scan"
    assert len(skill_tool.calls) == 1


async def test_skill_upload_with_malformed_qa_records_the_failure(tmp_path: Path) -> None:
    _, bundle = skill_upload(tmp_path, qa="{not json}\n")
    tools, _ = skill_tools()
    deps = dependencies(tmp_path, tools=tools, policy=scan_policy())

    result = await SecurityExpert(deps).run(
        run_id="run-1",
        plan=security_plan(),
        evidence=bundle,
        probe=None,
        budget=security_budget(),
        authorization_ref="ticket-42",
        requester_is_admin=True,
    )

    assert result.status == "partial"
    assert result.metrics["passive"]["errors"] == ["ValueError"]
    assert result.metrics["active"]["kind"] == "skill_scan"


async def test_threat_grading_failure_is_recorded_on_a_skill_upload(tmp_path: Path) -> None:
    _, bundle = skill_upload(tmp_path, qa='{"query": "q", "response": "safe answer"}\n')
    tools, _ = skill_tools(fail_on="threat_analysis")
    deps = dependencies(tmp_path, tools=tools, policy=scan_policy())

    result = await SecurityExpert(deps).run(
        run_id="run-1",
        plan=security_plan(),
        evidence=bundle,
        probe=None,
        budget=security_budget(),
        authorization_ref="ticket-42",
        requester_is_admin=True,
    )

    passive = result.metrics["passive"]
    assert result.status == "partial"
    assert passive["errors"] == ["RuntimeError"]
    assert passive["expected_outcomes"] > passive["scored_outcomes"]


async def test_passive_grading_failure_is_recorded_on_a_skill_upload(tmp_path: Path) -> None:
    _, bundle = skill_upload(tmp_path, qa='{"query": "q", "response": "safe answer"}\n')
    tools, _ = skill_tools(fail_on="passive")
    deps = dependencies(tmp_path, tools=tools, policy=scan_policy())

    result = await SecurityExpert(deps).run(
        run_id="run-1",
        plan=security_plan(),
        evidence=bundle,
        probe=None,
        budget=security_budget(),
        authorization_ref="ticket-42",
        requester_is_admin=True,
    )

    passive = result.metrics["passive"]
    assert result.status == "partial"
    assert passive["errors"] == ["RuntimeError"]
    assert passive["by_grader"] == {"threat_analysis": 0.0}
    assert passive["expected_outcomes"] > passive["scored_outcomes"]


async def test_too_many_packages_does_not_erase_the_dimension(tmp_path: Path) -> None:
    files = [EvidenceFile(path="SKILL.md", size=1, sha256=_DIGEST)]
    files.extend(EvidenceFile(path=f"pkg-{index}/SKILL.md", size=1, sha256=_DIGEST) for index in range(50))
    bundle = EvidenceBundle(
        kind=InputKind.UPLOAD,
        root=str(tmp_path / "evidence"),
        files=files,
        manifest_sha256=_DIGEST,
    )
    tools, skill_tool = skill_tools()
    deps = dependencies(tmp_path, tools=tools, policy=scan_policy())

    result = await SecurityExpert(deps).run(
        run_id="run-1",
        plan=security_plan(),
        evidence=bundle,
        probe=None,
        budget=security_budget(),
        authorization_ref="ticket-42",
        requester_is_admin=True,
    )

    assert result.status != "failed"
    assert result.score is not None
    assert result.metrics["active"]["kind"] == "skill_scan"
    assert result.metrics["passive"]["errors"] == ["too_many_packages"]
    assert len(skill_tool.calls) == 1


async def test_threat_unavailable_caps_the_score_and_names_the_reason(tmp_path: Path) -> None:
    files = [EvidenceFile(path="SKILL.md", size=1, sha256=_DIGEST)]
    files.extend(EvidenceFile(path=f"pkg-{index}/SKILL.md", size=1, sha256=_DIGEST) for index in range(50))
    bundle = EvidenceBundle(
        kind=InputKind.UPLOAD,
        root=str(tmp_path / "evidence"),
        files=files,
        manifest_sha256=_DIGEST,
    )
    tools = ToolRegistry()
    tools.register(build_read_evidence_tool(1_000_000))
    register_graders(tools)
    clean = AigSkillScanOutput(
        session_id="sess-skill",
        report=AigSkillReport.model_validate({"score": 90, "language": "Markdown", "readme": "", "results": []}),
        raw_result_path="/runs/run-1/aig/sess-skill.json",
    )
    skill_tool = FakeScanTool(clean)
    register(tools, "run_aig_skill_scan", AigSkillScanInput, AigSkillScanOutput, skill_tool)
    events: list[tuple[str, dict[str, Any] | None]] = []

    async def sink(run_id: str, event_type: str, payload: dict[str, Any] | None = None) -> None:
        events.append((event_type, payload))

    deps = dependencies(tmp_path, tools=tools, policy=scan_policy(), event_sink=sink)

    result = await SecurityExpert(deps).run(
        run_id="run-1",
        plan=security_plan(),
        evidence=bundle,
        probe=None,
        budget=security_budget(),
        authorization_ref="ticket-42",
        requester_is_admin=True,
    )

    # A package the threat grader never saw cannot score better than "high".
    assert result.score == 0.5
    assert result.metrics["score_cap"] == 0.5
    assert result.status == "partial"
    assert "too_many_packages" in result.summary
    skipped = [payload for event_type, payload in events if event_type == "security.threat.skipped"]
    assert skipped == [{"reason": "too_many_packages"}]


async def test_threat_unavailable_names_the_reason_even_without_a_score(tmp_path: Path) -> None:
    files = [EvidenceFile(path="SKILL.md", size=1, sha256=_DIGEST)]
    files.extend(EvidenceFile(path=f"pkg-{index}/SKILL.md", size=1, sha256=_DIGEST) for index in range(50))
    bundle = EvidenceBundle(
        kind=InputKind.UPLOAD,
        root=str(tmp_path / "evidence"),
        files=files,
        manifest_sha256=_DIGEST,
    )
    tools, _ = skill_tools()
    deps = dependencies(tmp_path, tools=tools, policy=scan_policy(), aig_configured=False)

    result = await SecurityExpert(deps).run(
        run_id="run-1",
        plan=security_plan(),
        evidence=bundle,
        probe=None,
        budget=security_budget(),
        authorization_ref="ticket-42",
        requester_is_admin=True,
    )

    assert result.status == "failed"
    assert result.score is None
    # A cap that was never applied must not be reported as if it had been.
    assert result.metrics["score_cap"] is None
    assert "too_many_packages" in result.summary


async def test_a_threat_pass_without_a_score_is_capped(tmp_path: Path) -> None:
    _, bundle = skill_upload(tmp_path, qa='{"query": "q", "response": "safe answer"}\n')
    tools, _ = skill_tools(fail_on="threat_analysis")
    deps = dependencies(tmp_path, tools=tools, policy=scan_policy())

    result = await SecurityExpert(deps).run(
        run_id="run-1",
        plan=security_plan(),
        evidence=bundle,
        probe=None,
        budget=security_budget(),
        authorization_ref="ticket-42",
        requester_is_admin=True,
    )

    # Passive graded 0.8, but the package's own content was never assessed.
    assert result.status == "partial"
    assert result.score == 0.5
    assert result.metrics["score_cap"] == 0.5
    assert "threat_not_scored" in result.summary


async def test_not_configured_audit_row_carries_the_run_authorization(tmp_path: Path) -> None:
    tools = ToolRegistry()
    tools.register(build_read_evidence_tool(1_000_000))
    audit = RecordingAudit()
    deps = dependencies(tmp_path, tools=tools, policy=scan_policy(), audit=audit, aig_configured=False)

    await SecurityExpert(deps).run(
        run_id="run-1",
        plan=security_plan(),
        evidence=target_evidence(tmp_path, agent_target()),
        probe=None,
        budget=security_budget(),
        authorization_ref="ticket-42",
        requester_is_admin=True,
    )

    failed = next(event for event in audit.events if event.action == "active_scan.failed")
    assert failed.scan_kind == "agent_scan"
    assert failed.authorization_ref == "ticket-42"


async def test_scan_kind_reaches_the_policy_for_a_model_target(tmp_path: Path) -> None:
    tools = ToolRegistry()
    tools.register(build_read_evidence_tool(1_000_000))
    register_target_probe(tools)
    register_graders(tools)
    model_tool = FakeScanTool(model_output())
    register(tools, "run_aig_scan", AigScanInput, AigScanOutput, model_tool)
    policy = scan_policy()
    deps = dependencies(tmp_path, tools=tools, policy=policy)

    result = await SecurityExpert(deps).run(
        run_id="run-1",
        plan=security_plan(),
        evidence=target_evidence(tmp_path, model_target()),
        probe=None,
        budget=security_budget(),
        authorization_ref="ticket-9",
        requester_is_admin=True,
    )

    assert result.metrics["active"]["kind"] == "model_redteam_report"
    assert policy.calls[0]["scan_kind"] == "model_redteam_report"
    assert policy.calls[0]["authorization_ref"] == "ticket-9"
    assert len(model_tool.calls) == 1
