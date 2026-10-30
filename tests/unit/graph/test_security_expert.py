import json
from datetime import UTC, datetime
from pathlib import Path

from eva_agent.adapters.aig.errors import AigTimeout
from eva_agent.adapters.aig.models import AigReport
from eva_agent.adapters.inputs.paste import write_paste_workspace
from eva_agent.config.settings import AigScanSettings, AigSettings
from eva_agent.domain.audit import AuditEvent
from eva_agent.domain.budget import Budget
from eva_agent.domain.evidence import EvidenceBundle
from eva_agent.domain.plan import Dimension, DimensionPlan
from eva_agent.domain.probe import ProbeRecord, ProbeSummary
from eva_agent.domain.result import GraderRunOutput, GraderSampleOutcome
from eva_agent.domain.run import InputKind, PastedInput
from eva_agent.domain.target import TargetKind, TargetSpec
from eva_agent.graph.dependencies import GraphDependencies
from eva_agent.graph.events import noop_event_sink
from eva_agent.graph.experts.security import SecurityExpert
from eva_agent.runtime.artifacts import run_paths
from eva_agent.runtime.probe import ProbeStore
from eva_agent.security.scan_policy import ScanPolicy
from eva_agent.security.target_policy import DemoTarget, SecurityPolicy
from eva_agent.tools.aig import AigScanInput, AigScanOutput
from eva_agent.tools.evidence import build_read_evidence_tool
from eva_agent.tools.graders import RunGradersInput
from eva_agent.tools.registry import ToolRegistry, ToolSpec
from eva_agent.tools.target import build_read_target_probe_tool
from tests.support.runtime import FakePlanner


class FakeAudit:
    def __init__(self) -> None:
        self.events: list[AuditEvent] = []

    async def record(self, event: AuditEvent) -> None:
        self.events.append(event)


class FakeEvents:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    async def __call__(self, run_id: str, event_type: str, payload: dict | None = None) -> None:
        self.events.append((event_type, payload or {}))


def target(*, base_url: str = "http://127.0.0.1:8100/v1") -> TargetSpec:
    return TargetSpec(
        name="demo",
        kind=TargetKind.OPENAI_COMPATIBLE,
        provider="demo",
        model="m",
        base_url=base_url,
    )


def evidence(
    tmp_path: Path,
    kind: InputKind = InputKind.TARGET,
    *,
    base_url: str = "http://127.0.0.1:8100/v1",
) -> EvidenceBundle:
    if kind is InputKind.PASTE:
        return write_paste_workspace(
            "run-1", PastedInput(query="q", response="def f():\n    return 1"), tmp_path / "data"
        )
    return EvidenceBundle(
        kind=kind,
        root=str(run_paths(tmp_path / "data", "run-1").evidence),
        target=target(base_url=base_url),
        target_prompts=["q"],
        manifest_sha256="a" * 64,
    )


def write_probe(tmp_path: Path) -> None:
    paths = run_paths(tmp_path / "data", "run-1")
    paths.root.mkdir(parents=True, exist_ok=True)
    store = ProbeStore(paths.target_probe)
    store.reset()
    store.append(
        ProbeRecord(
            index=0,
            prompt="q",
            started_at=datetime.now(UTC),
            status_code=200,
            latency_ms=5,
            response_text="def f():\n    return 1",
        )
    )


def security_plan() -> DimensionPlan:
    return DimensionPlan(dimension=Dimension.SECURITY, enabled=True, sample_size=2)


def security_budget() -> Budget:
    return Budget(max_cost_usd=1, max_wall_clock_s=60, max_target_requests=10)


def build_deps(
    tmp_path: Path,
    audit: FakeAudit,
    *,
    aig_enabled: bool = True,
    scan_error: Exception | None = None,
    events: FakeEvents | None = None,
    target_base_url: str = "http://127.0.0.1:8100/v1",
) -> GraphDependencies:
    tools = ToolRegistry()
    tools.register(build_read_evidence_tool(1_000_000))
    tools.register(build_read_target_probe_tool())

    async def graders(payload: RunGradersInput) -> GraderRunOutput:
        return GraderRunOutput(
            outcomes=[
                GraderSampleOutcome(grader=spec.name, sample_index=index, score=0.8)
                for spec in payload.specs
                for index in range(len(payload.rows))
            ]
        )

    tools.register(
        ToolSpec(
            name="run_graders",
            description="fake",
            input_model=RunGradersInput,
            output_model=GraderRunOutput,
            permission="internal",
            timeout_s=10,
            side_effects=False,
            idempotent=False,
            handler=graders,
        )
    )

    async def scan(payload) -> AigScanOutput:
        if scan_error is not None:
            raise scan_error
        return AigScanOutput(
            session_id="s1",
            reports=[
                AigReport.model_validate(
                    {
                        "modelName": "m",
                        "total": 10,
                        "jailbreak": 9,
                        "score": 10.0,
                        "extraBody": {
                            "vulnerabilityResults": [
                                {"vulnerability": "Harmful Content", "total": 10, "jailbreak": 9, "asr": 0.9}
                            ]
                        },
                    }
                )
            ],
            datasets=["d"],
            num_prompts=2,
            raw_result_path=str(tmp_path / "data" / "runs" / "run-1" / "aig" / "s1.json"),
        )

    tools.register(
        ToolSpec(
            name="run_aig_scan",
            description="fake",
            input_model=AigScanInput,
            output_model=AigScanOutput,
            permission="admin",
            timeout_s=10,
            side_effects=True,
            idempotent=False,
            handler=scan,
        )
    )
    policy = SecurityPolicy(
        active_scan_enabled=aig_enabled,
        requester_is_admin=True,
        authorization_ref="demo",
        demo_targets=[DemoTarget(name="demo", origin="http://127.0.0.1:8100")],
    )
    scan_policy = ScanPolicy(
        policy,
        aig=AigSettings(scan=AigScanSettings(datasets=["d"], num_prompts=2, random_seed=1)),
        eval_model_available=True,
    )
    return GraphDependencies(
        planner=FakePlanner(),
        tools=tools,
        catalog_ids=frozenset(),
        data_dir=str(tmp_path / "data"),
        max_input_bytes=1_000_000,
        max_dataset_rows=100,
        default_sample_size=5,
        targets={"demo": target(base_url=target_base_url)},
        scan_policy=scan_policy,
        audit=audit,
        aig_configured=True,
        event_sink=events if events is not None else noop_event_sink,
    )


async def test_passive_checks_run_without_a_target(tmp_path: Path) -> None:
    audit = FakeAudit()
    deps = build_deps(tmp_path, audit)
    result = await SecurityExpert(deps).run(
        run_id="run-1",
        plan=security_plan(),
        evidence=evidence(tmp_path, InputKind.PASTE),
        probe=None,
        budget=security_budget(),
    )
    assert result.status == "ok"
    assert result.score == 0.8
    assert result.metrics["active"]["status"] == "not_applicable"
    assert result.metrics["passive"]["by_grader"] == {"code_security": 0.8, "harmfulness": 0.8}


async def test_authorized_active_scan_caps_critical_score(tmp_path: Path) -> None:
    write_probe(tmp_path)
    audit = FakeAudit()
    events = FakeEvents()
    deps = build_deps(tmp_path, audit, events=events)
    probe = ProbeSummary(path="p", total=1, succeeded=1, failed=0)
    result = await SecurityExpert(deps).run(
        run_id="run-1",
        plan=security_plan(),
        evidence=evidence(tmp_path),
        probe=probe,
        budget=security_budget(),
    )
    assert result.status == "ok"
    assert result.score == 0.2
    assert result.metrics["active"]["status"] == "completed"
    assert result.metrics["active"]["severity_counts"] == {"critical": 1}
    assert [event.action for event in audit.events] == ["active_scan.requested"]
    assert audit.events[0].outcome == "allowed"
    assert [event_type for event_type, _ in events.events] == [
        "security.passive.completed",
        "security.active.requested",
        "security.active.started",
        "security.active.completed",
    ]


async def test_disabled_active_scan_is_refused_and_partial(tmp_path: Path) -> None:
    write_probe(tmp_path)
    audit = FakeAudit()
    deps = build_deps(tmp_path, audit, aig_enabled=False)
    probe = ProbeSummary(path="p", total=1, succeeded=1, failed=0)
    result = await SecurityExpert(deps).run(
        run_id="run-1",
        plan=security_plan(),
        evidence=evidence(tmp_path),
        probe=probe,
        budget=security_budget(),
    )
    assert result.status == "partial"
    assert result.score == 0.8
    assert result.metrics["active"]["status"] == "refused"
    assert result.metrics["active"]["refused_reason"] == "disabled"
    assert audit.events[0].outcome == "refused"


async def test_active_failure_is_degraded_and_partial(tmp_path: Path) -> None:
    write_probe(tmp_path)
    audit = FakeAudit()
    deps = build_deps(tmp_path, audit, scan_error=AigTimeout("slow"))
    result = await SecurityExpert(deps).run(
        run_id="run-1",
        plan=security_plan(),
        evidence=evidence(tmp_path),
        probe=ProbeSummary(path="p", total=1, succeeded=1, failed=0),
        budget=security_budget(),
    )
    assert result.status == "partial"
    assert result.score == 0.8
    assert result.metrics["active"]["status"] == "failed"
    assert result.metrics["active"]["degraded"] is True


async def test_scan_base_url_query_is_not_audited(tmp_path: Path) -> None:
    write_probe(tmp_path)
    audit = FakeAudit()
    secret_url = "http://127.0.0.1:8100/v1?key=SECRET"
    deps = build_deps(tmp_path, audit, target_base_url=secret_url)
    result = await SecurityExpert(deps).run(
        run_id="run-1",
        plan=security_plan(),
        evidence=evidence(tmp_path, base_url=secret_url),
        probe=ProbeSummary(path="p", total=1, succeeded=1, failed=0),
        budget=security_budget(),
    )
    assert result.metrics["active"]["status"] == "completed"
    requested = audit.events[0]
    assert requested.action == "active_scan.requested"
    assert "SECRET" not in json.dumps(requested.detail)
    assert requested.detail["scan_base_url"] == "http://127.0.0.1:8100"


async def test_registry_timeout_writes_terminal_audit(tmp_path: Path) -> None:
    write_probe(tmp_path)
    audit = FakeAudit()
    deps = build_deps(tmp_path, audit, scan_error=TimeoutError("tool 'run_aig_scan' timed out"))
    result = await SecurityExpert(deps).run(
        run_id="run-1",
        plan=security_plan(),
        evidence=evidence(tmp_path),
        probe=ProbeSummary(path="p", total=1, succeeded=1, failed=0),
        budget=security_budget(),
    )
    assert result.metrics["active"]["status"] == "failed"
    assert [event.action for event in audit.events] == ["active_scan.requested", "active_scan.failed"]
    assert audit.events[-1].detail["error_type"] == "TimeoutError"


class RecordedTimeoutError(TimeoutError):
    audit_recorded = True


async def test_tool_recorded_timeout_does_not_duplicate_audit(tmp_path: Path) -> None:
    write_probe(tmp_path)
    audit = FakeAudit()
    deps = build_deps(tmp_path, audit, scan_error=RecordedTimeoutError("tool 'run_aig_scan' timed out"))
    result = await SecurityExpert(deps).run(
        run_id="run-1",
        plan=security_plan(),
        evidence=evidence(tmp_path),
        probe=ProbeSummary(path="p", total=1, succeeded=1, failed=0),
        budget=security_budget(),
    )
    assert result.metrics["active"]["status"] == "failed"
    assert [event.action for event in audit.events] == ["active_scan.requested"]
