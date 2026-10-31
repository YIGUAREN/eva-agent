"""Budget reservation and truncation behavior for the security expert."""

from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from eva_agent.adapters.aig.models import AigReport
from eva_agent.budget.context import BudgetModels, bind_budget
from eva_agent.budget.estimator import CostEstimator
from eva_agent.budget.ledger import BudgetLedger
from eva_agent.config.settings import AigScanSettings, AigSettings, ModelConfig
from eva_agent.domain.audit import AuditEvent
from eva_agent.domain.budget import Budget
from eva_agent.domain.evidence import EvidenceBundle, EvidenceFile
from eva_agent.domain.plan import Dimension, DimensionPlan
from eva_agent.domain.probe import ProbeRecord, ProbeSummary
from eva_agent.domain.result import GraderRunOutput, GraderSampleOutcome
from eva_agent.domain.run import InputKind
from eva_agent.domain.target import TargetKind, TargetSpec
from eva_agent.graph.dependencies import GraphDependencies
from eva_agent.graph.experts.security import SecurityExpert
from eva_agent.pricing import ModelPrice, PricingTable
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

_DIGEST = "0" * 64


class NoopStore:
    async def load(self, run_id: str) -> None:
        return None

    async def save(self, run_id: str, usage: Any) -> None:
        return None


class FakeEvents:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []

    async def __call__(self, run_id: str, event_type: str, payload: dict[str, Any] | None = None) -> None:
        self.events.append((event_type, payload or {}))


class RecordingAudit:
    def __init__(self) -> None:
        self.events: list[AuditEvent] = []

    async def record(self, event: AuditEvent) -> None:
        self.events.append(event)


def model() -> ModelConfig:
    return ModelConfig(
        provider="litellm",
        model="demo-echo",
        base_url="http://127.0.0.1:8100/v1",
        api_key_env="DEMO_KEY",
    )


def estimator() -> CostEstimator:
    return CostEstimator(
        PricingTable(
            [
                ModelPrice(
                    provider="litellm",
                    model="demo-echo",
                    input_per_million=Decimal("0"),
                    output_per_million=Decimal("1"),
                )
            ]
        )
    )


def bound_models() -> BudgetModels:
    return BudgetModels(agent=model(), judge=model(), attack=model())


def model_target() -> TargetSpec:
    return TargetSpec(
        name="demo",
        kind=TargetKind.OPENAI_COMPATIBLE,
        provider="demo",
        model="m",
        base_url="http://127.0.0.1:8100/v1",
    )


def security_plan() -> DimensionPlan:
    return DimensionPlan(dimension=Dimension.SECURITY, enabled=True, sample_size=2)


def security_budget() -> Budget:
    return Budget(max_cost_usd=1.0, max_wall_clock_s=60.0, max_target_requests=10)


def target_evidence(tmp_path: Path) -> EvidenceBundle:
    return EvidenceBundle(
        kind=InputKind.TARGET,
        root=str(run_paths(tmp_path / "data", "run-1").evidence),
        target=model_target(),
        target_prompts=["q"],
        manifest_sha256=_DIGEST,
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
            response_text="safe answer",
        )
    )


def scan_output(tmp_path: Path) -> AigScanOutput:
    report = AigReport.model_validate(
        {
            "modelName": "m",
            "total": 10,
            "jailbreak": 1,
            "score": 90.0,
            "extraBody": {
                "vulnerabilityResults": [
                    {"vulnerability": "Harmful Content", "total": 10, "jailbreak": 1, "asr": 0.1}
                ]
            },
        }
    )
    return AigScanOutput(
        session_id="s1",
        reports=[report],
        datasets=["d"],
        num_prompts=2,
        raw_result_path=str(tmp_path / "data" / "runs" / "run-1" / "aig" / "s1.json"),
    )


def default_graders(payload: RunGradersInput) -> GraderRunOutput:
    return GraderRunOutput(
        outcomes=[
            GraderSampleOutcome(grader=spec.name, sample_index=index, score=0.8)
            for spec in payload.specs
            for index in range(len(payload.rows))
        ]
    )


def build_deps(
    tmp_path: Path,
    events: FakeEvents,
    *,
    grader=None,
    scan: AigScanOutput | None = None,
    aig_configured: bool = True,
) -> GraphDependencies:
    tools = ToolRegistry()
    tools.register(build_read_evidence_tool(1_000_000))
    tools.register(build_read_target_probe_tool())

    async def run_graders(payload: RunGradersInput) -> GraderRunOutput:
        return grader(payload) if grader is not None else default_graders(payload)

    tools.register(
        ToolSpec(
            name="run_graders",
            description="fake",
            input_model=RunGradersInput,
            output_model=GraderRunOutput,
            permission="internal",
            timeout_s=10.0,
            side_effects=False,
            idempotent=False,
            handler=run_graders,
        )
    )

    if scan is not None:

        async def run_aig_scan(payload: AigScanInput) -> AigScanOutput:
            return scan

        tools.register(
            ToolSpec(
                name="run_aig_scan",
                description="fake",
                input_model=AigScanInput,
                output_model=AigScanOutput,
                permission="admin",
                timeout_s=10.0,
                side_effects=True,
                idempotent=False,
                handler=run_aig_scan,
            )
        )

    policy = SecurityPolicy(
        active_scan_enabled=True,
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
        targets={"demo": model_target()},
        scan_policy=scan_policy,
        audit=RecordingAudit(),
        aig_configured=aig_configured,
        event_sink=events,
    )


def skill_evidence(tmp_path: Path) -> EvidenceBundle:
    root = tmp_path / "evidence"
    root.mkdir()
    (root / "SKILL.md").write_text("# Greeter\n\nSay hello.\n", encoding="utf-8")
    (root / "qa.jsonl").write_text('{"query": "q", "response": "safe answer"}\n', encoding="utf-8")
    files = [
        EvidenceFile(path=name, size=(root / name).stat().st_size, sha256=_DIGEST)
        for name in ("SKILL.md", "qa.jsonl")
    ]
    return EvidenceBundle(kind=InputKind.UPLOAD, root=str(root), files=files, manifest_sha256=_DIGEST)


def budget_events(events: FakeEvents) -> list[dict[str, Any]]:
    return [payload for event_type, payload in events.events if event_type == "budget.exhausted"]


async def test_aig_scan_reserves_only_the_aig_requests(tmp_path: Path) -> None:
    events = FakeEvents()
    deps = build_deps(tmp_path, events, scan=scan_output(tmp_path))
    write_probe(tmp_path)
    ledger = BudgetLedger(
        Budget(max_cost_usd=1.0, max_wall_clock_s=60.0, max_target_requests=100),
        store=NoopStore(),
    )
    with bind_budget(ledger, estimator(), bound_models()):
        # The probe node already charged these two target requests.
        await ledger.reserve_requests(2)
        result = await SecurityExpert(deps).run(
            run_id="run-1",
            plan=security_plan(),
            evidence=target_evidence(tmp_path),
            probe=ProbeSummary(path="p", total=2, succeeded=2, failed=0),
            budget=security_budget(),
        )
        snapshot = ledger.snapshot()

    assert result.metrics["active"]["status"] == "completed"
    assert snapshot.target_requests == 4
    assert snapshot.aig_requests == 2
    assert snapshot.spent_usd == estimator().estimate_aig_scan(model(), requests=2)
    assert snapshot.spent_usd == Decimal("0.001024")
    assert snapshot.reserved_usd == Decimal("0")
    assert snapshot.estimated is True
    assert budget_events(events) == []


async def test_a_refused_aig_scan_emits_exactly_one_budget_event(tmp_path: Path) -> None:
    events = FakeEvents()
    deps = build_deps(tmp_path, events, scan=scan_output(tmp_path))
    write_probe(tmp_path)
    ledger = BudgetLedger(
        Budget(max_cost_usd=0.0001, max_wall_clock_s=60.0, max_target_requests=100),
        store=NoopStore(),
    )
    with bind_budget(ledger, estimator(), bound_models()):
        result = await SecurityExpert(deps).run(
            run_id="run-1",
            plan=security_plan(),
            evidence=target_evidence(tmp_path),
            probe=ProbeSummary(path="p", total=2, succeeded=2, failed=0),
            budget=security_budget(),
        )

    assert result.metrics["active"]["status"] == "failed"
    assert result.truncated_reason == "budget"
    exhausted = budget_events(events)
    assert len(exhausted) == 1
    assert set(exhausted[0]) == {"spent_usd", "remaining_usd", "dimension"}
    assert exhausted[0]["dimension"] == "security"
    assert exhausted[0]["spent_usd"] == 0.0
    assert exhausted[0]["remaining_usd"] == pytest.approx(0.0001)


async def test_threat_only_truncation_reaches_the_result_and_one_event(tmp_path: Path) -> None:
    events = FakeEvents()

    def grader(payload: RunGradersInput) -> GraderRunOutput:
        names = {spec.name for spec in payload.specs}
        if "threat_analysis" in names:
            return GraderRunOutput(
                outcomes=[
                    GraderSampleOutcome(grader="threat_analysis", sample_index=index, score=1.0)
                    for index in range(len(payload.rows))
                ],
                truncated_reason="budget",
            )
        return default_graders(payload)

    deps = build_deps(tmp_path, events, grader=grader, aig_configured=False)
    result = await SecurityExpert(deps).run(
        run_id="run-1",
        plan=security_plan(),
        evidence=skill_evidence(tmp_path),
        probe=None,
        budget=security_budget(),
    )

    assert result.metrics["passive"]["truncated_reason"] == "budget"
    assert result.truncated_reason == "budget"
    exhausted = budget_events(events)
    assert len(exhausted) == 1
    assert exhausted[0]["dimension"] == "security"


async def test_unlimited_ledger_leaves_the_counters_and_events_untouched(tmp_path: Path) -> None:
    events = FakeEvents()
    deps = build_deps(tmp_path, events, scan=scan_output(tmp_path))
    write_probe(tmp_path)
    ledger = BudgetLedger.unlimited(Budget(max_cost_usd=0.01, max_wall_clock_s=1.0, max_target_requests=1))
    with bind_budget(ledger, estimator(), bound_models()):
        result = await SecurityExpert(deps).run(
            run_id="run-1",
            plan=security_plan(),
            evidence=target_evidence(tmp_path),
            probe=ProbeSummary(path="p", total=2, succeeded=2, failed=0),
            budget=security_budget(),
        )
        snapshot = ledger.snapshot()

    assert result.metrics["active"]["status"] == "completed"
    assert snapshot.target_requests == 0
    assert snapshot.aig_requests == 0
    assert snapshot.spent_usd == Decimal("0")
    assert snapshot.estimated is False
    assert budget_events(events) == []
