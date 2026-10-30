import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from eva_agent.domain.budget import Budget
from eva_agent.domain.evidence import EvidenceBundle
from eva_agent.domain.grader import CheckSpec, GraderSpec
from eva_agent.domain.plan import Dimension, DimensionPlan, SkippedDimension
from eva_agent.domain.run import InputKind
from eva_agent.domain.target import TargetKind, TargetSpec
from eva_agent.graph.plan_validation import PlanValidationError
from eva_agent.graph.planner import LlmPlanner, PlannerOutput
from eva_agent.tools.evidence import build_read_evidence_tool
from eva_agent.tools.graders import ListGradersInput, ListGradersOutput
from eva_agent.tools.registry import ToolRegistry, ToolSpec
from eva_agent.tools.rubric import GenerateRubricInput, RubricOutput


class FakeStructured:
    def __init__(self, output: PlannerOutput) -> None:
        self.output = output
        self.prompts: list[str] = []

    async def ainvoke(self, prompt: str) -> PlannerOutput:
        self.prompts.append(prompt)
        return self.output


async def list_handler(payload: ListGradersInput) -> ListGradersOutput:
    return ListGradersOutput(graders=[])


async def rubric_handler(payload: GenerateRubricInput) -> RubricOutput:
    return RubricOutput(rubrics=["unused"])


def build_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(build_read_evidence_tool(1_000_000))
    registry.register(
        ToolSpec(
            name="list_graders",
            description="",
            input_model=ListGradersInput,
            output_model=None,
            permission="public_read",
            timeout_s=1.0,
            side_effects=False,
            idempotent=True,
            handler=list_handler,
        )
    )
    registry.register(
        ToolSpec(
            name="generate_rubric",
            description="",
            input_model=GenerateRubricInput,
            output_model=RubricOutput,
            permission="internal",
            timeout_s=1.0,
            side_effects=False,
            idempotent=True,
            handler=rubric_handler,
        )
    )
    return registry


def demo_target() -> TargetSpec:
    return TargetSpec(
        name="demo",
        kind=TargetKind.OPENAI_COMPATIBLE,
        provider="demo",
        model="demo-echo",
        base_url="http://127.0.0.1:8100/v1",
    )


def target_evidence(tmp_path: Path) -> EvidenceBundle:
    root = tmp_path / "evidence"
    root.mkdir()
    payload = {"target": demo_target().model_dump(mode="json"), "prompts": ["q1", "q2"]}
    (root / "target.json").write_text(json.dumps(payload), encoding="utf-8")
    return EvidenceBundle(
        kind=InputKind.TARGET,
        root=str(root),
        target=demo_target(),
        target_prompts=["q1", "q2"],
        manifest_sha256="a" * 64,
    )


def custom_target_evidence(
    tmp_path: Path,
    target_payload: dict,
    target_spec: TargetSpec,
) -> EvidenceBundle:
    root = tmp_path / "evidence"
    root.mkdir()
    payload = {"target": target_payload, "prompts": ["q1", "q2"]}
    (root / "target.json").write_text(json.dumps(payload), encoding="utf-8")
    return EvidenceBundle(
        kind=InputKind.TARGET,
        root=str(root),
        target=target_spec,
        target_prompts=["q1", "q2"],
        manifest_sha256="a" * 64,
    )


def declarative(name: str, field: str, threshold: float) -> GraderSpec:
    return GraderSpec(
        level="declarative",
        name=name,
        input_map={field: f"$.{field}"},
        checks=[CheckSpec(type="numeric_threshold", field=field, op="<=", value=threshold)],
    )


def target_output() -> PlannerOutput:
    return PlannerOutput(
        objective="评估目标延迟与成本",
        evidence_summary="target demo, 2 prompts",
        dimensions=[
            DimensionPlan(
                dimension=Dimension.LATENCY,
                enabled=True,
                sample_size=3,
                graders=[declarative("latency_budget", "latency_ms", 3000)],
            ),
            DimensionPlan(
                dimension=Dimension.COST,
                enabled=True,
                sample_size=3,
                graders=[declarative("cost_budget", "cost_usd", 0.01)],
            ),
        ],
        skipped=[
            SkippedDimension(dimension=Dimension.QUALITY, reason="阶段 2 不支持"),
            SkippedDimension(dimension=Dimension.SECURITY, reason="阶段 3 实现"),
        ],
    )


def budget() -> Budget:
    return Budget(max_cost_usd=1.0, max_wall_clock_s=60.0, max_target_requests=5)


async def test_target_planner_builds_latency_and_cost(tmp_path: Path):
    structured = FakeStructured(target_output())
    planner = LlmPlanner(None, build_registry(), structured_runner=structured)

    plan = await planner.plan(target_evidence(tmp_path), budget())

    assert [item.dimension for item in plan.dimensions if item.enabled] == [Dimension.LATENCY, Dimension.COST]
    prompt = structured.prompts[0]
    assert "numeric_threshold" in prompt
    assert "json_parse" in prompt
    assert "q1" in prompt
    assert "max_target_requests=5" in prompt
    assert '"dimension": "quality"|"latency"|"cost"|"security"' in prompt
    assert 'security MUST set `"graders": []`' in prompt
    assert "`score.weights` keys must equal the check `name`, or the check `type` when it has no name." in prompt


async def test_target_planner_reads_target_json_not_manifest(tmp_path: Path):
    structured = FakeStructured(target_output())
    planner = LlmPlanner(None, build_registry(), structured_runner=structured)

    await planner.plan(target_evidence(tmp_path), budget())

    assert '"kind": "target"' in structured.prompts[0]
    assert '"name": "demo"' in structured.prompts[0]


async def test_target_planner_rejects_dimension_without_sample_size(tmp_path: Path):
    output = PlannerOutput(
        objective="x",
        evidence_summary="x",
        dimensions=[
            DimensionPlan(
                dimension=Dimension.QUALITY,
                enabled=True,
                graders=[GraderSpec(level="builtin", name="relevance", builtin_id="relevance")],
            )
        ],
        skipped=[
            SkippedDimension(dimension=Dimension.LATENCY, reason="x"),
            SkippedDimension(dimension=Dimension.COST, reason="x"),
            SkippedDimension(dimension=Dimension.SECURITY, reason="x"),
        ],
    )
    planner = LlmPlanner(None, build_registry(), structured_runner=FakeStructured(output))

    with pytest.raises(PlanValidationError, match="quality"):
        await planner.plan(target_evidence(tmp_path), budget())


async def test_target_prompt_marks_prompts_as_untrusted(tmp_path: Path):
    structured = FakeStructured(target_output())
    planner = LlmPlanner(None, build_registry(), structured_runner=structured)

    await planner.plan(target_evidence(tmp_path), budget())

    prompt = structured.prompts[0]
    assert "<evidence>" in prompt
    assert prompt.index("<evidence>") < prompt.index("q1") < prompt.index("</evidence>")
    assert "untrusted" in prompt


async def test_target_prompt_renders_only_name_kind_provider_and_model(tmp_path: Path):
    spec = TargetSpec(
        name="leaky",
        kind=TargetKind.HTTP_JSON,
        provider="acme",
        model="m-1",
        url="https://example.invalid/secret-endpoint",
        method="POST",
        headers={"Content-Type": "application/x-secret-marker"},
        request_template={"prompt": "{{prompt}}"},
        response_path="$.output",
    )
    payload_target = spec.model_dump(mode="json")
    payload_target["base_url"] = "https://base.example.invalid/secret-base"
    payload_target["header_env"] = {"X-Api-Key": "TARGET_API_KEY"}
    structured = FakeStructured(target_output())
    planner = LlmPlanner(None, build_registry(), structured_runner=structured)

    await planner.plan(custom_target_evidence(tmp_path, payload_target, spec), budget())

    prompt = structured.prompts[0]
    for rendered in ('"name": "leaky"', '"kind": "http_json"', '"provider": "acme"', '"model": "m-1"'):
        assert rendered in prompt
    for leaked in (
        "base_url",
        "https://base.example.invalid/secret-base",
        "https://example.invalid/secret-endpoint",
        "request_template",
        "application/x-secret-marker",
        "TARGET_API_KEY",
        "$.output",
    ):
        assert leaked not in prompt


async def test_target_prompt_states_unused_dimensions_need_skip_reason(tmp_path: Path):
    structured = FakeStructured(target_output())
    planner = LlmPlanner(None, build_registry(), structured_runner=structured)

    await planner.plan(target_evidence(tmp_path), budget())

    assert "Every dimension you do not enable MUST appear in skipped" in structured.prompts[0]


async def test_target_prompt_omits_declarative_fields_for_non_declarative_graders(tmp_path: Path):
    structured = FakeStructured(target_output())
    planner = LlmPlanner(None, build_registry(), structured_runner=structured)

    await planner.plan(target_evidence(tmp_path), budget())

    assert "builtin and llm_rubric graders must omit `input_map`, `checks`, and `score`" in structured.prompts[0]


def _llm_rubric_payload(**grader_overrides: object) -> dict:
    grader: dict[str, object] = {
        "level": "llm_rubric",
        "name": "helpfulness",
        "mode": "pointwise",
        "input_map": {"response": "$.response"},
        "checks": [],
        "score": {"aggregation": "weighted_mean", "weights": {}},
    }
    grader.update(grader_overrides)
    return {
        "objective": "评估回答质量",
        "evidence_summary": "target demo, 2 prompts",
        "dimensions": [
            {
                "dimension": "quality",
                "enabled": True,
                "sample_size": 2,
                "graders": [grader],
            }
        ],
        "skipped": [],
    }


def test_planner_output_drops_empty_declarative_fields_from_llm_rubric() -> None:
    output = PlannerOutput.model_validate(_llm_rubric_payload())

    grader = output.dimensions[0].graders[0]
    assert grader.level == "llm_rubric"
    assert grader.checks is None
    assert grader.score is None


def test_planner_output_rejects_non_empty_checks_from_llm_rubric() -> None:
    payload = _llm_rubric_payload(checks=[{"type": "json_parse", "field": "response"}])

    with pytest.raises(ValidationError, match="checks are only valid for declarative graders"):
        PlannerOutput.model_validate(payload)


def test_planner_output_rejects_non_empty_weights_from_llm_rubric() -> None:
    payload = _llm_rubric_payload(score={"aggregation": "weighted_mean", "weights": {"json_parse": 1.0}})

    with pytest.raises(ValidationError, match="score is only valid for declarative graders"):
        PlannerOutput.model_validate(payload)


def _target_declarative_payload(weights: dict[str, float]) -> dict:
    return {
        "objective": "评估目标延迟",
        "evidence_summary": "target demo, 2 prompts",
        "dimensions": [
            {
                "dimension": "latency",
                "enabled": True,
                "sample_size": 2,
                "graders": [
                    {
                        "level": "declarative",
                        "name": "latency_budget",
                        "mode": "pointwise",
                        "input_map": {"latency_ms": "$.latency_ms"},
                        "checks": [
                            {
                                "name": "latency_under_10s",
                                "type": "numeric_threshold",
                                "field": "latency_ms",
                                "op": "<",
                                "value": 10000,
                            }
                        ],
                        "score": {"aggregation": "weighted_mean", "weights": weights},
                    }
                ],
            }
        ],
        "skipped": [
            {"dimension": "cost", "reason": "不在目标范围"},
            {"dimension": "quality", "reason": "不在目标范围"},
            {"dimension": "security", "reason": "不在目标范围"},
        ],
    }


def test_planner_output_rewrites_weight_type_key_to_the_check_name() -> None:
    output = PlannerOutput.model_validate(_target_declarative_payload({"numeric_threshold": 1.0}))

    grader = output.dimensions[0].graders[0]
    assert grader.score is not None
    assert grader.score.weights == {"latency_under_10s": 1.0}


async def test_target_planner_rejects_weight_key_that_matches_no_check_type(tmp_path: Path):
    output = PlannerOutput.model_validate(_target_declarative_payload({"json_parse": 1.0}))
    planner = LlmPlanner(None, build_registry(), structured_runner=FakeStructured(output))

    with pytest.raises(PlanValidationError, match="unknown weight key: json_parse"):
        await planner.plan(target_evidence(tmp_path), budget())
