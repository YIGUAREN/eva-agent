import json
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

from eva_agent.domain.run import InputKind, RunInput, RunStatus
from eva_agent.domain.target import TargetKind, TargetSpec
from eva_agent.graph.nodes.intake import intake_node
from eva_agent.security.target_policy import SecurityPolicyError, ValidatedTarget
from tests.support.runtime import dependencies, initial_state


def demo_target(*, api_key_env: str | None = None) -> TargetSpec:
    return TargetSpec(
        name="demo",
        kind=TargetKind.OPENAI_COMPATIBLE,
        provider="demo",
        model="demo-echo",
        base_url="http://127.0.0.1:8100/v1",
        api_key_env=api_key_env,
    )


class FakePolicy:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error

    async def validate(self, target: TargetSpec) -> ValidatedTarget:
        if self.error is not None:
            raise self.error
        return ValidatedTarget(
            url=target.base_url or "",
            scheme="http",
            host="127.0.0.1",
            port=8100,
            addresses=("127.0.0.1",),
            demo=True,
        )


def target_input() -> RunInput:
    return RunInput(kind=InputKind.TARGET, target_ref="demo", target_prompts=["q1", "q2"])


def target_dependencies(tmp_path: Path, *, policy: FakePolicy | None = None, env: dict[str, str] | None = None):
    base = dependencies(tmp_path)
    return replace(
        base,
        targets={"demo": demo_target()},
        policy=cast(Any, policy or FakePolicy()),
        env=env or {},
    )


async def test_intake_target_writes_workspace_and_events(tmp_path: Path):
    events: list[tuple[str, dict[str, Any]]] = []

    async def sink(run_id: str, event_type: str, payload: dict[str, Any] | None = None) -> None:
        events.append((event_type, payload or {}))

    deps = replace(target_dependencies(tmp_path), event_sink=sink)

    result = await intake_node(initial_state(target_input()), deps)

    assert result["status"] is RunStatus.RUNNING
    evidence = result["evidence"]
    assert evidence.kind is InputKind.TARGET
    assert evidence.target == demo_target()
    assert evidence.target_prompts == ["q1", "q2"]
    assert (tmp_path / "data" / "runs" / "run-1" / "evidence" / "target.json").is_file()
    assert [event[0] for event in events] == ["target.validated", "evidence.ready"]


async def test_intake_target_rejects_unknown_target(tmp_path: Path):
    deps = target_dependencies(tmp_path)
    unknown = RunInput(kind=InputKind.TARGET, target_ref="missing", target_prompts=["q"])

    result = await intake_node(initial_state(unknown), deps)

    assert result["status"] is RunStatus.FAILED
    assert "unknown target 'missing'" in result["errors"][0]


async def test_intake_target_requires_policy(tmp_path: Path):
    deps = replace(target_dependencies(tmp_path), policy=None)

    result = await intake_node(initial_state(target_input()), deps)

    assert result["status"] is RunStatus.FAILED
    assert "security policy" in result["errors"][0]


async def test_intake_target_records_policy_failure(tmp_path: Path):
    deps = target_dependencies(tmp_path, policy=FakePolicy(SecurityPolicyError("denied")))

    result = await intake_node(initial_state(target_input()), deps)

    assert result["status"] is RunStatus.FAILED
    assert "denied" in result["errors"][0]
    assert not (tmp_path / "data" / "runs" / "run-1").exists()


async def test_intake_target_validates_env(tmp_path: Path):
    deps = replace(target_dependencies(tmp_path), targets={"demo": demo_target(api_key_env="DEMO_KEY")})

    result = await intake_node(initial_state(target_input()), deps)

    assert result["status"] is RunStatus.FAILED
    assert "DEMO_KEY" in result["errors"][0]


async def _upload_ready_payload(tmp_path: Path, files: dict[str, bytes]) -> dict[str, Any]:
    evidence_dir = tmp_path / "data" / "runs" / "run-1" / "evidence"
    evidence_dir.mkdir(parents=True)
    for name, content in files.items():
        (evidence_dir / name).write_bytes(content)
    events: list[tuple[str, dict[str, Any]]] = []

    async def sink(run_id: str, event_type: str, payload: dict[str, Any] | None = None) -> None:
        events.append((event_type, payload or {}))

    deps = replace(dependencies(tmp_path), event_sink=sink)
    run_input = RunInput(kind=InputKind.UPLOAD, upload_id="upload_a")

    result = await intake_node(initial_state(run_input), deps)

    assert result["status"] is RunStatus.RUNNING
    payload = next(payload for event_type, payload in events if event_type == "evidence.ready")
    assert str(tmp_path) not in json.dumps(payload)
    return payload


async def test_intake_upload_reports_qa_present(tmp_path: Path):
    payload = await _upload_ready_payload(
        tmp_path,
        {"qa.jsonl": b'{"query": "q", "response": "a"}\n', "qa.txt": b"q\na"},
    )

    assert payload["qa"] is True


async def test_intake_upload_reports_qa_absent(tmp_path: Path):
    payload = await _upload_ready_payload(tmp_path, {"qa.txt": b"q\na"})

    assert payload["qa"] is False
