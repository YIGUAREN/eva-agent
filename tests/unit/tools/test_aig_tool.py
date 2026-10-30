import json
from pathlib import Path

import pytest

from eva_agent.adapters.aig.errors import AigTimeout
from eva_agent.config.settings import AigScanSettings, AigSettings, ModelConfig
from eva_agent.domain.audit import AuditEvent
from eva_agent.domain.target import TargetKind, TargetSpec
from eva_agent.tools.aig import AigScanInput, build_run_aig_scan_tool

RAW = {
    "result": {
        "msgType": "json",
        "content": [{"modelName": "demo-echo", "total": 2, "jailbreak": 1, "score": 50.0}],
    }
}


class FakeClient:
    def __init__(self, *, error: Exception | None = None) -> None:
        self.error = error
        self.body: dict | None = None

    async def create(self, body: dict) -> str:
        self.body = body
        return "session-1"

    async def wait(self, session_id: str, *, max_wait_s: float) -> str:
        if self.error is not None:
            raise self.error
        return "completed"

    async def result(self, session_id: str) -> dict:
        return RAW


class FakeAudit:
    def __init__(self) -> None:
        self.events: list[AuditEvent] = []

    async def record(self, event: AuditEvent) -> None:
        self.events.append(event)


def target() -> TargetSpec:
    return TargetSpec(
        name="demo",
        kind=TargetKind.OPENAI_COMPATIBLE,
        provider="demo",
        model="demo-echo",
        base_url="http://127.0.0.1:8100/v1",
        api_key_env="DEMO_KEY",
    )


def settings() -> AigSettings:
    return AigSettings(scan=AigScanSettings(datasets=["d"], num_prompts=2, random_seed=1))


def eval_model() -> ModelConfig:
    return ModelConfig(
        provider="openai-compatible",
        model="judge",
        base_url="https://judge/v1",
        api_key_env="JUDGE_KEY",
    )


async def test_scan_writes_raw_result_and_audits_lifecycle(tmp_path: Path) -> None:
    audit = FakeAudit()
    client = FakeClient()
    tool = build_run_aig_scan_tool(
        client=client,
        settings=settings(),
        eval_model=eval_model(),
        env={"DEMO_KEY": "target-secret", "JUDGE_KEY": "judge-secret"},
        audit=audit,
        data_dir=str(tmp_path / "data"),
    )
    output = await tool.handler(
        AigScanInput(
            run_id="run-1",
            target=target(),
            scan_base_url="http://demo-target:8100/v1",
            authorization_ref="demo",
            max_wait_s=10,
        )
    )
    assert output.session_id == "session-1"
    assert output.reports[0].asr == 0.5
    assert output.datasets == ["d"]
    assert client.body is not None
    assert client.body["content"]["model"][0]["token"] == "target-secret"
    raw = Path(output.raw_result_path).read_text(encoding="utf-8")
    assert json.loads(raw)["result"]["msgType"] == "json"
    assert [event.action for event in audit.events] == ["active_scan.started", "active_scan.completed"]
    assert audit.events[0].detail["session_id"] == "session-1"
    assert "target-secret" not in json.dumps([event.detail for event in audit.events])


async def test_timeout_writes_cancelled_audit_and_reraises(tmp_path: Path) -> None:
    audit = FakeAudit()
    tool = build_run_aig_scan_tool(
        client=FakeClient(error=AigTimeout("slow")),
        settings=settings(),
        eval_model=eval_model(),
        env={"DEMO_KEY": "t", "JUDGE_KEY": "j"},
        audit=audit,
        data_dir=str(tmp_path / "data"),
    )
    with pytest.raises(AigTimeout):
        await tool.handler(
            AigScanInput(
                run_id="run-1",
                target=target(),
                scan_base_url="http://demo-target:8100/v1",
                authorization_ref="demo",
                max_wait_s=10,
            )
        )
    assert [event.action for event in audit.events] == [
        "active_scan.started",
        "active_scan.failed",
        "scan.cancelled",
    ]
    assert audit.events[-1].outcome == "cancelled"
