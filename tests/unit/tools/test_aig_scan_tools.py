import io
import json
import zipfile
from pathlib import Path
from typing import cast

import pytest

from eva_agent.adapters.aig.attachments import MAX_ATTACHMENTS
from eva_agent.adapters.aig.client import AigScanClient
from eva_agent.adapters.aig.errors import AigError, AigTimeout, AigUnavailable
from eva_agent.config.settings import AigSettings, AigSkillScanSettings, MissingSecretError, ModelConfig
from eva_agent.domain.target import TargetKind, TargetSpec
from eva_agent.security.audit import noop_audit_sink
from eva_agent.tools.aig import (
    AIG_TOOL_TIMEOUT_MARGIN_S,
    AigAgentScanInput,
    AigPackageError,
    AigScanInput,
    AigSkillScanInput,
    build_run_aig_agent_scan_tool,
    build_run_aig_scan_tool,
    build_run_aig_skill_scan_tool,
)

EVAL = ModelConfig(
    provider="openai-compatible",
    model="judge",
    base_url="http://judge.local/v1",
    api_key_env="JUDGE_KEY",
)

AGENT_RESULT = {
    "id": "msg-1",
    "type": "resultUpdate",
    "timestamp": 1,
    "result": {
        "schema_version": "agent-security-report@1",
        "agent_name": "http",
        "score": 100,
        "risk_type": "safe",
        "total_tests": 5,
        "vulnerable_tests": 0,
        "results": [],
        "owasp_agentic_2026_top10": [],
        "report_description": "",
        "language": "en",
    },
}

SKILL_RESULT = {
    "id": "msg-2",
    "type": "resultUpdate",
    "timestamp": 2,
    "result": {"readme": "", "score": 100, "language": "YAML", "results": [], "llm": "judge"},
}

ATTACHMENT = "attachment_0123456789abcdef0123456789abcdef_1793422813849140825_1793422813.csv"
ATTACHMENT_2 = "attachment_0123456789abcdef0123456789abcdef_1793422813849140826_1793422814.csv"
ATTACHMENT_3 = "attachment_0123456789abcdef0123456789abcdef_1793422813849140827_1793422815.csv"
CSV_BYTES = b"\xef\xbb\xbfstatus,modelName\ncompleted,demo\n"
MODEL_RESULT = {
    "result": {
        "msgType": "json",
        "content": [
            {
                "modelName": "demo-echo",
                "total": 2,
                "jailbreak": 1,
                "score": 50.0,
                "attachment": ATTACHMENT,
            }
        ],
    }
}
TWO_MODEL_RESULT = {
    "result": {
        "msgType": "json",
        "content": [
            {"modelName": "demo-echo", "total": 2, "jailbreak": 1, "score": 50.0, "attachment": ATTACHMENT},
            {"modelName": "demo-echo", "total": 2, "jailbreak": 1, "score": 50.0, "attachment": ATTACHMENT_2},
            {"modelName": "demo-echo", "total": 2, "jailbreak": 1, "score": 50.0, "attachment": ATTACHMENT},
        ],
    }
}
THREE_MODEL_RESULT = {
    "result": {
        "msgType": "json",
        "content": [
            {"modelName": "demo-echo", "total": 2, "jailbreak": 1, "score": 50.0, "attachment": ATTACHMENT},
            {"modelName": "demo-echo", "total": 2, "jailbreak": 1, "score": 50.0, "attachment": ATTACHMENT_2},
            {"modelName": "demo-echo", "total": 2, "jailbreak": 1, "score": 50.0, "attachment": ATTACHMENT_3},
        ],
    }
}


class FakeClient:
    def __init__(
        self,
        result: dict,
        *,
        wait_error: Exception | None = None,
        upload_error: Exception | None = None,
        download_error: Exception | None = None,
    ) -> None:
        self.result_payload = result
        self.wait_error = wait_error
        self.upload_error = upload_error
        self.download_error = download_error
        self.bodies: list[dict] = []
        self.uploads: list[tuple[str, bytes]] = []
        self.waits: list[tuple[str, float]] = []
        self.downloaded: list[tuple[str, str]] = []

    async def create(self, body: dict) -> str:
        self.bodies.append(body)
        return "sess-1"

    async def wait(self, session_id: str, *, max_wait_s: float) -> str:
        self.waits.append((session_id, max_wait_s))
        if self.wait_error is not None:
            raise self.wait_error
        return "done"

    async def result(self, session_id: str) -> dict:
        return self.result_payload

    async def upload(self, filename: str, content: bytes) -> str:
        if self.upload_error is not None:
            raise self.upload_error
        self.uploads.append((filename, content))
        return "pkg_1793423249288665728_1793423249.zip"

    async def download(self, session_id: str, file_url: str) -> bytes:
        if self.download_error is not None:
            raise self.download_error
        self.downloaded.append((session_id, file_url))
        return CSV_BYTES


class NoDownloadClient:
    """A scan client that predates the download capability."""

    def __init__(self, inner: FakeClient) -> None:
        self._inner = inner

    async def create(self, body: dict) -> str:
        return await self._inner.create(body)

    async def wait(self, session_id: str, *, max_wait_s: float) -> str:
        return await self._inner.wait(session_id, max_wait_s=max_wait_s)

    async def result(self, session_id: str) -> dict:
        return await self._inner.result(session_id)

    async def upload(self, filename: str, content: bytes) -> str:
        return await self._inner.upload(filename, content)


class RecordingAudit:
    def __init__(self) -> None:
        self.events = []

    async def record(self, event) -> None:
        self.events.append(event)


def agent_target() -> TargetSpec:
    return TargetSpec(
        name="support-agent",
        kind=TargetKind.AGENT_HTTP,
        provider="dify",
        model="agent",
        url="https://agent.example.com/chat",
        method="POST",
        headers={"X-Trace": "1"},
        header_env={"Authorization": "AGENT_TOKEN"},
        request_template={"message": "{{prompt}}"},
        response_path="$.reply",
    )


def model_target() -> TargetSpec:
    return TargetSpec(
        name="demo",
        kind=TargetKind.OPENAI_COMPATIBLE,
        provider="demo",
        model="demo-echo",
        base_url="http://demo:8100/v1",
    )


def model_input() -> AigScanInput:
    return AigScanInput(
        run_id="run-1",
        target=model_target(),
        scan_base_url="http://demo:8100/v1",
        authorization_ref="demo",
        max_wait_s=30.0,
    )


def model_tool(client: FakeClient, *, data_dir: Path, audit=None):
    return build_run_aig_scan_tool(
        client=client,
        settings=AigSettings(),
        eval_model=EVAL,
        env={"JUDGE_KEY": "judge-token"},
        audit=audit or noop_audit_sink,
        data_dir=str(data_dir),
    )


@pytest.mark.asyncio
async def test_agent_scan_resolves_header_secrets_and_persists_the_raw_result(tmp_path: Path) -> None:
    client = FakeClient(AGENT_RESULT)
    audit = RecordingAudit()
    tool = build_run_aig_agent_scan_tool(
        client=client,
        settings=AigSettings(),
        eval_model=EVAL,
        env={"JUDGE_KEY": "judge-token", "AGENT_TOKEN": "Bearer target"},
        audit=audit,
        data_dir=str(tmp_path),
    )

    output = await tool.handler(
        AigAgentScanInput(
            run_id="run-1",
            target=agent_target(),
            authorization_ref="ticket-42",
            max_wait_s=30.0,
        )
    )

    assert client.uploads == []
    assert client.bodies[0]["type"] == "agent_scan"
    config = client.bodies[0]["content"]["agent_config"]
    assert "Bearer target" in config
    assert output.session_id == "sess-1"
    assert output.report.total_tests == 5
    raw = tmp_path / "runs" / "run-1" / "aig" / "sess-1.json"
    assert json.loads(raw.read_text(encoding="utf-8")) == AGENT_RESULT
    assert [event.action for event in audit.events] == ["active_scan.started", "active_scan.completed"]
    assert {event.scan_kind for event in audit.events} == {"agent_scan"}
    assert {event.authorization_ref for event in audit.events} == {"ticket-42"}
    details = json.dumps([event.detail for event in audit.events])
    assert "Bearer target" not in details
    assert "judge-token" not in details


@pytest.mark.asyncio
async def test_agent_scan_missing_header_secret_fails_before_creating_a_task(tmp_path: Path) -> None:
    client = FakeClient(AGENT_RESULT)
    audit = RecordingAudit()
    tool = build_run_aig_agent_scan_tool(
        client=client,
        settings=AigSettings(),
        eval_model=EVAL,
        env={"JUDGE_KEY": "judge-token"},
        audit=audit,
        data_dir=str(tmp_path),
    )

    with pytest.raises(MissingSecretError):
        await tool.handler(
            AigAgentScanInput(
                run_id="run-1",
                target=agent_target(),
                authorization_ref="ticket-42",
                max_wait_s=30.0,
            )
        )

    assert client.bodies == []
    assert [event.action for event in audit.events] == ["active_scan.failed"]
    assert audit.events[0].detail == {"error_type": "missing_target_key"}


@pytest.mark.asyncio
async def test_agent_scan_timeout_records_failure_and_cancellation(tmp_path: Path) -> None:
    client = FakeClient(AGENT_RESULT, wait_error=AigTimeout("too slow"))
    audit = RecordingAudit()
    tool = build_run_aig_agent_scan_tool(
        client=client,
        settings=AigSettings(),
        eval_model=EVAL,
        env={"JUDGE_KEY": "judge-token", "AGENT_TOKEN": "Bearer target"},
        audit=audit,
        data_dir=str(tmp_path),
    )

    with pytest.raises(AigTimeout):
        await tool.handler(
            AigAgentScanInput(
                run_id="run-1",
                target=agent_target(),
                authorization_ref="ticket-42",
                max_wait_s=30.0,
            )
        )

    assert [event.action for event in audit.events] == [
        "active_scan.started",
        "active_scan.failed",
        "scan.cancelled",
    ]


def write_package(root: Path) -> list[str]:
    (root / "SKILL.md").write_text("# Skill\n", encoding="utf-8")
    (root / "scripts").mkdir()
    (root / "scripts" / "run.py").write_text("print('hi')\n", encoding="utf-8")
    return ["SKILL.md", "scripts/run.py"]


def skill_tool(client: FakeClient, *, data_dir: Path, audit=None, settings: AigSettings | None = None):
    return build_run_aig_skill_scan_tool(
        client=client,
        settings=settings or AigSettings(),
        eval_model=EVAL,
        env={"JUDGE_KEY": "judge-token"},
        audit=audit or noop_audit_sink,
        data_dir=str(data_dir),
    )


@pytest.mark.asyncio
async def test_skill_scan_uploads_one_archive_and_passes_the_file_url(tmp_path: Path) -> None:
    evidence_root = tmp_path / "evidence"
    evidence_root.mkdir()
    files = write_package(evidence_root)
    client = FakeClient(SKILL_RESULT)
    audit = RecordingAudit()

    output = await skill_tool(client, data_dir=tmp_path, audit=audit).handler(
        AigSkillScanInput(
            run_id="run-1",
            root=str(evidence_root),
            files=files,
            authorization_ref="ticket-42",
            max_wait_s=30.0,
        )
    )

    assert len(client.uploads) == 1
    filename, content = client.uploads[0]
    assert filename.endswith(".zip")
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        assert sorted(archive.namelist()) == files
    body = client.bodies[0]
    assert body["type"] == "skill_scan"
    assert body["content"]["attachments"] == "pkg_1793423249288665728_1793423249.zip"
    assert "eval_model" not in body["content"]
    assert output.report.score == 100
    assert (tmp_path / "runs" / "run-1" / "aig" / "sess-1.json").is_file()
    assert {event.scan_kind for event in audit.events} == {"skill_scan"}
    assert "judge-token" not in json.dumps([event.detail for event in audit.events])


@pytest.mark.asyncio
async def test_skill_scan_refuses_an_oversized_package_before_uploading(tmp_path: Path) -> None:
    evidence_root = tmp_path / "evidence"
    evidence_root.mkdir()
    files = write_package(evidence_root)
    (evidence_root / "big.bin").write_bytes(b"0" * 4_096)
    client = FakeClient(SKILL_RESULT)
    audit = RecordingAudit()
    settings = AigSettings(skill_scan=AigSkillScanSettings(max_package_bytes=1_024))

    tool = skill_tool(client, data_dir=tmp_path, audit=audit, settings=settings)
    with pytest.raises(AigPackageError):
        await tool.handler(
            AigSkillScanInput(
                run_id="run-1",
                root=str(evidence_root),
                files=[*files, "big.bin"],
                authorization_ref="ticket-42",
                max_wait_s=30.0,
            )
        )

    assert client.uploads == []
    assert [event.action for event in audit.events] == ["active_scan.failed"]


@pytest.mark.asyncio
async def test_skill_scan_rejects_an_oversized_member_without_reading_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evidence_root = tmp_path / "evidence"
    evidence_root.mkdir()
    files = write_package(evidence_root)
    (evidence_root / "big.bin").write_bytes(b"0" * 4_096)
    client = FakeClient(SKILL_RESULT)
    audit = RecordingAudit()
    settings = AigSettings(skill_scan=AigSkillScanSettings(max_package_bytes=1_024))
    original_read_bytes = Path.read_bytes

    def read_bytes_guard(self: Path) -> bytes:
        if self.name == "big.bin":
            raise AssertionError("an oversized member must be rejected before read_bytes")
        return original_read_bytes(self)

    monkeypatch.setattr(Path, "read_bytes", read_bytes_guard)

    tool = skill_tool(client, data_dir=tmp_path, audit=audit, settings=settings)
    with pytest.raises(AigPackageError, match="exceeds 1024 bytes"):
        await tool.handler(
            AigSkillScanInput(
                run_id="run-1",
                root=str(evidence_root),
                files=[*files, "big.bin"],
                authorization_ref="ticket-42",
                max_wait_s=30.0,
            )
        )

    assert client.uploads == []
    assert [event.action for event in audit.events] == ["active_scan.failed"]
    assert audit.events[0].detail == {"error_type": "AigPackageError"}


@pytest.mark.asyncio
async def test_skill_scan_rejects_a_file_outside_the_evidence_root(tmp_path: Path) -> None:
    evidence_root = tmp_path / "evidence"
    evidence_root.mkdir()
    write_package(evidence_root)
    tool = skill_tool(FakeClient(SKILL_RESULT), data_dir=tmp_path)
    with pytest.raises(AigPackageError):
        await tool.handler(
            AigSkillScanInput(
                run_id="run-1",
                root=str(evidence_root),
                files=["../../etc/passwd"],
                authorization_ref="ticket-42",
                max_wait_s=30.0,
            )
        )


@pytest.mark.asyncio
async def test_skill_scan_rewrites_dotdot_members_to_the_resolved_relative_path(tmp_path: Path) -> None:
    evidence_root = tmp_path / "evidence"
    evidence_root.mkdir()
    write_package(evidence_root)
    client = FakeClient(SKILL_RESULT)

    await skill_tool(client, data_dir=tmp_path).handler(
        AigSkillScanInput(
            run_id="run-1",
            root=str(evidence_root),
            files=["scripts/../SKILL.md"],
            authorization_ref="ticket-42",
            max_wait_s=30.0,
        )
    )

    assert len(client.uploads) == 1
    with zipfile.ZipFile(io.BytesIO(client.uploads[0][1])) as archive:
        names = archive.namelist()
    assert names == ["SKILL.md"]
    assert all(".." not in name and "\\" not in name for name in names)


@pytest.mark.asyncio
async def test_skill_scan_upload_failure_records_a_terminal_failed_row(tmp_path: Path) -> None:
    evidence_root = tmp_path / "evidence"
    evidence_root.mkdir()
    files = write_package(evidence_root)
    client = FakeClient(SKILL_RESULT, upload_error=AigError("upload exploded"))
    audit = RecordingAudit()

    with pytest.raises(AigError):
        await skill_tool(client, data_dir=tmp_path, audit=audit).handler(
            AigSkillScanInput(
                run_id="run-1",
                root=str(evidence_root),
                files=files,
                authorization_ref="ticket-42",
                max_wait_s=30.0,
            )
        )

    assert client.bodies == []
    assert [event.action for event in audit.events] == ["active_scan.failed"]
    assert audit.events[0].detail == {"error_type": "AigError"}


@pytest.mark.asyncio
async def test_skill_scan_wraps_a_read_failure_without_leaking_the_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evidence_root = tmp_path / "evidence"
    evidence_root.mkdir()
    files = write_package(evidence_root)
    audit = RecordingAudit()

    def raise_oserror(self: Path) -> bytes:
        raise OSError(f"cannot read {self}")

    monkeypatch.setattr(Path, "read_bytes", raise_oserror)

    with pytest.raises(AigPackageError) as excinfo:
        await skill_tool(FakeClient(SKILL_RESULT), data_dir=tmp_path, audit=audit).handler(
            AigSkillScanInput(
                run_id="run-1",
                root=str(evidence_root),
                files=files,
                authorization_ref="ticket-42",
                max_wait_s=30.0,
            )
        )

    assert str(evidence_root) not in str(excinfo.value)
    assert "SKILL.md" not in str(excinfo.value)
    assert [event.action for event in audit.events] == ["active_scan.failed"]


@pytest.mark.asyncio
async def test_model_redteam_scan_saves_the_attachment_csv(tmp_path: Path) -> None:
    client = FakeClient(MODEL_RESULT)
    audit = RecordingAudit()

    output = await model_tool(client, data_dir=tmp_path, audit=audit).handler(
        AigScanInput(
            run_id="run-1",
            target=model_target(),
            scan_base_url="http://demo:8100/v1",
            authorization_ref="demo",
            max_wait_s=30.0,
        )
    )

    csv_path = tmp_path / "runs" / "run-1" / "aig" / f"{output.session_id}.csv"
    assert client.downloaded == [(output.session_id, ATTACHMENT)]
    assert csv_path.read_bytes() == CSV_BYTES
    assert [event.action for event in audit.events] == ["active_scan.started", "active_scan.completed"]
    details = json.dumps([event.detail for event in audit.events])
    assert ATTACHMENT not in details
    assert "completed,demo" not in details


@pytest.mark.asyncio
async def test_attachment_failure_keeps_the_scan_result(tmp_path: Path) -> None:
    client = FakeClient(MODEL_RESULT, download_error=AigUnavailable("boom"))
    audit = RecordingAudit()

    output = await model_tool(client, data_dir=tmp_path, audit=audit).handler(
        AigScanInput(
            run_id="run-1",
            target=model_target(),
            scan_base_url="http://demo:8100/v1",
            authorization_ref="demo",
            max_wait_s=30.0,
        )
    )

    assert output.reports
    assert not (tmp_path / "runs" / "run-1" / "aig" / f"{output.session_id}.csv").exists()
    assert [event.action for event in audit.events] == [
        "active_scan.started",
        "active_scan.attachment_failed",
        "active_scan.completed",
    ]
    failed = audit.events[1]
    assert failed.detail == {"reason": "AigUnavailable"}
    assert failed.scan_kind == "model_redteam_report"
    assert ATTACHMENT not in json.dumps([event.detail for event in audit.events])
    assert "boom" not in json.dumps([event.detail for event in audit.events])


@pytest.mark.asyncio
async def test_every_distinct_attachment_is_saved_with_the_first_name_unchanged(tmp_path: Path) -> None:
    client = FakeClient(TWO_MODEL_RESULT)
    audit = RecordingAudit()

    output = await model_tool(client, data_dir=tmp_path, audit=audit).handler(model_input())

    aig_dir = tmp_path / "runs" / "run-1" / "aig"
    assert client.downloaded == [
        (output.session_id, ATTACHMENT),
        (output.session_id, ATTACHMENT_2),
    ]
    assert (aig_dir / f"{output.session_id}.csv").read_bytes() == CSV_BYTES
    assert (aig_dir / f"{output.session_id}-2.csv").read_bytes() == CSV_BYTES
    assert [event.action for event in audit.events] == ["active_scan.started", "active_scan.completed"]


@pytest.mark.asyncio
async def test_downloads_are_capped_and_the_extra_attachment_is_audited(tmp_path: Path) -> None:
    client = FakeClient(THREE_MODEL_RESULT)
    audit = RecordingAudit()

    output = await model_tool(client, data_dir=tmp_path, audit=audit).handler(model_input())

    aig_dir = tmp_path / "runs" / "run-1" / "aig"
    assert client.downloaded == [
        (output.session_id, ATTACHMENT),
        (output.session_id, ATTACHMENT_2),
    ]
    assert (aig_dir / f"{output.session_id}.csv").read_bytes() == CSV_BYTES
    assert (aig_dir / f"{output.session_id}-2.csv").read_bytes() == CSV_BYTES
    assert not (aig_dir / f"{output.session_id}-3.csv").exists()
    assert [event.action for event in audit.events] == [
        "active_scan.started",
        "active_scan.attachment_failed",
        "active_scan.completed",
    ]
    assert audit.events[1].detail == {"reason": "too_many_attachments"}
    details = json.dumps([event.detail for event in audit.events])
    assert ATTACHMENT_3 not in details
    assert "demo-echo" not in details


@pytest.mark.asyncio
async def test_unreadable_result_audits_and_completes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    original = Path.read_text

    def fail_json(self: Path, *args: object, **kwargs: object) -> str:
        if self.suffix == ".json":
            raise OSError("unreadable")
        return original(self, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(Path, "read_text", fail_json)
    audit = RecordingAudit()

    output = await model_tool(FakeClient(MODEL_RESULT), data_dir=tmp_path, audit=audit).handler(model_input())

    assert output.reports
    assert [event.action for event in audit.events] == [
        "active_scan.started",
        "active_scan.attachment_failed",
        "active_scan.completed",
    ]
    assert audit.events[1].detail == {"reason": "unreadable_result"}


@pytest.mark.asyncio
async def test_csv_write_failure_audits_and_completes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_write(self: Path, data: bytes) -> int:
        raise OSError("disk full")

    monkeypatch.setattr(Path, "write_bytes", fail_write)
    audit = RecordingAudit()

    output = await model_tool(FakeClient(MODEL_RESULT), data_dir=tmp_path, audit=audit).handler(model_input())

    assert output.reports
    assert [event.action for event in audit.events] == [
        "active_scan.started",
        "active_scan.attachment_failed",
        "active_scan.completed",
    ]
    assert audit.events[1].detail == {"reason": "OSError"}


@pytest.mark.asyncio
async def test_client_without_download_degrades_instead_of_crashing(tmp_path: Path) -> None:
    inner = FakeClient(MODEL_RESULT)
    audit = RecordingAudit()
    tool = build_run_aig_scan_tool(
        client=cast(AigScanClient, NoDownloadClient(inner)),
        settings=AigSettings(),
        eval_model=EVAL,
        env={"JUDGE_KEY": "judge-token"},
        audit=audit,
        data_dir=str(tmp_path),
    )

    output = await tool.handler(model_input())

    assert output.reports
    assert not (tmp_path / "runs" / "run-1" / "aig" / f"{output.session_id}.csv").exists()
    assert [event.action for event in audit.events] == [
        "active_scan.started",
        "active_scan.attachment_failed",
        "active_scan.completed",
    ]
    assert audit.events[1].detail == {"reason": "download_unsupported"}


def test_model_and_skill_budgets_match_their_http_calls(tmp_path: Path) -> None:
    settings = AigSettings()
    skill = build_run_aig_skill_scan_tool(
        client=FakeClient(SKILL_RESULT),
        settings=settings,
        eval_model=EVAL,
        env={"JUDGE_KEY": "judge-token"},
        audit=noop_audit_sink,
        data_dir=str(tmp_path),
    )
    assert skill.timeout_s == settings.max_wait_s + 4 * settings.timeout_s + AIG_TOOL_TIMEOUT_MARGIN_S
    model = build_run_aig_scan_tool(
        client=FakeClient(SKILL_RESULT),
        settings=settings,
        eval_model=EVAL,
        env={"JUDGE_KEY": "judge-token"},
        audit=noop_audit_sink,
        data_dir=str(tmp_path),
    )
    assert (
        model.timeout_s
        == settings.max_wait_s + (3 + MAX_ATTACHMENTS) * settings.timeout_s + AIG_TOOL_TIMEOUT_MARGIN_S
    )
    agent = build_run_aig_agent_scan_tool(
        client=FakeClient(SKILL_RESULT),
        settings=settings,
        eval_model=EVAL,
        env={"JUDGE_KEY": "judge-token"},
        audit=noop_audit_sink,
        data_dir=str(tmp_path),
    )
    assert agent.timeout_s == settings.max_wait_s + 3 * settings.timeout_s + AIG_TOOL_TIMEOUT_MARGIN_S
