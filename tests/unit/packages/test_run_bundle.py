import json
import zipfile

from eva_agent.packages.run_bundle import build_run_bundle
from eva_agent.report.render import REPORT_VERSION
from eva_agent.runtime.artifacts import artifact_file, run_paths
from eva_agent.storage.db import create_engine, create_session_factory
from eva_agent.storage.reports import ArtifactRepository, ReportRepository
from eva_agent.storage.repository import RunRepository


async def _published_run(write_run):
    """The conftest run, published, with a stored report and two artifacts on disk."""
    config, run_row = await write_run()
    factory = create_session_factory(create_engine(config.storage.url))
    paths = run_paths(config.storage.data_dir, run_row.id)
    payload = json.loads(paths.report_json.read_text(encoding="utf-8"))
    payload = {**payload, "api_key": "sk-live-123"}
    markdown = f"# 报告\n- 报告文件：{paths.report_json}\n"
    # ``set_public`` returns the freshly loaded row; the fixture's row predates publication.
    run_row = await RunRepository(factory).set_public(run_row.id, True)
    await ReportRepository(factory).upsert(run_row.id, REPORT_VERSION, payload, markdown)
    artifacts = ArtifactRepository(factory)
    for name, visibility in (("public.json", "public"), ("secret.json", "internal")):
        target = artifact_file(config.storage.data_dir, run_row.id, name)
        target.write_text("{}", encoding="utf-8")
        await artifacts.add(
            artifact_id=f"art-{name}",
            run_id=run_row.id,
            name=name,
            sha256="a" * 64,
            size_bytes=2,
            content_type="application/json",
            visibility=visibility,
        )
    return config, run_row, factory


async def test_bundle_carries_only_public_artifacts(write_run, tmp_path) -> None:
    config, run_row, factory = await _published_run(write_run)
    report = await ReportRepository(factory).get(run_row.id)
    artifacts = await ArtifactRepository(factory).list_public(run_row.id)
    archive, manifest = await build_run_bundle(
        config=config,
        run_row=run_row,
        report=report,
        artifacts=artifacts,
        out_dir=tmp_path / "out",
    )
    with zipfile.ZipFile(archive) as bundle:
        names = set(bundle.namelist())
    assert manifest.kind == "run_bundle"
    assert manifest.replayable is False
    assert {"manifest.json", "run.json", "report.json", "report.md"} <= names
    assert "artifacts/public.json" in names
    assert not any("secret" in name for name in names)


async def test_bundle_redacts_paths_and_secrets(write_run, tmp_path) -> None:
    config, run_row, factory = await _published_run(write_run)
    report = await ReportRepository(factory).get(run_row.id)
    artifacts = await ArtifactRepository(factory).list_public(run_row.id)
    archive, _ = await build_run_bundle(
        config=config,
        run_row=run_row,
        report=report,
        artifacts=artifacts,
        out_dir=tmp_path / "out",
    )
    with zipfile.ZipFile(archive) as bundle:
        markdown = bundle.read("report.md").decode("utf-8")
        payload = json.loads(bundle.read("report.json"))
    assert str(config.storage.data_dir) not in markdown
    assert "api_key" not in payload
    assert "sk-live-123" not in json.dumps(payload)
