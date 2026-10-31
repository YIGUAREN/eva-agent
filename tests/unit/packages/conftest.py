"""Fixtures that build a completed run for grader-package tests."""

from __future__ import annotations

import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import pytest

from eva_agent.adapters.inputs.paste import write_paste_workspace
from eva_agent.adapters.openjudge.cache import GradingCache, cache_key
from eva_agent.domain.plan import Dimension
from eva_agent.domain.result import GraderRunOutput, GraderSampleOutcome, MergedResult
from eva_agent.domain.run import RunStatus
from eva_agent.graph.experts.quality import aggregate_outcomes, summarize_quality
from eva_agent.packages.inputs import collect_package_input
from eva_agent.packages.manifest import (
    FileEntry,
    PackageManifest,
    canonical_json,
    sha256_bytes,
    sha256_file,
    write_manifest,
    write_zip,
)
from eva_agent.report.render import write_report
from eva_agent.storage.db import create_engine, create_session_factory, init_db
from eva_agent.storage.models import utcnow
from eva_agent.storage.repository import RunRepository
from tests.support.runtime import budget, fake_config, paste_input, sample_plan


@pytest.fixture
def archive_dir(tmp_path: Path) -> Path:
    """The output directory a package test writes its archive into."""
    return tmp_path / "packages"


@pytest.fixture
async def write_run(tmp_path: Path):
    """A completed paste run whose report was derived from its own grading cache.

    The report is computed from the same rows ``collect_package_input`` puts in
    the package, so a faithful replay must reproduce it exactly.
    """

    async def _make(*, truncated_dimensions: list[str] | None = None):
        config = fake_config(tmp_path)
        engine = create_engine(config.storage.url)
        await init_db(engine)
        factory = create_session_factory(engine)
        runs = RunRepository(factory)
        run_input = paste_input()
        plan = sample_plan()
        await runs.create("run-1", run_input, budget())
        await runs.set_plan("run-1", plan)
        await runs.set_status("run-1", RunStatus.COMPLETED)
        assert run_input.paste is not None
        bundle = write_paste_workspace("run-1", run_input.paste, config.storage.data_dir)
        run_row = await runs.get("run-1")
        assert run_row is not None
        inputs = await collect_package_input(config=config, run_row=run_row)
        dimension = next(item for item in plan.dimensions if item.dimension is Dimension.QUALITY)
        specs = [spec for spec in dimension.graders if spec.mode == "pointwise"]
        cache = GradingCache(Path(config.storage.data_dir), "run-1")
        outcomes: list[GraderSampleOutcome] = []
        for spec in specs:
            for index, row in enumerate(inputs.rows):
                outcome = GraderSampleOutcome(
                    grader=spec.name, sample_index=index, score=0.8, raw_score=4.0, reason="ok"
                )
                cache.write(spec.name, cache_key(spec, row), outcome)
                outcomes.append(outcome)
        normalized = aggregate_outcomes(
            GraderRunOutput(outcomes=outcomes),
            specs=specs,
            rows=inputs.rows,
            evidence=bundle,
            source_indexes=inputs.source_indexes,
        )
        truncated = "budget" if truncated_dimensions else None
        merged = MergedResult(
            status=RunStatus.COMPLETED,
            results=[
                summarize_quality(
                    normalized,
                    sample_count=len(inputs.rows),
                    spec_count=len(specs),
                    truncated_reason=truncated,
                )
            ],
        )
        write_report(
            data_dir=config.storage.data_dir,
            run_id="run-1",
            plan=plan,
            merged=merged,
            generated_at=datetime.now(UTC),
        )
        await engine.dispose()
        return config, run_row

    return _make


@pytest.fixture
def write_archive(tmp_path: Path):
    """Build an archive from explicit members and an independent manifest file list.

    ``declared`` lets a test list a member in the manifest that the archive does not
    contain, so the extraction layer's missing-member check is exercised.
    """

    def _make(
        members: dict[str, bytes],
        *,
        declared: list[str] | None = None,
        kind: Literal["grader_package", "run_bundle"] = "grader_package",
    ) -> Path:
        names = list(members) if declared is None else list(declared)
        manifest = PackageManifest(
            kind=kind,
            run_id="run-1",
            created_at=utcnow().isoformat(),
            report_version=4,
            files={
                name: FileEntry(
                    sha256=sha256_bytes(members.get(name, b"")),
                    size=len(members.get(name, b"")),
                )
                for name in names
            },
        )
        archive = tmp_path / "package.zip"
        with zipfile.ZipFile(archive, "w") as handle:
            handle.writestr("manifest.json", canonical_json(manifest.model_dump(mode="json")))
            for name, data in members.items():
                handle.writestr(name, data)
        return archive

    return _make


@pytest.fixture
def write_run_archive(tmp_path: Path):
    """A structurally valid archive whose payload changed after the manifest hashed it."""

    def _make() -> Path:
        root = tmp_path / "tampered"
        (root / "snapshots").mkdir(parents=True)
        members = {
            "README.md": b"# Grader package\n",
            "input.json": b'{"rows":[]}',
            "plan.json": b'{"objective":"o"}',
            "report.json": b'{"version":4}',
            "snapshots/gradings.json": b'{"outcomes":[]}',
        }
        for name, data in members.items():
            (root / name).write_bytes(data)
        manifest = PackageManifest(
            kind="grader_package",
            run_id="run-1",
            created_at=utcnow().isoformat(),
            report_version=4,
            files={
                name: FileEntry(sha256=sha256_file(root / name), size=len(data))
                for name, data in members.items()
            },
        )
        write_manifest(root, manifest)
        # Change the payload after it was hashed; extraction must reject it.
        (root / "report.json").write_bytes(b'{"version":4} ')
        return write_zip(root, tmp_path / "tampered.zip", manifest=manifest)

    return _make
