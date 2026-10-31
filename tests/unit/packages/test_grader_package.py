import pytest

from eva_agent.domain.plan import Dimension
from eva_agent.packages.grader_package import build_grader_package, read_grader_package
from eva_agent.packages.manifest import PackageError


async def test_build_grader_package_writes_every_declared_member(write_run, archive_dir):
    config, run_row = await write_run()
    archive, manifest = await build_grader_package(config=config, run_row=run_row, out_dir=archive_dir)
    assert manifest.kind == "grader_package"
    assert manifest.report_version == 4
    assert sorted(manifest.files) == [
        "README.md",
        "input.json",
        "plan.json",
        "report.json",
        "snapshots/gradings.json",
    ]
    package = read_grader_package(archive)
    assert package.plan.dimensions[0].dimension is Dimension.QUALITY
    assert package.inputs.rows[0]["query"] == "q"
    assert package.report["version"] == 4
    assert "usage" in package.report


async def test_build_grader_package_marks_a_truncated_run_unreplayable(write_run, archive_dir):
    config, run_row = await write_run(truncated_dimensions=["quality"])
    _, manifest = await build_grader_package(config=config, run_row=run_row, out_dir=archive_dir)
    assert manifest.replayable is False
    assert any("budget" in blocker for blocker in manifest.replay_blockers)


def test_read_grader_package_rejects_a_tampered_archive(write_run_archive):
    archive = write_run_archive()
    with pytest.raises(PackageError):
        read_grader_package(archive)
