from pathlib import Path

from eva_agent.config.settings import AigScanSettings, AigSettings
from eva_agent.runtime.dependencies import build_runtime_dependencies
from tests.support.target_runtime import runtime_config


def test_runtime_builds_scan_policy_when_aig_is_configured(tmp_path: Path) -> None:
    config = runtime_config(tmp_path).model_copy(
        update={"aig": AigSettings(scan=AigScanSettings(datasets=["d"], num_prompts=2, random_seed=1))}
    )
    deps = build_runtime_dependencies(config, env={"FAKE_API_KEY": "x"})
    assert deps.scan_policy is not None
    assert deps.aig_configured is True
    assert "run_aig_scan" in deps.tools.names()


def test_runtime_without_aig_has_no_scan_policy(tmp_path: Path) -> None:
    deps = build_runtime_dependencies(runtime_config(tmp_path), env={"FAKE_API_KEY": "x"})
    assert deps.scan_policy is None
    assert deps.aig_configured is False
    assert "run_aig_scan" not in deps.tools.names()
