from pathlib import Path

from eva_agent.runtime.dependencies import build_runtime_dependencies
from tests.support.target_runtime import runtime_config


def test_runtime_dependencies_wire_target_tools(tmp_path: Path) -> None:
    config = runtime_config(tmp_path, timeout_s=12.5)

    deps = build_runtime_dependencies(config, env={"FAKE_API_KEY": "test-key"})

    assert set(deps.targets) == {"demo"}
    assert deps.policy is not None
    assert deps.target_timeout_s == 12.5
    assert deps.max_response_bytes == 1_048_576
    assert deps.token_encoding == "cl100k_base"
    assert {"call_target", "get_pricing"} <= set(deps.tools.names())
