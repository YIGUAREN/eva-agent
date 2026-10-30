from pathlib import Path

from eva_agent.config.settings import AigScanSettings, AigSettings
from eva_agent.domain.run import InputKind
from eva_agent.runtime.dependencies import build_runtime_dependencies
from tests.support.target_runtime import demo_target, runtime_config, target_budget


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


def test_runtime_denies_scan_when_eval_model_key_is_missing(tmp_path: Path) -> None:
    config = runtime_config(tmp_path)
    # Allow an active scan on the demo target so the decision reaches the eval-key
    # gate instead of stopping at "disabled" or "not_admin".
    Path(config.security.policy_path).write_text(
        """
active_scan_enabled: true
requester_is_admin: true
authorization_ref: demo
demo_targets:
  - name: demo
    origin: "http://127.0.0.1:8100"
allowlist: []
""".strip()
        + "\n",
        encoding="utf-8",
    )
    config = config.model_copy(
        update={
            "aig": AigSettings(scan=AigScanSettings(datasets=["d"], num_prompts=2, random_seed=1)),
            # The attack model's key is absent from `env`, while judge/agent keys
            # remain present so dependency construction succeeds.
            "models": config.models.model_copy(
                update={
                    "attack": config.models.attack.model_copy(update={"api_key_env": "ATTACK_API_KEY"}),
                }
            ),
        }
    )

    deps = build_runtime_dependencies(config, env={"FAKE_API_KEY": "x"})

    assert deps.scan_policy is not None
    decision = deps.scan_policy.decide(
        target=demo_target(),
        input_kind=InputKind.TARGET,
        probe_count=0,
        budget=target_budget(),
    )
    assert decision.allowed is False
    assert decision.reason == "missing_eval_model_key"
