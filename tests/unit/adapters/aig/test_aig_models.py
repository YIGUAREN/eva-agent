import pytest

from eva_agent.adapters.aig.body import build_create_body
from eva_agent.adapters.aig.errors import AigResultError
from eva_agent.adapters.aig.models import parse_result
from eva_agent.config.settings import AigScanSettings, ModelConfig
from eva_agent.domain.target import TargetKind, TargetSpec

RAW = {
    "result": {
        "msgType": "json",
        "content": [
            {
                "modelName": "demo-echo",
                "baseTotal": 4,
                "total": 4,
                "jailbreak": 1,
                "score": 75.0,
                "errored": 0,
                "useless": 0,
                "extraBody": {
                    "vulnerabilityResults": [
                        {"vulnerability": "Harmful Content", "total": 2, "jailbreak": 1, "score": 50.0,
                         "asr": 0.5, "errored": 0}
                    ],
                    "attackMethodResults": [
                        {"attackMethod": "Direct", "total": 2, "jailbreak": 0, "score": 100.0,
                         "asr": 0.0, "errored": 0}
                    ],
                },
            }
        ],
    }
}


def test_parse_result_maps_names_and_counts() -> None:
    reports = parse_result(RAW)
    assert len(reports) == 1
    report = reports[0]
    assert report.model_name == "demo-echo"
    assert report.jailbreak == 1
    assert report.asr == 0.25
    assert report.extra_body.vulnerability_results[0].vulnerability == "Harmful Content"
    assert report.extra_body.attack_method_results[0].attack_method == "Direct"


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"result": {"msgType": "text", "content": "log"}},
        {"result": {"msgType": "json", "content": []}},
        {"result": {"msgType": "json", "content": [{"modelName": "m", "total": 1, "jailbreak": 2}]}},
    ],
)
def test_parse_result_fails_closed(payload) -> None:
    with pytest.raises(AigResultError):
        parse_result(payload)


def test_build_body_uses_aig_container_url_and_tokens() -> None:
    target = TargetSpec(
        name="demo", kind=TargetKind.OPENAI_COMPATIBLE, provider="demo", model="demo-echo",
        base_url="http://127.0.0.1:8100/v1",
    )
    eval_model = ModelConfig(
        provider="openai-compatible", model="judge", base_url="https://judge.example/v1",
        api_key_env="JUDGE_KEY",
    )
    body = build_create_body(
        target=target,
        scan_base_url="http://demo-target:8100/v1",
        eval_model=eval_model,
        eval_token="judge-secret",
        target_token="target-secret",
        scan=AigScanSettings(datasets=["d"], num_prompts=3, random_seed=7),
    )
    assert body["type"] == "model_redteam_report"
    content = body["content"]
    assert content["model"] == [
        {"model": "demo-echo", "token": "target-secret", "base_url": "http://demo-target:8100/v1"}
    ]
    assert content["eval_model"]["base_url"] == "https://judge.example/v1"
    assert content["dataset"] == {"dataFile": ["d"], "numPrompts": 3, "randomSeed": 7}
    assert content["techniques"] == []
