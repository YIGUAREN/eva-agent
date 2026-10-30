# tests/unit/deploy/test_compose.py
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
COMPOSE = ROOT / "deploy" / "compose.dev.yaml"
DOCKERFILE = ROOT / "demo_target" / "Dockerfile"


def services() -> dict:
    return yaml.safe_load(COMPOSE.read_text(encoding="utf-8"))["services"]


def test_aig_ports_are_loopback_only() -> None:
    stack = services()
    assert stack["aig-server"]["ports"] == ["127.0.0.1:8088:8088"]
    assert "ports" not in stack["aig-agent"]
    assert stack["aig-agent"]["expose"] == ["8000"]


def test_aig_images_share_a_pinned_release_tag() -> None:
    stack = services()
    server = stack["aig-server"]["image"]
    agent = stack["aig-agent"]["image"]
    assert server.startswith("zhuquelab/aig-server:")
    assert agent.startswith("zhuquelab/aig-agent:")
    server_tag = server.rsplit(":", 1)[1]
    agent_tag = agent.rsplit(":", 1)[1]
    assert server_tag == agent_tag
    assert re.fullmatch(r"v\d+\.\d+\.\d+", server_tag)
    assert "latest" not in (server_tag, agent_tag)


def test_aig_agent_has_the_sandbox_settings_it_needs() -> None:
    agent = services()["aig-agent"]
    assert agent["cap_add"] == ["SYS_ADMIN"]
    assert agent["security_opt"] == ["seccomp:unconfined"]
    assert agent["shm_size"] == "2gb"
    assert agent["healthcheck"]["test"]


def test_aig_server_points_at_the_agent_and_has_a_healthcheck() -> None:
    server = services()["aig-server"]
    assert "AIG_API_CHECKER_URL=http://aig-agent:8000" in server["environment"]
    assert server["depends_on"]["aig-agent"]["condition"] == "service_healthy"
    assert server["healthcheck"]["test"]


def test_demo_target_is_built_locally_and_bound_to_loopback() -> None:
    demo = services()["demo-target"]
    assert demo["build"]["context"] == "../demo_target"
    assert demo["ports"] == ["127.0.0.1:8100:8100"]
    assert demo["networks"] == ["eva-aig"]
    assert "eva-aig" in yaml.safe_load(COMPOSE.read_text(encoding="utf-8"))["networks"]


def test_demo_target_image_serves_the_fastapi_app() -> None:
    text = DOCKERFILE.read_text(encoding="utf-8")
    assert "FROM python:3.12-slim" in text
    assert "fastapi" in text
    assert "uvicorn" in text
    assert 'CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8100"]' in text
