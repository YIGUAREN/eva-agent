import socket
from pathlib import Path

import pytest

from eva_agent.config.settings import ConfigError
from eva_agent.domain.target import TargetKind, TargetSpec
from eva_agent.security.target_policy import (
    SecurityPolicy,
    SecurityPolicyError,
    TargetPolicy,
    is_denied,
    load_security_policy,
)


def resolver(*addresses: str):
    async def resolve(host: str, port: int) -> list[str]:
        return list(addresses)

    return resolve


def openai_target(base_url: str, name: str = "demo") -> TargetSpec:
    return TargetSpec(
        name=name,
        kind=TargetKind.OPENAI_COMPATIBLE,
        provider="demo",
        model="demo-echo",
        base_url=base_url,
    )


def policy() -> SecurityPolicy:
    return SecurityPolicy(demo_targets=[{"name": "demo", "origin": "http://127.0.0.1:8100"}])  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "address",
    [
        "0.0.0.0",
        "10.0.0.1",
        "100.64.0.1",
        "127.0.0.1",
        "169.254.169.254",
        "172.16.0.1",
        "192.168.1.1",
        "224.0.0.1",
        "240.0.0.1",
        "::",
        "::1",
        "fc00::1",
        "fe80::1",
        "ff02::1",
        "::ffff:127.0.0.1",
    ],
)
def test_denied_addresses(address: str):
    assert is_denied(address) is True


@pytest.mark.parametrize("address", ["8.8.8.8", "93.184.216.34", "2001:4860:4860::8888"])
def test_public_addresses_allowed(address: str):
    assert is_denied(address) is False


@pytest.mark.parametrize(
    "address",
    [
        "2002:7f00:1::",  # 6to4 embedding 127.0.0.1
        "2001::1",  # Teredo
        "64:ff9b::7f00:1",  # NAT64 well-known prefix embedding 127.0.0.1
        "64:ff9b:1::1",  # local-use NAT64
        "::ffff:0:127.0.0.1",  # IPv4-translated
        "::127.0.0.1",  # deprecated IPv4-compatible
        "fec0::1",  # deprecated site-local
    ],
)
def test_denied_ipv6_transition_addresses(address: str):
    assert is_denied(address) is True


@pytest.mark.parametrize("address", ["1.1.1.1", "2606:4700:4700::1111"])
async def test_public_addresses_still_validate(address: str):
    target = openai_target("https://api.example.com/v1")

    validated = await TargetPolicy(SecurityPolicy(), resolver=resolver(address)).validate(target)

    assert validated.addresses == (address,)


async def test_https_public_target_is_accepted():
    target = openai_target("https://api.example.com/v1")

    validated = await TargetPolicy(SecurityPolicy(), resolver=resolver("93.184.216.34")).validate(target)

    assert validated.scheme == "https"
    assert validated.port == 443
    assert validated.addresses == ("93.184.216.34",)
    assert validated.demo is False


async def test_demo_target_requires_exact_name_and_origin():
    validated = await TargetPolicy(policy(), resolver=resolver("127.0.0.1")).validate(
        openai_target("http://127.0.0.1:8100/v1")
    )
    assert validated.demo is True

    with pytest.raises(SecurityPolicyError, match="denied address"):
        await TargetPolicy(policy(), resolver=resolver("127.0.0.1")).validate(
            openai_target("http://127.0.0.1:8100/v1", name="other")
        )
    with pytest.raises(SecurityPolicyError, match="denied address"):
        await TargetPolicy(policy(), resolver=resolver("127.0.0.1")).validate(openai_target("http://127.0.0.1:9999/v1"))


async def test_any_denied_address_rejects_the_target():
    target = openai_target("https://api.example.com/v1")

    with pytest.raises(SecurityPolicyError, match="denied address"):
        await TargetPolicy(SecurityPolicy(), resolver=resolver("93.184.216.34", "127.0.0.1")).validate(target)


async def test_empty_resolution_is_rejected():
    with pytest.raises(SecurityPolicyError, match="resolved to no addresses"):
        await TargetPolicy(SecurityPolicy(), resolver=resolver()).validate(openai_target("https://api.example.com/v1"))


@pytest.mark.parametrize(
    "url",
    [
        "ftp://api.example.com/v1",
        "http://user:pass@api.example.com/v1",
        "http:///v1",
    ],
)
async def test_invalid_urls_are_rejected(url: str):
    with pytest.raises(SecurityPolicyError):
        await TargetPolicy(SecurityPolicy(), resolver=resolver("93.184.216.34")).validate(openai_target(url))


async def test_empty_userinfo_is_rejected():
    with pytest.raises(SecurityPolicyError, match="must not contain credentials"):
        await TargetPolicy(SecurityPolicy(), resolver=resolver("93.184.216.34")).validate(
            openai_target("http://@api.example.com/v1")
        )


async def test_resolution_failure_is_wrapped():
    async def fail(host: str, port: int) -> list[str]:
        raise socket.gaierror(-2, "Name or service not known")

    with pytest.raises(SecurityPolicyError, match="api.example.com:443") as excinfo:
        await TargetPolicy(SecurityPolicy(), resolver=fail).validate(openai_target("https://api.example.com/v1"))

    assert "https://" not in str(excinfo.value)


def test_load_security_policy(tmp_path: Path):
    path = tmp_path / "policy.yaml"
    path.write_text(
        """
active_scan_enabled: false
demo_targets:
  - name: demo
    origin: "http://127.0.0.1:8100"
allowlist: []
""",
        encoding="utf-8",
    )

    loaded = load_security_policy(path)

    assert loaded.demo_targets[0].name == "demo"
    assert loaded.allowlist == []


def test_load_security_policy_rejects_origin_with_empty_userinfo(tmp_path: Path):
    path = tmp_path / "policy.yaml"
    path.write_text(
        """
demo_targets:
  - name: demo
    origin: "http://@127.0.0.1:8100"
""",
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="credentials"):
        load_security_policy(path)


def test_load_security_policy_rejects_origin_without_port(tmp_path: Path):
    path = tmp_path / "policy.yaml"
    path.write_text(
        """
demo_targets:
  - name: demo
    origin: "http://127.0.0.1"
""",
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="origin must include"):
        load_security_policy(path)
