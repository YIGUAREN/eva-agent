"""SSRF checks for online targets (default deny for private address space)."""

from __future__ import annotations

import asyncio
import ipaddress
import socket
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from eva_agent.config.settings import ConfigError
from eva_agent.domain.target import TargetKind, TargetSpec

AddressResolver = Callable[[str, int], Awaitable[list[str]]]

_DENIED_V4 = tuple(
    ipaddress.ip_network(network)
    for network in (
        "0.0.0.0/8",
        "10.0.0.0/8",
        "100.64.0.0/10",
        "127.0.0.0/8",
        "169.254.0.0/16",
        "172.16.0.0/12",
        "192.168.0.0/16",
        "224.0.0.0/4",
        "240.0.0.0/4",
    )
)
_DENIED_V6 = tuple(
    ipaddress.ip_network(network) for network in ("::/128", "::1/128", "fc00::/7", "fe80::/10", "ff00::/8")
)


class SecurityPolicyError(ValueError):
    """Raised when a target violates the SSRF policy."""


class DemoTarget(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    origin: str = Field(min_length=1)


class SecurityPolicy(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    active_scan_enabled: bool = False
    demo_targets: list[DemoTarget] = Field(default_factory=list)
    allowlist: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_origins(self) -> SecurityPolicy:
        for item in self.demo_targets:
            parse_origin(item.origin)
        return self


@dataclass(frozen=True)
class ValidatedTarget:
    url: str
    scheme: str
    host: str
    port: int
    addresses: tuple[str, ...]
    demo: bool


def parse_origin(origin: str) -> tuple[str, str, int]:
    parsed = urlparse(origin)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("demo target origin must include scheme, host, and port")
    if parsed.username or parsed.password:
        raise ValueError("demo target origin must not include credentials")
    if parsed.path not in ("", "/") or parsed.query or parsed.fragment:
        raise ValueError("demo target origin must be scheme://host:port only")
    try:
        port = parsed.port
    except ValueError:
        port = None
    if port is None:
        raise ValueError("demo target origin must include scheme, host, and port")
    return parsed.scheme, parsed.hostname.lower(), port


def load_security_policy(path: Path) -> SecurityPolicy:
    if not path.is_file():
        raise ConfigError(f"security policy file not found: {path}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"security policy must be a mapping: {path}")
    try:
        return SecurityPolicy.model_validate(raw)
    except ValidationError as exc:
        details = "; ".join(
            f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}"
            for error in exc.errors(include_input=False, include_url=False)
        )
        raise ConfigError(f"invalid security policy in {path}: {details}") from None


async def resolve_addresses(host: str, port: int) -> list[str]:
    loop = asyncio.get_running_loop()
    infos = await loop.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    return sorted({str(info[4][0]).split("%")[0] for info in infos})


def is_denied(address: str) -> bool:
    ip = ipaddress.ip_address(address)
    if isinstance(ip, ipaddress.IPv6Address):
        if ip.ipv4_mapped is not None:
            return is_denied(str(ip.ipv4_mapped))
        return any(ip in network for network in _DENIED_V6)
    return any(ip in network for network in _DENIED_V4)


class TargetPolicy:
    def __init__(self, policy: SecurityPolicy, *, resolver: AddressResolver = resolve_addresses) -> None:
        self._policy = policy
        self._resolver = resolver

    async def validate(self, target: TargetSpec) -> ValidatedTarget:
        url = target.base_url if target.kind is TargetKind.OPENAI_COMPATIBLE else target.url
        if not url:
            raise SecurityPolicyError("target has no URL")

        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"}:
            raise SecurityPolicyError("target scheme must be http or https")
        if parsed.username or parsed.password:
            raise SecurityPolicyError("target URL must not contain credentials")
        if not parsed.hostname:
            raise SecurityPolicyError("target URL must include a host")
        try:
            explicit_port = parsed.port
        except ValueError:
            raise SecurityPolicyError("target URL has an invalid port") from None
        port = explicit_port or (443 if parsed.scheme == "https" else 80)
        host = parsed.hostname.lower()

        demo = any(
            item.name == target.name and parse_origin(item.origin) == (parsed.scheme, host, port)
            for item in self._policy.demo_targets
        )

        addresses = await self._resolver(host, port)
        if not addresses:
            raise SecurityPolicyError(f"target {host}:{port} resolved to no addresses")

        if not demo:
            for address in addresses:
                try:
                    denied = is_denied(address)
                except ValueError:
                    raise SecurityPolicyError(f"target {host}:{port} resolved to an invalid address") from None
                if denied:
                    raise SecurityPolicyError(f"target {host}:{port} resolved to a denied address")

        return ValidatedTarget(
            url=url,
            scheme=parsed.scheme,
            host=host,
            port=port,
            addresses=tuple(addresses),
            demo=demo,
        )
