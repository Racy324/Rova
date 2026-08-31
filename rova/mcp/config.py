from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import os
from pathlib import Path
import re
from urllib.parse import urlsplit

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised on Python 3.10
    import tomli as tomllib


class MCPConfigError(ValueError):
    pass


@dataclass(frozen=True)
class MCPServerSettings:
    server_id: str
    transport: str
    include_tools: tuple[str, ...]
    command: str | None = None
    args: tuple[str, ...] = ()
    url: str | None = None
    headers: Mapping[str, str] = field(default_factory=dict, repr=False)
    environment: Mapping[str, str] = field(default_factory=dict, repr=False)


@dataclass(frozen=True)
class MCPSettings:
    servers: tuple[MCPServerSettings, ...] = ()


_ENV_REFERENCE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def load_mcp_settings(path: Path | None, *, environment: Mapping[str, str] | None = None) -> MCPSettings:
    if path is None:
        return MCPSettings()
    try:
        document = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise MCPConfigError(f"MCP config file does not exist: {path}") from error
    except tomllib.TOMLDecodeError as error:
        raise MCPConfigError(f"invalid MCP config: {error}") from error
    raw_servers = document.get("mcp_servers", {})
    if not isinstance(raw_servers, dict):
        raise MCPConfigError("mcp_servers must be a table")
    source = os.environ if environment is None else environment
    servers: list[MCPServerSettings] = []
    for server_id, raw in raw_servers.items():
        if not isinstance(server_id, str) or not isinstance(raw, dict):
            raise MCPConfigError("each MCP server must be a table with a string name")
        if raw.get("enabled", False) is not True:
            continue
        servers.append(_parse_server(server_id, raw, source))
    return MCPSettings(tuple(servers))


def safe_stdio_environment(host_environment: Mapping[str, str], explicit_environment: Mapping[str, str]) -> dict[str, str]:
    safe = {
        name: value
        for name, value in host_environment.items()
        if name.upper() in {"PATH", "SYSTEMROOT", "WINDIR", "COMSPEC", "PATHEXT", "HOME", "USERPROFILE", "TMP", "TEMP"}
    }
    safe.update(explicit_environment)
    return safe


def _parse_server(server_id: str, raw: Mapping[str, object], environment: Mapping[str, str]) -> MCPServerSettings:
    transport = raw.get("transport")
    if transport not in {"stdio", "streamable_http"}:
        raise MCPConfigError(f"MCP server {server_id!r} has unsupported transport")
    include_tools = _string_list(raw.get("include_tools"), f"MCP server {server_id!r} include_tools")
    if not include_tools or len(set(include_tools)) != len(include_tools):
        raise MCPConfigError(f"MCP server {server_id!r} include_tools must be a non-empty unique string list")
    headers = _resolve_mapping(raw.get("headers", {}), environment, f"MCP server {server_id!r} headers")
    explicit_environment = _resolve_mapping(raw.get("env", {}), environment, f"MCP server {server_id!r} env")
    if transport == "stdio":
        command = raw.get("command")
        if not isinstance(command, str) or not command.strip():
            raise MCPConfigError(f"MCP server {server_id!r} stdio command is required")
        return MCPServerSettings(server_id, transport, tuple(include_tools), command=command, args=tuple(_string_list(raw.get("args", []), "stdio args")), headers=headers, environment=explicit_environment)
    url = raw.get("url")
    if not isinstance(url, str) or not _safe_http_url(url):
        raise MCPConfigError(f"MCP server {server_id!r} requires a safe streamable_http URL")
    return MCPServerSettings(server_id, transport, tuple(include_tools), url=url, headers=headers, environment=explicit_environment)


def _string_list(value: object, label: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) and item.strip() for item in value):
        raise MCPConfigError(f"{label} must be a string list")
    return list(value)


def _resolve_mapping(value: object, environment: Mapping[str, str], label: str) -> dict[str, str]:
    if not isinstance(value, dict) or not all(isinstance(key, str) and isinstance(item, str) for key, item in value.items()):
        raise MCPConfigError(f"{label} must be a string mapping")
    resolved: dict[str, str] = {}
    for key, item in value.items():
        matches = tuple(_ENV_REFERENCE.finditer(item))
        if not matches:
            raise MCPConfigError(f"{label} values must use environment references")
        rendered = item
        for match in matches:
            secret = environment.get(match.group(1))
            if not secret:
                raise MCPConfigError("MCP config environment variable is not set")
            rendered = rendered.replace(match.group(0), secret)
        resolved[key] = rendered
    return resolved


def _safe_http_url(value: str) -> bool:
    parsed = urlsplit(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc) and not any((parsed.username, parsed.password, parsed.query, parsed.fragment))
