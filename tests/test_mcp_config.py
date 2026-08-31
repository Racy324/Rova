from __future__ import annotations

from pathlib import Path

import pytest

from rova.mcp.config import MCPConfigError, load_mcp_settings, safe_stdio_environment
from rova.app.settings import AppSettings


def test_no_explicit_mcp_config_disables_mcp() -> None:
    assert load_mcp_settings(None, environment={}).servers == ()


def test_config_parses_include_tools_and_explicit_secret_reference(tmp_path: Path) -> None:
    path = tmp_path / "mcp.toml"
    path.write_text(
        """[mcp_servers.github]
enabled = true
transport = "stdio"
command = "uvx"
args = ["github-mcp"]
env = { GITHUB_TOKEN = "${MCP_TOKEN}" }
include_tools = ["search_code", "create_issue"]
""",
        encoding="utf-8",
    )

    settings = load_mcp_settings(path, environment={"MCP_TOKEN": "configured-secret"})

    assert settings.servers[0].server_id == "github"
    assert settings.servers[0].include_tools == ("search_code", "create_issue")
    assert settings.servers[0].environment == {"GITHUB_TOKEN": "configured-secret"}
    assert "configured-secret" not in repr(settings)


def test_config_rejects_missing_secret_without_leaking_name_value(tmp_path: Path) -> None:
    path = tmp_path / "mcp.toml"
    path.write_text(
        """[mcp_servers.github]
enabled = true
transport = "streamable_http"
url = "https://mcp.example.test/mcp"
headers = { Authorization = "Bearer ${MISSING_TOKEN}" }
include_tools = ["search_code"]
""",
        encoding="utf-8",
    )

    with pytest.raises(MCPConfigError, match="environment variable is not set") as captured:
        load_mcp_settings(path, environment={})

    assert "MISSING_TOKEN" not in str(captured.value)


def test_safe_stdio_environment_excludes_host_secrets_and_keeps_explicit_values() -> None:
    environment = safe_stdio_environment(
        {"PATH": "safe-path", "OPENAI_API_KEY": "provider-secret", "ROVA_FLAG": "x", "OTHER_API_KEY": "other"},
        {"MCP_TOKEN": "configured-secret"},
    )

    assert environment["MCP_TOKEN"] == "configured-secret"
    assert environment["PATH"] == "safe-path"
    assert "OPENAI_API_KEY" not in environment
    assert "ROVA_FLAG" not in environment
    assert "OTHER_API_KEY" not in environment


def test_app_settings_reads_only_explicit_mcp_config_path() -> None:
    settings = AppSettings.from_env({"ROVA_MCP_CONFIG": "configs/mcp.toml"})

    assert settings.mcp_config_path == Path("configs/mcp.toml")
