from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from rova.config import load_project_env
from rova.mcp.client import MCPServerConfig


@dataclass(frozen=True)
class WebSettings:
    """Configuration for the direct or MCP public-web backend."""

    search_endpoint: str = "https://html.duckduckgo.com/html/"
    backend: str = "direct"
    mcp_url: str = "https://mcp.tavily.com/mcp/"
    tavily_api_key: str | None = field(default=None, repr=False)

    @classmethod
    def from_env(
        cls,
        env: Mapping[str, str] | None = None,
        *,
        dotenv_path: Path | None = None,
    ) -> "WebSettings":
        source = load_project_env(env, dotenv_path=dotenv_path)
        return cls(
            search_endpoint=source.get("ROVA_RESEARCH_SEARCH_ENDPOINT", cls.search_endpoint),
            backend=source.get("ROVA_RESEARCH_BACKEND", cls.backend),
            mcp_url=source.get("ROVA_RESEARCH_MCP_URL", cls.mcp_url),
            tavily_api_key=source.get("TAVILY_API_KEY") or None,
        )

    def mcp_server_config(self) -> MCPServerConfig:
        if not self.tavily_api_key:
            raise ValueError("TAVILY_API_KEY is required for the MCP web backend")
        return MCPServerConfig("tavily", "streamable_http", self.mcp_url, {"Authorization": f"Bearer {self.tavily_api_key}"})
