"""Factories for Rova's configured public-web backends."""

from .http import DirectHttpFetchBackend, DuckDuckGoHtmlSearchBackend
from .settings import WebSettings


def create_web_backends(settings: WebSettings):
    if settings.backend == "direct":
        return DuckDuckGoHtmlSearchBackend(settings.search_endpoint), DirectHttpFetchBackend()
    if settings.backend == "mcp":
        from rova.mcp.client import MCPClient

        from .mcp_backends import McpWebFetchBackend, McpWebSearchBackend

        config = settings.mcp_server_config()
        return McpWebSearchBackend(lambda: MCPClient(config)), McpWebFetchBackend(lambda: MCPClient(config))
    raise ValueError(f"Unsupported web backend: {settings.backend}")
