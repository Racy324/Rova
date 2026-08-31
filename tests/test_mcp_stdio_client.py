from __future__ import annotations

import sys
from pathlib import Path

import pytest

from rova.mcp.client import MCPClient, MCPServerConfig


@pytest.mark.asyncio
async def test_fake_stdio_mcp_discovery_call_and_close(tmp_path: Path):
    server = tmp_path / "fake_mcp_server.py"
    server.write_text(
        """from mcp.server import MCPServer
server = MCPServer('fake')
@server.tool()
def echo(text: str) -> str:
    return f'echo:{text}'
server.run('stdio')
""",
        encoding="utf-8",
    )
    config = MCPServerConfig("fake", "stdio", command=sys.executable, args=(str(server),))

    client = MCPClient(config)
    await client.initialize()
    tools = await client.list_tools()
    result = await client.call_tool("echo", {"text": "hello"})
    await client.close()

    assert [tool.name for tool in tools] == ["echo"]
    assert result.text_blocks == ["echo:hello"]
