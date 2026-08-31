from __future__ import annotations

import asyncio
import socket
import sys
from pathlib import Path

import pytest

from rova.mcp.client import MCPClient, MCPConnectionError, MCPServerConfig


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.mark.asyncio
async def test_fake_streamable_http_mcp_discovery_call_and_close(tmp_path: Path):
    port = _free_port()
    server = tmp_path / "fake_mcp_http_server.py"
    server.write_text(
        """from mcp.server import MCPServer
server = MCPServer('fake-http')
@server.tool()
def echo(text: str) -> str:
    return f'echo:{text}'
server.run('streamable-http', host='127.0.0.1', port=%d)
""" % port,
        encoding="utf-8",
    )
    process = await asyncio.create_subprocess_exec(sys.executable, str(server))
    config = MCPServerConfig("fake-http", "streamable_http", f"http://127.0.0.1:{port}/mcp")
    client = MCPClient(config)
    try:
        for _ in range(30):
            try:
                await client.initialize()
                break
            except MCPConnectionError:
                await asyncio.sleep(0.1)
        else:
            pytest.fail("fake HTTP MCP server did not become ready")
        tools = await client.list_tools()
        result = await client.call_tool("echo", {"text": "hello"})
        assert [tool.name for tool in tools] == ["echo"]
        assert result.text_blocks == ["echo:hello"]
    finally:
        await client.close()
        process.terminate()
        await process.wait()
