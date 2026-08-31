from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
import json
import re
from typing import Any

from rova.ai.messages import TextBlock
from rova.ai.tools import Tool
from rova.agent_core.tools import AgentTool, AgentToolResult, ToolExecutionError

from .client import MCPCallResult, MCPToolDefinition


class MCPToolAdapterError(ValueError):
    pass


MCPCall = Callable[[str, dict[str, Any]], Awaitable[MCPCallResult]]


def build_mcp_tools(server_id: str, definitions: Sequence[MCPToolDefinition], *, include_tools: Sequence[str], call_tool: MCPCall) -> list[AgentTool]:
    included = [definition for definition in definitions if definition.name in include_tools]
    names: dict[str, str] = {}
    for definition in included:
        public_name = f"mcp_{_normalize(server_id)}_{_normalize(definition.name)}"
        existing = names.get(public_name)
        if existing is not None:
            raise MCPToolAdapterError(f"normalization collision: {existing!r} and {definition.name!r}")
        names[public_name] = definition.name
    return [_adapter(server_id, definition, f"mcp_{_normalize(server_id)}_{_normalize(definition.name)}", call_tool) for definition in included]


def _adapter(server_id: str, definition: MCPToolDefinition, public_name: str, call_tool: MCPCall) -> AgentTool:
    async def execute(_tool_call_id: str, arguments: dict) -> AgentToolResult:
        result = await call_tool(definition.name, arguments)
        if result.is_error:
            raise ToolExecutionError("MCP tool returned an error", metadata={"origin": "mcp", "server_id": server_id, "raw_tool_name": definition.name, "public_name": public_name})
        rendered = "\n".join(result.text_blocks)
        if result.structured_content is not None:
            rendered = f"{rendered}\n\nStructured content:\n{json.dumps(result.structured_content, ensure_ascii=False, default=str)[:12000]}".strip()
        return AgentToolResult([TextBlock(rendered or "MCP tool returned no content.")])
    metadata = {"origin": "mcp", "server_id": server_id, "raw_tool_name": definition.name, "public_name": public_name}
    return AgentTool(Tool(public_name, definition.description, input_schema=definition.input_schema), execute, metadata)


def _normalize(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    if not normalized:
        raise MCPToolAdapterError("server and tool names must contain letters or digits")
    return normalized
