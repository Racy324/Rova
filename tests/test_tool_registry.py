from __future__ import annotations

import pytest

from rova.ai.messages import ToolCall
from rova.ai.tools import Tool, validate_tool_arguments
from rova.agent_core.tools import AgentTool, AgentToolResult, ToolRegistry


def _tool(name: str) -> AgentTool:
    async def execute(_id: str, _args: dict) -> AgentToolResult:
        return AgentToolResult([])
    return AgentTool(Tool(name, name, {}), execute)


def test_full_json_schema_is_preserved_and_validated() -> None:
    schema = {"type": "object", "properties": {"urls": {"type": "array", "items": {"type": "string"}}}, "required": ["urls"], "additionalProperties": False}
    tool = Tool("mcp_extract", "extract", input_schema=schema)
    assert tool.input_schema == schema
    assert validate_tool_arguments(tool, {"urls": ["https://example.test"]})["urls"] == ["https://example.test"]
    with pytest.raises(ValueError, match="validation failed"):
        validate_tool_arguments(tool, {"urls": [1]})


@pytest.mark.asyncio
async def test_register_tools_is_atomic_when_a_batch_collides() -> None:
    registry = ToolRegistry([_tool("native")])
    with pytest.raises(ValueError, match="duplicate tool name"):
        registry.register_tools([_tool("mcp_one"), _tool("native")])
    assert [schema.name for schema in registry.schemas] == ["native"]
    assert registry.get("mcp_one") is None


def test_register_tools_exposes_a_complete_batch() -> None:
    registry = ToolRegistry([])
    registry.register_tools([_tool("mcp_one"), _tool("mcp_two")])
    assert [schema.name for schema in registry.schemas] == ["mcp_one", "mcp_two"]


def test_registry_rejects_duplicate_initial_tools_instead_of_silently_overwriting() -> None:
    with pytest.raises(ValueError, match="duplicate tool name"):
        ToolRegistry([_tool("duplicate"), _tool("duplicate")])


def test_registry_only_exposes_registration_lookup_and_schemas() -> None:
    native = _tool("native")
    registry = ToolRegistry([native])

    assert registry.get("native") is native
    assert registry.get("missing") is None
    assert not hasattr(registry, "execute")
