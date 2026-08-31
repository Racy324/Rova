"""Copy this file into ~/.rova/extensions/ or <workspace>/.rova/extensions/ to load it."""

from rova.ai.messages import TextBlock
from rova.ai.tools import Tool
from rova.agent_core.tools import AgentTool, AgentToolResult
from rova.app.extensions import ContextContribution


async def _say_hello(_tool_call_id: str, params: dict) -> AgentToolResult:
    return AgentToolResult([TextBlock(f"Hello, {params['name']}.")])


def _on_agent_end(_event) -> None:
    """Add local notification or metrics code here if needed."""


def _context() -> ContextContribution:
    return ContextContribution("minimal-demo", "A local demo Extension is enabled.")


def setup(api) -> None:
    api.register_tool(
        AgentTool(Tool("extension_hello", "Return a greeting from the demo Extension.", {"name": str}), _say_hello)
    )
    api.on("agent_end", _on_agent_end)
    api.register_context_provider(_context)
