from __future__ import annotations

import ast
import operator

from rova.ai.messages import TextBlock
from rova.ai.tools import Tool
from rova.agent_core.tools import AgentTool, AgentToolResult, ToolExecutionError


_OPERATORS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}


def make_test_calc_tool() -> AgentTool:
    async def execute(_tool_call_id: str, params: dict) -> AgentToolResult:
        expression = params["expression"]
        try:
            node = ast.parse(expression, mode="eval").body
        except SyntaxError as error:
            raise ToolExecutionError("calc accepts one numeric binary expression") from error
        if not isinstance(node, ast.BinOp) or type(node.op) not in _OPERATORS:
            raise ToolExecutionError("calc accepts one numeric binary expression")
        if not isinstance(node.left, ast.Constant) or not isinstance(node.right, ast.Constant):
            raise ToolExecutionError("calc accepts one numeric binary expression")
        if not isinstance(node.left.value, (int, float)) or not isinstance(node.right.value, (int, float)):
            raise ToolExecutionError("calc accepts numeric operands")
        try:
            result = _OPERATORS[type(node.op)](node.left.value, node.right.value)
        except ZeroDivisionError as error:
            raise ToolExecutionError("calc cannot divide by zero") from error
        return AgentToolResult([TextBlock(str(result))])

    return AgentTool(Tool("calc", "Test calculation tool", {"expression": str}), execute)
