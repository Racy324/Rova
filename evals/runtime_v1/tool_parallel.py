from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

from rova.ai.messages import TextBlock, ToolCall
from rova.ai.tools import Tool
from rova.agent_core.tools import AgentTool, AgentToolResult, ToolExecutionError, ToolExecutionMode, ToolRegistry, ToolRuntime


@dataclass(frozen=True)
class ToolParallelObservation:
    scenario_id: str
    mode: str
    duration_ms: float
    source_order_correct: bool
    failure_isolated: bool
    mutation_fallback: bool


@dataclass(frozen=True)
class ToolParallelDryRunReport:
    observations: tuple[ToolParallelObservation, ...]
    provider_request_count: int
    failure_isolation_violations: int
    mutation_fallback_violations: int


async def _tool(name: str, delay: float, state: dict[str, int], *, fail: bool = False, mode: ToolExecutionMode = ToolExecutionMode.PARALLEL) -> AgentTool:
    async def execute(_tool_call_id: str, _arguments: dict) -> AgentToolResult:
        state["active"] = state.get("active", 0) + 1
        state["max_active"] = max(state.get("max_active", 0), state["active"])
        try:
            await asyncio.sleep(delay)
            if fail:
                raise ToolExecutionError(f"{name} failed")
            return AgentToolResult([TextBlock(name)])
        finally:
            state["active"] -= 1
    return AgentTool(Tool(name, name, {}), execute, execution_mode=mode)


async def _execute(scenario_id: str, *, runtime_mode: ToolExecutionMode, delays: list[float], fail_index: int | None = None, mutating: bool = False) -> ToolParallelObservation:
    state: dict[str, int] = {}
    tools = [
        await _tool(
            f"tool_{index}", delay, state, fail=index == fail_index,
            mode=ToolExecutionMode.SEQUENTIAL if mutating and index == len(delays) - 1 else ToolExecutionMode.PARALLEL,
        )
        for index, delay in enumerate(delays)
    ]
    committed: list[str] = []
    started = time.perf_counter()
    results = await ToolRuntime(ToolRegistry(tools)).execute_batch(
        [ToolCall(f"{scenario_id}-{index}", tool.tool.name, {}) for index, tool in enumerate(tools)],
        runtime_mode=runtime_mode,
        on_result_committed=lambda item: _commit(committed, item.text),
    )
    duration_ms = (time.perf_counter() - started) * 1000
    expected = [f"tool_{index} failed" if index == fail_index else f"tool_{index}" for index in range(len(tools))]
    source_order = [item.text for item in results] == expected and committed == expected
    failure_isolated = fail_index is None or sum(item.is_error for item in results) == 1
    mutation_fallback = not mutating or state.get("max_active") == 1
    return ToolParallelObservation(scenario_id, runtime_mode.value, duration_ms, source_order, failure_isolated, mutation_fallback)


async def _commit(target: list[str], value: str) -> None:
    target.append(value)


async def run_tool_parallel_dry_run() -> ToolParallelDryRunReport:
    """Execute one non-formal deterministic sample for each planned scenario."""
    observations: list[ToolParallelObservation] = []
    for size in (1, 2, 4, 8):
        delays = [0.003] * size
        for mode in (ToolExecutionMode.SEQUENTIAL, ToolExecutionMode.PARALLEL):
            observations.append(await _execute(f"TP_uniform_{size}_{mode.value}", runtime_mode=mode, delays=delays))
    for mode in (ToolExecutionMode.SEQUENTIAL, ToolExecutionMode.PARALLEL):
        observations.append(await _execute(f"TP_skewed_{mode.value}", runtime_mode=mode, delays=[0.001, 0.002, 0.006, 0.012]))
    observations.append(await _execute("TP_failure_isolation", runtime_mode=ToolExecutionMode.PARALLEL, delays=[0.003] * 4, fail_index=1))
    observations.append(await _execute("TP_mutation_fallback", runtime_mode=ToolExecutionMode.PARALLEL, delays=[0.003, 0.003], mutating=True))
    return ToolParallelDryRunReport(
        tuple(observations),
        provider_request_count=0,
        failure_isolation_violations=sum(not item.failure_isolated for item in observations),
        mutation_fallback_violations=sum(not item.mutation_fallback for item in observations),
    )
