from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Iterable, Mapping

from rova.ai.messages import TextBlock, ToolCall
from rova.ai.tools import Tool
from rova.agent_core.tools import AgentTool, AgentToolResult, ToolExecutionError, ToolExecutionMode, ToolRegistry, ToolRuntime


@dataclass(frozen=True)
class FrozenToolScenario:
    scenario_id: str
    batch_size: int
    tool_classes: tuple[str, ...]
    per_tool_latency_ms: tuple[int, ...]
    requested_runtime_mode: str
    expected_execution_mode: str
    failing_call_index: int | None = None

    def __post_init__(self) -> None:
        if self.batch_size < 1 or len(self.tool_classes) != self.batch_size or len(self.per_tool_latency_ms) != self.batch_size:
            raise ValueError("frozen Tool scenario must provide one class and latency for every call")
        if self.requested_runtime_mode not in {"parallel", "sequential"}:
            raise ValueError("requested_runtime_mode must be parallel or sequential")
        if self.expected_execution_mode not in {"parallel", "sequential"}:
            raise ValueError("expected_execution_mode must be parallel or sequential")
        if any(item not in {"read_only", "mutation"} for item in self.tool_classes):
            raise ValueError("tool classes must be read_only or mutation")
        if any(item <= 0 for item in self.per_tool_latency_ms):
            raise ValueError("Tool scenario latencies must be positive")
        if self.failing_call_index is not None and not 0 <= self.failing_call_index < self.batch_size:
            raise ValueError("failing_call_index must be within the batch")

    def to_dict(self) -> dict[str, object]:
        return {
            "scenario_id": self.scenario_id,
            "batch_size": self.batch_size,
            "tool_classes": list(self.tool_classes),
            "per_tool_latency_ms": list(self.per_tool_latency_ms),
            "requested_runtime_mode": self.requested_runtime_mode,
            "expected_execution_mode": self.expected_execution_mode,
            "failing_call_index": self.failing_call_index,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "FrozenToolScenario":
        return cls(
            scenario_id=str(value["scenario_id"]),
            batch_size=int(value["batch_size"]),
            tool_classes=tuple(str(item) for item in value["tool_classes"]),
            per_tool_latency_ms=tuple(int(item) for item in value["per_tool_latency_ms"]),
            requested_runtime_mode=str(value["requested_runtime_mode"]),
            expected_execution_mode=str(value["expected_execution_mode"]),
            failing_call_index=(None if value.get("failing_call_index") is None else int(value["failing_call_index"])),
        )


def frozen_tool_scenarios() -> tuple[FrozenToolScenario, ...]:
    """The pre-freeze scenario template used only to publish a new manifest."""
    scenarios: list[FrozenToolScenario] = []
    for size in (1, 2, 4, 8):
        for mode in ("sequential", "parallel"):
            scenarios.append(FrozenToolScenario(
                f"TP_uniform_{size}_{mode}", size, ("read_only",) * size, (25,) * size,
                mode, mode,
            ))
    for mode in ("sequential", "parallel"):
        scenarios.append(FrozenToolScenario(
            f"TP_skewed_{mode}", 4, ("read_only",) * 4, (5, 15, 60, 120), mode, mode,
        ))
    scenarios.append(FrozenToolScenario(
        "TP_failure_isolation", 4, ("read_only",) * 4, (25,) * 4, "parallel", "parallel", 1,
    ))
    scenarios.append(FrozenToolScenario(
        "TP_mutation_fallback", 2, ("read_only", "mutation"), (25, 25), "parallel", "sequential",
    ))
    return tuple(scenarios)


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


async def _tool(name: str, delay_ms: int, state: dict[str, int], *, fail: bool, tool_class: str) -> AgentTool:
    async def execute(_tool_call_id: str, _arguments: dict) -> AgentToolResult:
        state["active"] = state.get("active", 0) + 1
        state["max_active"] = max(state.get("max_active", 0), state["active"])
        try:
            await asyncio.sleep(delay_ms / 1_000)
            if fail:
                raise ToolExecutionError(f"{name} failed")
            return AgentToolResult([TextBlock(name)])
        finally:
            state["active"] -= 1

    return AgentTool(
        Tool(name, name, {}),
        execute,
        execution_mode=(ToolExecutionMode.SEQUENTIAL if tool_class == "mutation" else ToolExecutionMode.PARALLEL),
    )


async def _execute(scenario: FrozenToolScenario) -> ToolParallelObservation:
    state: dict[str, int] = {}
    tools = [
        await _tool(
            f"tool_{index}", scenario.per_tool_latency_ms[index], state,
            fail=index == scenario.failing_call_index,
            tool_class=scenario.tool_classes[index],
        )
        for index in range(scenario.batch_size)
    ]
    committed: list[str] = []
    started = time.perf_counter()
    results = await ToolRuntime(ToolRegistry(tools)).execute_batch(
        [ToolCall(f"{scenario.scenario_id}-{index}", tool.tool.name, {}) for index, tool in enumerate(tools)],
        runtime_mode=ToolExecutionMode(scenario.requested_runtime_mode),
        on_result_committed=lambda item: _commit(committed, item.text),
    )
    duration_ms = (time.perf_counter() - started) * 1_000
    expected = [
        f"tool_{index} failed" if index == scenario.failing_call_index else f"tool_{index}"
        for index in range(scenario.batch_size)
    ]
    source_order = [item.text for item in results] == expected and committed == expected
    failure_isolated = scenario.failing_call_index is None or sum(item.is_error for item in results) == 1
    mutation_fallback = scenario.expected_execution_mode != "sequential" or state.get("max_active") == 1
    return ToolParallelObservation(
        scenario.scenario_id,
        scenario.expected_execution_mode,
        duration_ms,
        source_order,
        failure_isolated,
        mutation_fallback,
    )


async def _commit(target: list[str], value: str) -> None:
    target.append(value)


async def run_tool_parallel_scenarios(
    scenarios: Iterable[FrozenToolScenario | Mapping[str, object]],
) -> tuple[ToolParallelObservation, ...]:
    """Execute precisely the scenario objects loaded from a frozen manifest."""
    resolved = [
        item if isinstance(item, FrozenToolScenario) else FrozenToolScenario.from_mapping(item)
        for item in scenarios
    ]
    return tuple([await _execute(item) for item in resolved])


async def run_tool_parallel_dry_run() -> ToolParallelDryRunReport:
    """Execute one deterministic sample for every v2 scenario template."""
    observations = await run_tool_parallel_scenarios(frozen_tool_scenarios())
    return ToolParallelDryRunReport(
        observations,
        provider_request_count=0,
        failure_isolation_violations=sum(not item.failure_isolated for item in observations),
        mutation_fallback_violations=sum(not item.mutation_fallback for item in observations),
    )
