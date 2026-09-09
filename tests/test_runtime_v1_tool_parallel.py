from __future__ import annotations

import pytest


def test_frozen_tool_scenarios_use_the_planned_latency_distributions() -> None:
    from evals.runtime_v1.tool_parallel import frozen_tool_scenarios

    scenarios = {item.scenario_id: item for item in frozen_tool_scenarios()}

    assert len(scenarios) == 12
    assert scenarios["TP_uniform_8_parallel"].per_tool_latency_ms == (25,) * 8
    assert scenarios["TP_skewed_parallel"].per_tool_latency_ms == (5, 15, 60, 120)
    assert scenarios["TP_failure_isolation"].failing_call_index == 1
    assert scenarios["TP_mutation_fallback"].tool_classes == ("read_only", "mutation")
    assert scenarios["TP_mutation_fallback"].expected_execution_mode == "sequential"


@pytest.mark.asyncio
async def test_tool_parallel_dry_run_executes_all_twelve_planned_scenarios_without_provider() -> None:
    from evals.runtime_v1.tool_parallel import run_tool_parallel_dry_run

    report = await run_tool_parallel_dry_run()

    assert len(report.observations) == 12
    assert report.provider_request_count == 0
    assert all(item.source_order_correct for item in report.observations)
    assert report.failure_isolation_violations == 0
    assert report.mutation_fallback_violations == 0


@pytest.mark.asyncio
async def test_tool_parallel_execution_consumes_the_frozen_scenario_definition() -> None:
    from evals.runtime_v1.tool_parallel import run_tool_parallel_scenarios

    observations = await run_tool_parallel_scenarios([
        {
            "scenario_id": "frozen-scenario",
            "batch_size": 2,
            "tool_classes": ["read_only", "read_only"],
            "per_tool_latency_ms": [1, 3],
            "requested_runtime_mode": "parallel",
            "expected_execution_mode": "parallel",
            "failing_call_index": None,
        }
    ])

    assert observations[0].scenario_id == "frozen-scenario"
    assert observations[0].source_order_correct is True
