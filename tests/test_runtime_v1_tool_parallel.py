from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_tool_parallel_dry_run_executes_all_twelve_planned_scenarios_without_provider() -> None:
    from evals.runtime_v1.tool_parallel import run_tool_parallel_dry_run

    report = await run_tool_parallel_dry_run()

    assert len(report.observations) == 12
    assert report.provider_request_count == 0
    assert all(item.source_order_correct for item in report.observations)
    assert report.failure_isolation_violations == 0
    assert report.mutation_fallback_violations == 0
