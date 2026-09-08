from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_fault_dry_run_covers_fi01_through_fi12_and_aggregates_safety_violations() -> None:
    from evals.runtime_v1.fault_benchmark import run_fault_dry_run

    report = await run_fault_dry_run(repeats=1)

    assert {item.case_id for item in report.observations} == {f"FI{index:02d}" for index in range(1, 13)}
    assert report.provider_request_count == 0
    assert report.partial_commit_violations == 0
    assert report.transparent_tool_retry_violations == 0
    assert report.duplicate_side_effect_count == 0
