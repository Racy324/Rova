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


@pytest.mark.asyncio
async def test_fault_observations_are_collected_from_actual_runtime_execution() -> None:
    from evals.runtime_v1.fault_benchmark import run_fault_case

    rate_limit = await run_fault_case("FI01", repeat_index=1)
    exhausted = await run_fault_case("FI02", repeat_index=1)
    overflow = await run_fault_case("FI04", repeat_index=1)
    partial_tool = await run_fault_case("FI07", repeat_index=1)
    permanent = await run_fault_case("FI08", repeat_index=1)
    unclassified = await run_fault_case("FI09", repeat_index=1)
    side_effect = await run_fault_case("FI11", repeat_index=1)
    cancelled = await run_fault_case("FI12", repeat_index=1)

    assert rate_limit.observation.provider_attempt_count == 2
    assert exhausted.observation.provider_attempt_count == 2
    assert overflow.observation.compaction_count == 1
    assert partial_tool.observation.partial_tool_execution_count == 0
    assert partial_tool.observation.tool_execution_count == 1
    assert permanent.observation.provider_attempt_count == 1
    assert unclassified.observation.provider_attempt_count == 1
    assert side_effect.observation.side_effect_execution_count == 1
    assert cancelled.observation.provider_attempt_count == 1
    assert all(result.passed for result in (
        rate_limit, exhausted, overflow, partial_tool, permanent, unclassified, side_effect, cancelled,
    ))


def test_fault_evaluation_detects_expected_actual_attempt_mismatch() -> None:
    from evals.runtime_v1.fault_benchmark import FaultRunObservation, evaluate_fault_observation, expected_fault_policy

    observation = FaultRunObservation(
        provider_attempt_count=3,
        recovered=True,
        terminated=False,
        termination_reason=None,
        compaction_count=0,
        tool_execution_count=0,
        side_effect_execution_count=0,
        partial_tool_execution_count=0,
        partial_commit_violations=0,
        transparent_tool_retry_violations=0,
    )

    result = evaluate_fault_observation(expected_fault_policy("FI01"), observation)

    assert result.passed is False
    assert "unexpected_retry_violation" in result.violations
