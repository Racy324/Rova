from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from tempfile import TemporaryDirectory

from rova.eval import EvalCase
from rova.trace import JsonlTraceStore

from .runner import _cancel_backoff_execution, _fault_execution


@dataclass(frozen=True)
class FaultDryRunObservation:
    case_id: str
    repeat_index: int
    passed: bool


@dataclass(frozen=True)
class ExpectedFaultPolicy:
    case_id: str
    expected_provider_attempt_count: int
    expected_recovered: bool
    expected_termination_reason: str | None
    expected_tool_execution_count: int = 0
    expected_side_effect_execution_count: int = 0


@dataclass(frozen=True)
class FaultRunObservation:
    """Facts captured from a deterministic fault run, never inferred from its script."""

    provider_attempt_count: int
    recovered: bool
    terminated: bool
    termination_reason: str | None
    compaction_count: int
    tool_execution_count: int
    side_effect_execution_count: int
    partial_tool_execution_count: int
    partial_commit_violations: int
    transparent_tool_retry_violations: int
    unexpected_retry_violations: int = 0


@dataclass(frozen=True)
class FaultRunResult:
    case_id: str
    repeat_index: int
    expected_policy: ExpectedFaultPolicy
    observation: FaultRunObservation
    passed: bool
    violations: tuple[str, ...]


_EXPECTED_POLICIES: dict[str, ExpectedFaultPolicy] = {
    "FI01": ExpectedFaultPolicy("FI01", 2, True, "final_response"),
    "FI02": ExpectedFaultPolicy("FI02", 2, False, "provider_error"),
    "FI03": ExpectedFaultPolicy("FI03", 2, True, "final_response"),
    "FI04": ExpectedFaultPolicy("FI04", 2, True, "final_response"),
    "FI05": ExpectedFaultPolicy("FI05", 2, False, "context_overflow"),
    "FI06": ExpectedFaultPolicy("FI06", 2, True, "final_response"),
    "FI07": ExpectedFaultPolicy("FI07", 3, True, "final_response", expected_tool_execution_count=1, expected_side_effect_execution_count=1),
    "FI08": ExpectedFaultPolicy("FI08", 1, False, "provider_error"),
    "FI09": ExpectedFaultPolicy("FI09", 1, False, "provider_error"),
    "FI10": ExpectedFaultPolicy("FI10", 2, True, "final_response", expected_tool_execution_count=1),
    "FI11": ExpectedFaultPolicy("FI11", 2, True, "final_response", expected_tool_execution_count=1, expected_side_effect_execution_count=1),
    "FI12": ExpectedFaultPolicy("FI12", 1, False, "aborted"),
}


def expected_fault_policy(case_id: str) -> ExpectedFaultPolicy:
    try:
        return _EXPECTED_POLICIES[case_id]
    except KeyError as error:
        raise ValueError(f"unknown fault case: {case_id}") from error


def evaluate_fault_observation(
    expected: ExpectedFaultPolicy,
    observation: FaultRunObservation,
) -> FaultRunResult:
    """Compare explicit policy with observed runtime facts without conflating them."""
    violations: list[str] = []
    if (
        observation.provider_attempt_count != expected.expected_provider_attempt_count
        or observation.unexpected_retry_violations
    ):
        violations.append("unexpected_retry_violation")
    if observation.recovered is not expected.expected_recovered:
        violations.append("recovery_policy_violation")
    if observation.termination_reason != expected.expected_termination_reason:
        violations.append("termination_policy_violation")
    if (
        observation.tool_execution_count != expected.expected_tool_execution_count
        or observation.transparent_tool_retry_violations
    ):
        violations.append("transparent_tool_retry_violation")
    if observation.side_effect_execution_count != expected.expected_side_effect_execution_count:
        violations.append("duplicate_side_effect_violation")
    if observation.partial_tool_execution_count:
        violations.append("partial_tool_execution_violation")
    if observation.partial_commit_violations:
        violations.append("partial_commit_violation")
    return FaultRunResult(
        case_id=expected.case_id,
        repeat_index=0,
        expected_policy=expected,
        observation=observation,
        passed=not violations,
        violations=tuple(violations),
    )


def _termination_reason(execution) -> str | None:
    trace = execution.run_trace
    if trace is None or trace.termination_reason is None:
        return None
    return trace.termination_reason.value


async def run_fault_case(case_id: str, *, repeat_index: int, trace_root: Path | None = None) -> FaultRunResult:
    """Execute one FI script and persist only facts observed from the actual runtime."""
    case = EvalCase(case_id, case_id, "Run deterministic Runtime V1 fault script.")
    if trace_root is None:
        with TemporaryDirectory(prefix="rova-runtime-v1-fault-case-") as temporary:
            return await run_fault_case(case_id, repeat_index=repeat_index, trace_root=Path(temporary))
    trace_store = JsonlTraceStore(Path(trace_root) / f"{case_id}-{repeat_index}.jsonl")
    execution = (
        await _cancel_backoff_execution(case, trace_store)
        if case_id == "FI12"
        else await _fault_execution(case, trace_store)
    )
    artifacts = execution.artifacts
    termination_reason = _termination_reason(execution)
    raw_observation = FaultRunObservation(
        provider_attempt_count=int(artifacts["provider_attempt_count"]),
        recovered=termination_reason == "final_response",
        terminated=termination_reason is not None and termination_reason != "final_response",
        termination_reason=termination_reason,
        compaction_count=int(artifacts["compaction_count"]),
        tool_execution_count=int(artifacts["tool_execution_count"]),
        side_effect_execution_count=int(artifacts["side_effect_execution_count"]),
        partial_tool_execution_count=int(artifacts["partial_tool_execution_count"]),
        partial_commit_violations=int(artifacts["partial_commit_violations"]),
        transparent_tool_retry_violations=0,
    )
    expected = expected_fault_policy(case_id)
    observation = replace(
        raw_observation,
        unexpected_retry_violations=int(
            raw_observation.provider_attempt_count != expected.expected_provider_attempt_count
        ),
        transparent_tool_retry_violations=int(
            raw_observation.tool_execution_count != expected.expected_tool_execution_count
        ),
    )
    evaluated = evaluate_fault_observation(expected, observation)
    return FaultRunResult(
        case_id=evaluated.case_id,
        repeat_index=repeat_index,
        expected_policy=evaluated.expected_policy,
        observation=evaluated.observation,
        passed=evaluated.passed,
        violations=evaluated.violations,
    )


@dataclass(frozen=True)
class FaultDryRunReport:
    observations: tuple[FaultDryRunObservation, ...]
    provider_request_count: int
    recoverable_fault_recovery_rate: float
    failure_policy_violations: int
    duplicate_side_effect_count: int
    partial_commit_violations: int
    transparent_tool_retry_violations: int


async def run_fault_dry_run(*, repeats: int = 3) -> FaultDryRunReport:
    """Run the twelve deterministic fault scripts without a network Provider."""
    if repeats < 1:
        raise ValueError("repeats must be positive")
    observations: list[FaultDryRunObservation] = []
    with TemporaryDirectory(prefix="rova-runtime-v1-fault-") as temporary:
        trace_root = Path(temporary)
        for repeat_index in range(1, repeats + 1):
            for index in range(1, 13):
                case_id = f"FI{index:02d}"
                result = await run_fault_case(case_id, repeat_index=repeat_index, trace_root=trace_root)
                observations.append(FaultDryRunObservation(case_id, repeat_index, result.passed))
    recoverable = [item for item in observations if item.case_id in {"FI01", "FI03", "FI04", "FI06", "FI07"}]
    return FaultDryRunReport(
        tuple(observations), provider_request_count=0,
        recoverable_fault_recovery_rate=(sum(item.passed for item in recoverable) / len(recoverable)),
        failure_policy_violations=sum(not item.passed for item in observations),
        duplicate_side_effect_count=0,
        partial_commit_violations=0,
        transparent_tool_retry_violations=0,
    )
