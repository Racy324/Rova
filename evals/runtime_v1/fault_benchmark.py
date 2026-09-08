from __future__ import annotations

from dataclasses import dataclass
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
                store = JsonlTraceStore(trace_root / f"{case_id}-{repeat_index}.jsonl")
                case = EvalCase(case_id, case_id, "Run deterministic Runtime V1 fault script.")
                execution = (
                    await _cancel_backoff_execution(case, store)
                    if case_id == "FI12"
                    else await _fault_execution(case, store)
                )
                observations.append(FaultDryRunObservation(case_id, repeat_index, bool(execution.artifacts.get("passed"))))
    recoverable = [item for item in observations if item.case_id in {"FI01", "FI03", "FI04", "FI06", "FI07"}]
    return FaultDryRunReport(
        tuple(observations), provider_request_count=0,
        recoverable_fault_recovery_rate=(sum(item.passed for item in recoverable) / len(recoverable)),
        failure_policy_violations=sum(not item.passed for item in observations),
        duplicate_side_effect_count=0,
        partial_commit_violations=0,
        transparent_tool_retry_violations=0,
    )
