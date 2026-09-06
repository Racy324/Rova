from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import math
from typing import Any

from rova.ai.messages import AssistantMessage, Usage
from rova.trace.models import RunTrace, TerminationReason, TraceError, ToolOutcome


class EvalStatus(Enum):
    PASS = "pass"
    FAIL = "fail"
    ERROR = "error"


class FailureReason(Enum):
    TASK_VALIDATION_FAILED = "task_validation_failed"
    RUNTIME_PROVIDER_ERROR = "runtime_provider_error"
    RUNTIME_HARNESS_ERROR = "runtime_harness_error"
    RUNTIME_ABORTED = "runtime_aborted"
    MAX_TURNS = "max_turns"
    SESSION_PERSISTENCE_ERROR = "session_persistence_error"
    EVALUATOR_ERROR = "evaluator_error"


@dataclass(frozen=True)
class EvalCase:
    """声明一个与具体 Agent Runtime 解耦的评估任务。"""

    case_id: str
    name: str
    prompt: str
    description: str = ""
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    message: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", _json_metadata_copy(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "message": self.message,
            "metadata": _json_metadata_copy(self.metadata),
        }


@dataclass
class EvalExecution:
    """Executor 实际运行一个 Case 后产生的运行事实。"""

    case_id: str
    run_trace: RunTrace | None
    final_assistant: AssistantMessage | None = None
    execution_error: TraceError | None = None
    artifacts: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BehaviorMetrics:
    """从 RunTrace 派生的运行行为统计，不表达任务是否成功。"""

    turn_count: int
    tool_call_count: int
    tool_error_count: int
    policy_denied_count: int
    approval_denied_count: int
    shell_nonzero_count: int
    shell_timeout_count: int
    compaction_count: int
    actual_usage_available: bool
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    run_duration_ms: float | None

    def __post_init__(self) -> None:
        for field_name in (
            "turn_count",
            "tool_call_count",
            "tool_error_count",
            "policy_denied_count",
            "approval_denied_count",
            "shell_nonzero_count",
            "shell_timeout_count",
            "compaction_count",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{field_name} must be a non-negative integer")
        for field_name in ("input_tokens", "output_tokens", "total_tokens"):
            value = getattr(self, field_name)
            if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value < 0):
                raise ValueError(f"{field_name} must be a non-negative integer or None")
        if self.actual_usage_available != (self.input_tokens is not None):
            raise ValueError("actual usage availability must match token fields")
        if self.run_duration_ms is not None:
            _ensure_finite_number(self.run_duration_ms, "run_duration_ms")

    @classmethod
    def from_run_trace(cls, trace: RunTrace) -> BehaviorMetrics:
        if trace.steps:
            usage = trace.actual_usage if trace.actual_usage_complete else None
            tools = [tool for step in trace.steps for tool in step.tool_calls]
            turn_count = len(trace.steps)
            tool_error_count = sum(tool.outcome is not None and tool.outcome is not ToolOutcome.SUCCESS for tool in tools)
        else:
            usage = trace.usage or _aggregate_turn_usage(trace)
            tools = trace.tool_executions
            turn_count = len(trace.turns)
            tool_error_count = sum(tool.is_error is True for tool in tools)
        return cls(
            turn_count=turn_count,
            tool_call_count=len(tools),
            tool_error_count=tool_error_count,
            policy_denied_count=sum(
                tool.outcome is ToolOutcome.POLICY_DENIED for tool in tools
            ),
            approval_denied_count=sum(
                tool.outcome is ToolOutcome.APPROVAL_DENIED for tool in tools
            ),
            shell_nonzero_count=sum(
                tool.outcome is ToolOutcome.COMMAND_NONZERO_EXIT for tool in tools
            ),
            shell_timeout_count=sum(
                tool.outcome is ToolOutcome.COMMAND_TIMEOUT for tool in tools
            ),
            compaction_count=len(trace.compactions),
            actual_usage_available=usage is not None,
            input_tokens=usage.input_tokens if usage else None,
            output_tokens=usage.output_tokens if usage else None,
            total_tokens=usage.total_tokens if usage else None,
            run_duration_ms=trace.duration_ms,
        )


@dataclass(frozen=True)
class EvalResult:
    """一个 Case 的最终评估结果；只关联 run_id，不复制完整 Trace。"""

    eval_run_id: str
    case_id: str
    status: EvalStatus
    started_at: datetime
    ended_at: datetime
    duration_ms: float
    run_id: str | None
    checks: list[CheckResult]
    task_success: bool | None
    runtime_termination_reason: TerminationReason | None
    failure_reason: FailureReason | None
    metrics: BehaviorMetrics
    error: TraceError | None = None

    def __post_init__(self) -> None:
        _ensure_finite_number(self.duration_ms, "duration_ms")
        if self.status is EvalStatus.PASS:
            if self.task_success is not True:
                raise ValueError("PASS requires task_success=True")
            if self.failure_reason is not None or self.error is not None:
                raise ValueError("PASS cannot include a failure reason or error")
        elif self.status is EvalStatus.FAIL:
            if self.task_success is not False:
                raise ValueError("FAIL requires task_success=False")
            if self.failure_reason is FailureReason.EVALUATOR_ERROR:
                raise ValueError("FAIL cannot use EVALUATOR_ERROR")
        elif self.status is EvalStatus.ERROR:
            if self.task_success is not None:
                raise ValueError("ERROR requires task_success=None")
            if self.failure_reason is not FailureReason.EVALUATOR_ERROR:
                raise ValueError("ERROR requires failure_reason=EVALUATOR_ERROR")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "eval_run_id": self.eval_run_id,
            "case_id": self.case_id,
            "status": self.status.value,
            "started_at": self.started_at.isoformat(),
            "ended_at": self.ended_at.isoformat(),
            "duration_ms": self.duration_ms,
            "run_id": self.run_id,
            "checks": [_check_to_dict(check) for check in self.checks],
            "task_success": self.task_success,
            "runtime_termination_reason": (
                self.runtime_termination_reason.value
                if self.runtime_termination_reason is not None
                else None
            ),
            "failure_reason": (
                self.failure_reason.value if self.failure_reason is not None else None
            ),
            "metrics": _metrics_to_dict(self.metrics),
            "error": _trace_error_to_dict(self.error),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> EvalResult:
        if not isinstance(value, dict):
            raise ValueError("EvalResult must be an object")
        expected_fields = {
            "schema_version",
            "eval_run_id",
            "case_id",
            "status",
            "started_at",
            "ended_at",
            "duration_ms",
            "run_id",
            "checks",
            "task_success",
            "runtime_termination_reason",
            "failure_reason",
            "metrics",
            "error",
        }
        unknown_fields = set(value) - expected_fields
        if unknown_fields:
            raise ValueError(f"EvalResult contains unknown fields: {sorted(unknown_fields)}")
        missing_fields = expected_fields - set(value)
        if missing_fields:
            raise ValueError(f"EvalResult is missing fields: {sorted(missing_fields)}")
        if value.get("schema_version") != 1:
            raise ValueError("unsupported EvalResult schema_version")
        return cls(
            eval_run_id=_required_str(value, "eval_run_id"),
            case_id=_required_str(value, "case_id"),
            status=EvalStatus(_required_str(value, "status")),
            started_at=_required_datetime(value, "started_at"),
            ended_at=_required_datetime(value, "ended_at"),
            duration_ms=_required_number(value, "duration_ms"),
            run_id=_optional_str(value, "run_id"),
            checks=[_check_from_dict(item) for item in _required_list(value, "checks")],
            task_success=_optional_bool(value, "task_success"),
            runtime_termination_reason=_optional_enum(
                value, "runtime_termination_reason", TerminationReason
            ),
            failure_reason=_optional_enum(value, "failure_reason", FailureReason),
            metrics=_metrics_from_dict(_required_dict(value, "metrics")),
            error=_trace_error_from_dict(value.get("error")),
        )


@dataclass(frozen=True)
class EvalSuiteResult:
    suite_id: str
    case_results: list[EvalResult]

    @property
    def total(self) -> int:
        return len(self.case_results)

    @property
    def passed(self) -> int:
        return sum(result.status is EvalStatus.PASS for result in self.case_results)

    @property
    def failed(self) -> int:
        return sum(result.status is EvalStatus.FAIL for result in self.case_results)

    @property
    def errors(self) -> int:
        return sum(result.status is EvalStatus.ERROR for result in self.case_results)

    @property
    def task_success_rate(self) -> float:
        decided_results = [
            result for result in self.case_results if result.task_success is not None
        ]
        if not decided_results:
            return 0.0
        return sum(result.task_success is True for result in decided_results) / len(
            decided_results
        )


def _aggregate_turn_usage(trace: RunTrace) -> Usage | None:
    usages = [turn.usage for turn in trace.turns if turn.usage is not None]
    if not usages:
        return None
    return Usage(
        input_tokens=sum(usage.input_tokens for usage in usages),
        output_tokens=sum(usage.output_tokens for usage in usages),
        total_tokens=sum(usage.total_tokens for usage in usages),
    )


def _check_to_dict(check: CheckResult) -> dict[str, Any]:
    return check.to_dict()


def _check_from_dict(value: Any) -> CheckResult:
    if not isinstance(value, dict):
        raise ValueError("check must be an object")
    passed = value.get("passed")
    if not isinstance(passed, bool):
        raise ValueError("check.passed must be a bool")
    return CheckResult(
        name=_required_str(value, "name"),
        passed=passed,
        message=_required_str(value, "message"),
        metadata=_required_dict(value, "metadata"),
    )


def _metrics_to_dict(metrics: BehaviorMetrics) -> dict[str, Any]:
    return {
        "turn_count": metrics.turn_count,
        "tool_call_count": metrics.tool_call_count,
        "tool_error_count": metrics.tool_error_count,
        "policy_denied_count": metrics.policy_denied_count,
        "approval_denied_count": metrics.approval_denied_count,
        "shell_nonzero_count": metrics.shell_nonzero_count,
        "shell_timeout_count": metrics.shell_timeout_count,
        "compaction_count": metrics.compaction_count,
        "actual_usage_available": metrics.actual_usage_available,
        "input_tokens": metrics.input_tokens,
        "output_tokens": metrics.output_tokens,
        "total_tokens": metrics.total_tokens,
        "run_duration_ms": metrics.run_duration_ms,
    }


def _metrics_from_dict(value: dict[str, Any]) -> BehaviorMetrics:
    int_fields = (
        "turn_count",
        "tool_call_count",
        "tool_error_count",
        "policy_denied_count",
        "approval_denied_count",
        "shell_nonzero_count",
        "shell_timeout_count",
        "compaction_count",
    )
    values = {field_name: _required_int(value, field_name) for field_name in int_fields}
    actual_usage_available = _bool(value.get("actual_usage_available"), "actual_usage_available")
    for field_name in ("input_tokens", "output_tokens", "total_tokens"):
        token_value = value.get(field_name)
        values[field_name] = None if token_value is None else _required_int(value, field_name)
    duration = value.get("run_duration_ms")
    if duration is not None:
        _ensure_finite_number(duration, "metrics.run_duration_ms")
    return BehaviorMetrics(
        **values,
        actual_usage_available=actual_usage_available,
        run_duration_ms=float(duration) if duration is not None else None,
    )


def _trace_error_to_dict(error: TraceError | None) -> dict[str, str] | None:
    if error is None:
        return None
    return {"error_type": error.error_type, "message": error.message}


def _trace_error_from_dict(value: Any) -> TraceError | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("error must be an object or null")
    return TraceError(
        error_type=_required_str(value, "error_type"),
        message=_required_str(value, "message"),
    )


def _required_str(value: dict[str, Any], name: str) -> str:
    result = value.get(name)
    if not isinstance(result, str):
        raise ValueError(f"{name} must be a string")
    return result


def _optional_str(value: dict[str, Any], name: str) -> str | None:
    result = value.get(name)
    if result is not None and not isinstance(result, str):
        raise ValueError(f"{name} must be a string or null")
    return result


def _required_number(value: dict[str, Any], name: str) -> float:
    result = value.get(name)
    _ensure_finite_number(result, name)
    return float(result)


def _required_int(value: dict[str, Any], name: str) -> int:
    result = value.get(name)
    if not isinstance(result, int) or isinstance(result, bool):
        raise ValueError(f"{name} must be an integer")
    return result


def _bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a bool")
    return value


def _required_list(value: dict[str, Any], name: str) -> list[Any]:
    result = value.get(name)
    if not isinstance(result, list):
        raise ValueError(f"{name} must be a list")
    return result


def _required_dict(value: dict[str, Any], name: str) -> dict[str, Any]:
    result = value.get(name)
    if not isinstance(result, dict):
        raise ValueError(f"{name} must be an object")
    return result


def _required_datetime(value: dict[str, Any], name: str) -> datetime:
    raw_value = _required_str(value, name)
    try:
        result = datetime.fromisoformat(raw_value)
    except ValueError as error:
        raise ValueError(f"{name} must be an ISO-8601 datetime") from error
    if result.tzinfo is None:
        raise ValueError(f"{name} must include a timezone")
    return result


def _optional_bool(value: dict[str, Any], name: str) -> bool | None:
    result = value.get(name)
    if result is not None and not isinstance(result, bool):
        raise ValueError(f"{name} must be a bool or null")
    return result


def _optional_enum(
    value: dict[str, Any], name: str, enum_type: type[Enum]
) -> Any | None:
    raw_value = value.get(name)
    if raw_value is None:
        return None
    if not isinstance(raw_value, str):
        raise ValueError(f"{name} must be a string or null")
    try:
        return enum_type(raw_value)
    except ValueError as error:
        raise ValueError(f"{name} contains an unknown value") from error


def _ensure_finite_number(value: Any, name: str) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{name} must be a number")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")


def _json_metadata_copy(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("metadata must be a JSON object")
    copied = _json_value_copy(value)
    assert isinstance(copied, dict)
    return copied


def _json_value_copy(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("metadata numbers must be finite")
        return value
    if isinstance(value, list):
        return [_json_value_copy(item) for item in value]
    if isinstance(value, dict):
        copied: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("metadata JSON object keys must be strings")
            copied[key] = _json_value_copy(item)
        return copied
    raise ValueError("metadata must contain only JSON-compatible values")
