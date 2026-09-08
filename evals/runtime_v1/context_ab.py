from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable

from rova.ai.context import Context
from rova.ai.events import StreamDone
from rova.ai.messages import AssistantMessage, TextBlock, ToolCall, ToolResultMessage
from rova.ai.models import Model
from rova.ai.stream import stream_simple
from rova.agent_session.compaction import CompactionPolicy
from rova.eval import CheckResult, EvalCase, EvalExecution, EvalResult, EvalRunner, JsonlEvalStore
from rova.trace import JsonlTraceStore

from .context_cases import prompt_script, snapshot_workspace, validate_workspace
from .fixtures import SmokeFixture, dry_run_context_fixtures, fresh_workspace
from .runtime_factory import ContextManagementProfile, build_context_runtime
from .spec import RunManifestRecord


_FULL_POLICY = CompactionPolicy(reserve_tokens=12_000, keep_recent_tokens=20_000)
_DRY_MODEL = Model("runtime-v1-development", context_window=64_000)


@dataclass(frozen=True)
class ContextDryRunObservation:
    case_id: str
    profile: str
    provider_requests: int
    compaction_count: int
    externalization_count: int
    overflow_recovery_count: int
    termination_reason: str | None
    input_tokens: int | None
    output_tokens: int | None
    duration_ms: float | None


@dataclass(frozen=True)
class ContextDryRunReport:
    results: tuple[EvalResult, ...]
    observations: tuple[ContextDryRunObservation, ...]
    provider_request_count: int


class _ContextValidator:
    async def evaluate(self, case: EvalCase, execution: EvalExecution) -> CheckResult:
        return CheckResult(
            "deterministic_workspace_validator",
            bool(execution.artifacts.get("validator_passed")),
            "fixed fixture output",
        )


class _DevelopmentContextProvider:
    """Deterministic development provider; it is not used for the real capability call."""

    def __init__(self, case_id: str) -> None:
        self.case_id = case_id
        self.calls = 0

    async def __call__(self, _model: Model, context: Context, _options: object | None = None):
        self.calls += 1
        if "You summarize historical conversation data" in context.system_prompt:
            yield StreamDone(AssistantMessage([TextBlock(
                "## Goal\nPreserve the fixture task.\n## Constraints\nKeep violet constraints.\n"
                "## Progress / Results\nHistory compacted.\n## Key Decisions\nUse fixed mappings.\n"
                "## Next Steps\nComplete the final repair.\n## Critical Context\nThe final value is deterministic."
            )]))
            return
        tool_results = [item for item in context.messages if isinstance(item, ToolResultMessage)]
        if self.case_id == "CM01_large_tool_output_repair":
            if not tool_results:
                yield StreamDone(AssistantMessage([ToolCall("read-large", "read", {"path": "reference.txt"})], stop_reason="tool_calls"))
            elif len(tool_results) == 1:
                yield StreamDone(AssistantMessage([ToolCall("write-repair", "write", {
                    "path": "src/rule_engine.py",
                    "content": 'def selected_mapping() -> str:\n    return "violet-47"\n',
                })], stop_reason="tool_calls"))
            else:
                yield StreamDone(AssistantMessage([TextBlock("CM01 complete")]))
            return
        final_turn = any("FINAL_IMPLEMENT" in getattr(item, "content", "") for item in context.messages)
        if final_turn and not tool_results:
            yield StreamDone(AssistantMessage([ToolCall("write-followthrough", "write", {
                "path": "src/followthrough.py",
                "content": 'def required_constraint() -> str:\n    return "keep-violet-3"\n',
            })], stop_reason="tool_calls"))
        elif final_turn:
            yield StreamDone(AssistantMessage([TextBlock("CM02 complete")]))
        else:
            yield StreamDone(AssistantMessage([TextBlock("constraint acknowledged")]))


async def _run_case(
    fixture: SmokeFixture,
    profile: ContextManagementProfile,
    *,
    root: Path,
    model: Model,
    stream_fn: Callable,
    use_development_provider: bool,
    isolated_sandbox: bool,
    sandbox_image: str | None,
    keep_failed_workspace: bool,
) -> tuple[EvalExecution, int, ContextDryRunObservation]:
    provider = _DevelopmentContextProvider(fixture.case_id) if use_development_provider else None
    provider_requests = 0

    async def counted_stream(model: Model, context: Context, options: object | None = None):
        nonlocal provider_requests
        provider_requests += 1
        async for event in stream_fn(model, context, options):
            yield event

    effective_stream = provider if provider is not None else counted_stream
    with fresh_workspace(fixture, root / "workspaces", keep_failed=keep_failed_workspace) as workspace:
        baseline = snapshot_workspace(workspace)
        state_root = root / "state"
        runtime = build_context_runtime(
            profile=profile,
            model=model,
            stream_fn=effective_stream,
            workspace_root=workspace,
            state_root=state_root,
            trace_root=state_root / "traces",
            isolated_sandbox=isolated_sandbox,
            sandbox_root=state_root / "sandboxes",
            sandbox_image=sandbox_image,
        )
        try:
            responses = []
            for prompt in prompt_script(fixture.case_id, workspace):
                responses = await runtime.prompt(prompt)
                if responses[-1].stop_reason in {"error", "aborted"}:
                    break
            traces = JsonlTraceStore(state_root / "traces" / "runs.jsonl").load_all()
            trace = traces[-1]
            tool_calls = [call for item in traces for step in item.steps for call in step.tool_calls]
            externalization_count = sum(
                call.result is not None and call.result.externalized for call in tool_calls
            )
            overflow_recovery_count = sum(
                item.trigger.value == "overflow_recovery"
                for trace_item in traces for item in trace_item.compactions
            )
            observation = ContextDryRunObservation(
                fixture.case_id,
                profile.name,
                provider.calls if provider is not None else provider_requests,
                sum(len(item.compactions) for item in traces),
                externalization_count,
                overflow_recovery_count,
                trace.termination_reason.value if trace.termination_reason else None,
                trace.actual_usage.input_tokens if trace.actual_usage else None,
                trace.actual_usage.output_tokens if trace.actual_usage else None,
                sum(item.duration_ms or 0.0 for item in traces),
            )
            validator_root = execution_workspace_root(runtime, workspace)
            validation = validate_workspace(fixture.case_id, validator_root, baseline)
            return EvalExecution(
                fixture.case_id,
                trace,
                responses[-1],
                artifacts={
                    "validator_passed": validation.passed,
                    "validator_reason": validation.reason,
                },
            ), observation.provider_requests, observation
        finally:
            await runtime.close()
            if runtime.sandbox_control is not None:
                runtime.sandbox_control.discard(confirm=lambda _plan: True)


async def run_context_dry_run(
    result_root: Path,
    *,
    use_development_provider: bool = False,
    model: Model | None = None,
    stream_fn: Callable = stream_simple,
    isolated_sandbox: bool = False,
    sandbox_image: str | None = None,
    keep_failed_workspace: bool = False,
) -> ContextDryRunReport:
    """Run one non-formal repeat per Context case/profile through the product factory."""
    if isolated_sandbox and not sandbox_image:
        raise ValueError("sandbox_image is required when isolated_sandbox is enabled")
    root = Path(result_root)
    root.mkdir(parents=True, exist_ok=True)
    effective_model = model or _DRY_MODEL
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "suite": "runtime-v1-evaluation",
                "phase": "dry-run",
                "provider_kind": "development" if use_development_provider else "configured-live",
                "context_window": effective_model.context_window if model is not None else _DRY_MODEL.context_window,
                "fixtures": [{"case_id": item.case_id, "sha256": item.sha256} for item in dry_run_context_fixtures()],
                "profiles": {
                    "base": {"compaction": False, "externalization": False, "overflow_recovery": False},
                    "full": {"compaction": True, "externalization": True, "overflow_recovery": True},
                },
            },
            ensure_ascii=False,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )
    store = JsonlEvalStore(root / "eval-results.jsonl")
    manifest = root / "run-manifest.jsonl"
    traces = JsonlTraceStore(root / "traces" / "runs.jsonl")
    results: list[EvalResult] = []
    observations: list[ContextDryRunObservation] = []
    for fixture in dry_run_context_fixtures():
        for profile in (ContextManagementProfile.base(), ContextManagementProfile.full(_FULL_POLICY)):
            case = EvalCase(fixture.case_id, fixture.case_id, "Run the immutable Runtime V1 Context dry-run fixture.")

            class _Executor:
                async def execute(self, _case: EvalCase) -> EvalExecution:
                    execution, provider_requests, observation = await _run_case(
                        fixture, profile, root=root / fixture.case_id / profile.name,
                        model=effective_model, stream_fn=stream_fn,
                        use_development_provider=use_development_provider,
                        isolated_sandbox=isolated_sandbox, sandbox_image=sandbox_image,
                        keep_failed_workspace=keep_failed_workspace,
                    )
                    self.provider_requests = provider_requests
                    self.observation = observation
                    return execution

            executor = _Executor()
            suite = await EvalRunner(executor, lambda _case: [_ContextValidator()], suite_id="runtime-v1-dry-run").run([case])
            result = suite.case_results[0]
            store.append(result)
            traces.append(_load_trace(root / fixture.case_id / profile.name))
            record = RunManifestRecord(
                experiment="context_ab", case_id=fixture.case_id, profile=profile.name, repeat_index=1,
                fixture_id=fixture.case_id, fixture_sha256=fixture.sha256, eval_run_id=result.eval_run_id,
                run_id=result.run_id, provider_requests=executor.provider_requests,
            ).to_dict()
            with manifest.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            results.append(result)
            observations.append(executor.observation)
    report = ContextDryRunReport(tuple(results), tuple(observations), sum(item.provider_requests for item in observations))
    _write_context_report(root, report)
    return report


def _load_trace(root: Path):
    return JsonlTraceStore(root / "state" / "traces" / "runs.jsonl").load_all()[-1]


def execution_workspace_root(runtime, host_workspace: Path) -> Path:
    """Return the filesystem actually mutated by the evaluated Runtime."""
    environment = getattr(runtime, "execution_environment", None)
    filesystem = getattr(environment, "filesystem", None)
    if filesystem is None:
        return Path(host_workspace)
    return Path(filesystem.resolve("."))


def _write_context_report(root: Path, report: ContextDryRunReport) -> None:
    payload = {
        "provider_request_count": report.provider_request_count,
        "results": [
            {
                "case_id": item.case_id,
                "status": item.status.value,
                "task_success": item.task_success,
                "termination_reason": item.runtime_termination_reason.value if item.runtime_termination_reason else None,
                "input_tokens": item.metrics.input_tokens,
                "output_tokens": item.metrics.output_tokens,
                "duration_ms": item.duration_ms,
            }
            for item in report.results
        ],
        "observations": [item.__dict__ for item in report.observations],
    }
    (root / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# Runtime V1 Context Dry Run", "", f"- Provider requests: {report.provider_request_count}", ""]
    for item in report.observations:
        lines.append(
            f"- {item.case_id} / {item.profile}: compactions={item.compaction_count}, "
            f"externalizations={item.externalization_count}, termination={item.termination_reason or 'unavailable'}"
        )
    (root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def restore_context_dry_run_report(result_root: Path) -> ContextDryRunReport:
    """Rebuild a report from already persisted Dry Run JSONL authority."""
    root = Path(result_root)
    results = JsonlEvalStore(root / "eval-results.jsonl").load_all()
    records = [json.loads(line) for line in (root / "run-manifest.jsonl").read_text(encoding="utf-8").splitlines() if line]
    observations: list[ContextDryRunObservation] = []
    for record in records:
        state_traces = JsonlTraceStore(
            root / record["case_id"] / record["profile"] / "state" / "traces" / "runs.jsonl"
        ).load_all()
        final = state_traces[-1]
        calls = [call for trace in state_traces for step in trace.steps for call in step.tool_calls]
        # One committed StepTrace corresponds to one completed provider request;
        # every completed compaction uses one isolated summary request.  This
        # repairs early Dry Run manifests written before stream-boundary
        # counting was added, without inferring discarded-attempt billing.
        provider_requests = sum(len(trace.steps) + len(trace.compactions) for trace in state_traces)
        record["provider_requests"] = provider_requests
        observations.append(ContextDryRunObservation(
            record["case_id"], record["profile"], provider_requests,
            sum(len(trace.compactions) for trace in state_traces),
            sum(call.result is not None and call.result.externalized for call in calls),
            sum(compaction.trigger.value == "overflow_recovery" for trace in state_traces for compaction in trace.compactions),
            final.termination_reason.value if final.termination_reason else None,
            final.actual_usage.input_tokens if final.actual_usage else None,
            final.actual_usage.output_tokens if final.actual_usage else None,
            sum(trace.duration_ms or 0.0 for trace in state_traces),
        ))
    (root / "run-manifest.jsonl").write_text(
        "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
        newline="\n",
    )
    report = ContextDryRunReport(tuple(results), tuple(observations), sum(item.provider_requests for item in observations))
    _write_context_report(root, report)
    return report
