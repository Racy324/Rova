from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable

from rova.ai.context import Context
from rova.ai.events import ProviderFailure, Start, StreamDone, StreamError, TextDelta, ToolCallDelta
from rova.ai.messages import AssistantMessage, TextBlock, ToolCall, ToolResultMessage, UserMessage
from rova.ai.models import Model
from rova.ai.tools import Tool
from rova.agent_core.agent import Agent
from rova.agent_core.retry import ProviderRetryPolicy
from rova.agent_core.tools import AgentTool, AgentToolResult, ToolExecutionError, ToolExecutionMode, ToolRegistry, ToolRuntime
from rova.agent_session.compaction import CompactionPolicy
from rova.eval import CheckResult, EvalCase, EvalExecution, EvalResult, EvalRunner, JsonlEvalStore
from rova.trace import JsonlTraceStore, TraceRecorder
from rova.trace.models import RunStatus, RunTrace, TerminationReason

from .fixtures import SmokeFixture, fresh_workspace, smoke_fixtures
from .runtime_factory import ContextManagementProfile, build_context_runtime
from .spec import RUN_MANIFEST_SCHEMA_VERSION, RunManifestRecord


SMOKE_MANIFEST_VERSION = RUN_MANIFEST_SCHEMA_VERSION


@dataclass(frozen=True)
class InfrastructureSmokeReport:
    results: tuple[EvalResult, ...]
    context_provider_request_count: int
    context_result_count: int
    tool_result_count: int
    fault_result_count: int
    workspace_cleanup_verified: bool


class _ResultStore:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.eval_store = JsonlEvalStore(self.root / "eval-results.jsonl")
        self.trace_store = JsonlTraceStore(self.root / "traces" / "runs.jsonl")
        self.manifest_path = self.root / "manifest.json"
        self.run_manifest_path = self.root / "run-manifest.jsonl"

    def initialize(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest_path.write_text(
            json.dumps(
                {
                    "schema_version": SMOKE_MANIFEST_VERSION,
                    "suite": "runtime-v1-evaluation",
                    "phase": "infrastructure-smoke",
                    "fixtures": [
                        {"case_id": item.case_id, "sha256": item.sha256}
                        for item in smoke_fixtures()
                    ],
                },
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    def append(self, result: EvalResult, *, experiment: str, profile: str, fixture: SmokeFixture | None, provider_requests: int) -> None:
        self.eval_store.append(result)
        record = RunManifestRecord(
            experiment=experiment,
            case_id=result.case_id,
            profile=profile,
            repeat_index=1,
            fixture_id=fixture.case_id if fixture is not None else None,
            fixture_sha256=fixture.sha256 if fixture is not None else None,
            eval_run_id=result.eval_run_id,
            run_id=result.run_id,
            provider_requests=provider_requests,
        ).to_dict()
        with self.run_manifest_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


class _SmokeEvaluator:
    async def evaluate(self, _case: EvalCase, execution: EvalExecution) -> CheckResult:
        return CheckResult("smoke_contract", bool(execution.artifacts.get("passed")), "deterministic smoke contract")


async def _run_eval(case: EvalCase, execute: Callable[[EvalCase], Awaitable[EvalExecution]]) -> EvalResult:
    class Executor:
        async def execute(self, item: EvalCase) -> EvalExecution:
            return await execute(item)

    suite = await EvalRunner(Executor(), lambda _case: [_SmokeEvaluator()], suite_id="runtime-v1-smoke").run([case])
    return suite.case_results[0]


class _ContextSmokeProvider:
    def __init__(self, case_id: str) -> None:
        self.case_id = case_id
        self.calls = 0

    async def __call__(self, _model: Model, context: Context, _options: object | None = None):
        self.calls += 1
        if any(isinstance(message, ToolResultMessage) for message in context.messages):
            yield StreamDone(AssistantMessage([TextBlock(f"smoke complete {self.case_id}")]))
            return
        yield StreamDone(AssistantMessage(
            [ToolCall("read-reference", "eval_read_reference", {})],
            stop_reason="tool_calls",
        ))


async def _context_execution(
    case: EvalCase,
    *,
    profile: ContextManagementProfile,
    fixture: SmokeFixture,
    result_store: _ResultStore,
    workspace_root: Path,
    keep_failed_workspace: bool,
) -> tuple[EvalExecution, int]:
    provider = _ContextSmokeProvider(case.case_id)
    state_root = workspace_root / "state"
    with fresh_workspace(
        fixture,
        workspace_root / "workspaces",
        keep_failed=keep_failed_workspace,
    ) as workspace:
        runtime = build_context_runtime(
            profile=profile,
            model=Model("runtime-v1-smoke", context_window=4_000),
            stream_fn=provider,
            workspace_root=workspace,
            state_root=state_root,
            trace_root=result_store.root / "traces",
        )

        async def read_reference(_tool_call_id: str, _arguments: dict) -> AgentToolResult:
            return AgentToolResult([TextBlock((workspace / "reference.txt").read_text(encoding="utf-8"))])

        runtime.agent.registry.register_tools([
            AgentTool(Tool("eval_read_reference", "Read this smoke fixture.", {}), read_reference)
        ])
        try:
            responses = await runtime.prompt(case.prompt)
            trace = result_store.trace_store.load_all()[-1]
            read_calls = [
                call
                for step in trace.steps
                for call in step.tool_calls
                if call.tool_name == "eval_read_reference"
            ]
            passed = (
                provider.calls == 2
                and responses[-1].text == f"smoke complete {case.case_id}"
                and len(read_calls) == 1
                and read_calls[0].result is not None
                and read_calls[0].result.externalized is profile.enable_tool_result_externalization
            )
            return EvalExecution(case.case_id, trace, responses[-1], artifacts={"passed": passed}), provider.calls
        finally:
            await runtime.close()


async def _run_context_smoke(
    store: _ResultStore,
    root: Path,
    *,
    keep_failed_workspace: bool,
) -> tuple[list[EvalResult], int]:
    results: list[EvalResult] = []
    requests = 0
    full_policy = CompactionPolicy(reserve_tokens=1_000, keep_recent_tokens=500)
    for profile in (ContextManagementProfile.base(), ContextManagementProfile.full(full_policy)):
        for fixture in smoke_fixtures():
            case = EvalCase(fixture.case_id, fixture.case_id, f"Validate {fixture.case_id}.")
            observed_requests = 0

            async def execute(item: EvalCase) -> EvalExecution:
                nonlocal observed_requests
                execution, observed_requests = await _context_execution(
                    item,
                    profile=profile,
                    fixture=fixture,
                    result_store=store,
                    workspace_root=root / "context" / profile.name / fixture.case_id,
                    keep_failed_workspace=keep_failed_workspace,
                )
                return execution

            result = await _run_eval(case, execute)
            store.append(result, experiment="context_ab", profile=profile.name, fixture=fixture, provider_requests=observed_requests)
            results.append(result)
            requests += observed_requests
    return results, requests


async def _tool_execution(name: str, *, state: dict[str, int], fail: bool = False) -> AgentTool:
    async def execute(_tool_call_id: str, _arguments: dict) -> AgentToolResult:
        state["active"] = state.get("active", 0) + 1
        state["max_active"] = max(state.get("max_active", 0), state["active"])
        try:
            await asyncio.sleep(0)
            if fail:
                raise ToolExecutionError("smoke failure")
            return AgentToolResult([TextBlock(name)])
        finally:
            state["active"] -= 1

    return AgentTool(Tool(name, name, {}), execute)


async def _tool_smoke_execution(case: EvalCase) -> EvalExecution:
    state: dict[str, int] = {}
    if case.case_id == "TP_sequential":
        tools = [await _tool_execution("a", state=state), await _tool_execution("b", state=state)]
        mode = ToolExecutionMode.SEQUENTIAL
        expected = ["a", "b"]
        passed_max_active = 1
    elif case.case_id == "TP_parallel":
        tools = [await _tool_execution("a", state=state), await _tool_execution("b", state=state)]
        mode = ToolExecutionMode.PARALLEL
        expected = ["a", "b"]
        passed_max_active = 2
    elif case.case_id == "TP_failure_isolation":
        tools = [await _tool_execution("a", state=state), await _tool_execution("b", state=state, fail=True), await _tool_execution("c", state=state)]
        mode = ToolExecutionMode.PARALLEL
        expected = ["a", "smoke failure", "c"]
        passed_max_active = 2
    else:
        tools = [await _tool_execution("a", state=state), AgentTool(Tool("mutating", "mutating", {}), (await _tool_execution("mutating", state=state)).execute, execution_mode=ToolExecutionMode.SEQUENTIAL)]
        mode = ToolExecutionMode.PARALLEL
        expected = ["a", "mutating"]
        passed_max_active = 1
    committed: list[str] = []
    results = await ToolRuntime(ToolRegistry(tools)).execute_batch(
        [ToolCall(f"{case.case_id}-{index}", tool.tool.name, {}) for index, tool in enumerate(tools)],
        runtime_mode=mode,
        on_result_committed=lambda result: _append(committed, result.text),
    )
    concurrency_ok = (
        state.get("max_active") == passed_max_active
        if case.case_id != "TP_failure_isolation"
        else state.get("max_active", 0) >= passed_max_active
    )
    passed = [result.text for result in results] == expected and committed == expected and concurrency_ok
    return EvalExecution(case.case_id, None, artifacts={"passed": passed})


async def _append(target: list[str], value: str) -> None:
    target.append(value)


async def _run_tool_smoke(store: _ResultStore) -> list[EvalResult]:
    results = []
    for case_id in ("TP_sequential", "TP_parallel", "TP_failure_isolation", "TP_mutation_fallback"):
        case = EvalCase(case_id, case_id, "Run deterministic ToolRuntime smoke.")
        result = await _run_eval(case, _tool_smoke_execution)
        store.append(result, experiment="tool_parallelism", profile=case_id, fixture=None, provider_requests=0)
        results.append(result)
    return results


def _failure(category: str) -> StreamError:
    return StreamError(
        "error",
        AssistantMessage([TextBlock(category)], stop_reason="error"),
        failure=ProviderFailure(category, code=category),
    )


def _no_wait_policy() -> ProviderRetryPolicy:
    async def no_sleep(_delay: float) -> None:
        return None

    return ProviderRetryPolicy(max_retries=1, sleep=no_sleep, random_source=lambda: 0.5)


async def _fault_execution(case: EvalCase, trace_store: JsonlTraceStore) -> EvalExecution:
    case_id = case.case_id
    attempts = 0
    tool_calls = 0

    async def recover(context: Context) -> Context:
        return Context("compacted", list(context.messages), list(context.tools))

    async def failing_tool(_tool_call_id: str, _arguments: dict) -> AgentToolResult:
        nonlocal tool_calls
        tool_calls += 1
        raise ToolExecutionError("expected tool failure")

    async def side_effect_tool(_tool_call_id: str, _arguments: dict) -> AgentToolResult:
        nonlocal tool_calls
        tool_calls += 1
        return AgentToolResult([TextBlock("side effect once")])

    def stream(_model: Model, _context: Context, _options: object | None = None):
        async def events():
            nonlocal attempts
            attempts += 1
            if case_id in {"FI01", "FI03"} and attempts == 1:
                yield _failure("transient")
            elif case_id == "FI02":
                yield _failure("transient")
            elif case_id == "FI04" and attempts == 1:
                yield _failure("context_overflow")
            elif case_id == "FI05":
                yield _failure("context_overflow")
            elif case_id == "FI06" and attempts == 1:
                partial = AssistantMessage([TextBlock("partial")], partial=True)
                yield Start(AssistantMessage([], partial=True))
                yield TextDelta("partial", partial)
                yield _failure("transient")
            elif case_id == "FI07" and attempts == 1:
                yield Start(AssistantMessage([], partial=True))
                yield ToolCallDelta(0, AssistantMessage([], partial=True), id_fragment="partial", name_fragment="sample", arguments_fragment="{")
                yield _failure("transient")
            elif case_id == "FI07" and attempts == 2:
                yield StreamDone(AssistantMessage([ToolCall("complete", "sample", {})], stop_reason="tool_calls"))
            elif case_id in {"FI10", "FI11"} and attempts == 1:
                yield StreamDone(AssistantMessage([ToolCall("complete", "sample", {})], stop_reason="tool_calls"))
            elif case_id == "FI08":
                yield _failure("permanent")
            elif case_id == "FI09":
                yield _failure("unclassified")
            else:
                yield StreamDone(AssistantMessage([TextBlock("fault smoke complete")]))
        return events()

    tools = []
    if case_id in {"FI07", "FI10", "FI11"}:
        tool = failing_tool if case_id == "FI10" else side_effect_tool
        tools.append(AgentTool(Tool("sample", "sample", {}), tool))
    agent = Agent(
        Model("fault-smoke"),
        "",
        tools,
        stream,
        provider_retry_policy=_no_wait_policy(),
        context_overflow_recovery=recover if case_id in {"FI04", "FI05"} else None,
    )
    recorder = TraceRecorder()
    responses, trace = await recorder.capture_run(agent, lambda: agent.run([UserMessage(case_id)]))
    trace_store.append(trace)
    terminal = responses[-1].text
    expected_terminal = "fault smoke complete"
    if case_id == "FI02":
        expected_terminal = "transient"
    elif case_id == "FI05":
        expected_terminal = "context_overflow"
    elif case_id == "FI08":
        expected_terminal = "permanent"
    elif case_id == "FI09":
        expected_terminal = "unclassified"
    passed = terminal == expected_terminal
    if case_id == "FI07":
        passed = passed and tool_calls == 1
    if case_id == "FI10":
        passed = passed and tool_calls == 1
    if case_id == "FI11":
        passed = passed and tool_calls == 1
    if case_id == "FI06":
        passed = passed and all(message.text != "partial" for message in agent.messages if isinstance(message, AssistantMessage))
    return EvalExecution(case.case_id, trace, responses[-1], artifacts={"passed": passed})


async def _run_fault_smoke(store: _ResultStore) -> list[EvalResult]:
    results: list[EvalResult] = []
    for index in range(1, 13):
        case = EvalCase(f"FI{index:02d}", f"FI{index:02d}", "Run deterministic fault smoke.")
        if case.case_id == "FI12":
            execution = lambda item, store=store: _cancel_backoff_execution(item, store.trace_store)
        else:
            execution = lambda item, store=store: _fault_execution(item, store.trace_store)
        result = await _run_eval(case, execution)
        store.append(result, experiment="fault_injection", profile="fault-script", fixture=None, provider_requests=0)
        results.append(result)
    return results


async def _cancel_backoff_execution(case: EvalCase, trace_store: JsonlTraceStore) -> EvalExecution:
    sleep_started = asyncio.Event()
    attempts = 0

    async def wait(_delay: float) -> None:
        sleep_started.set()
        await asyncio.Event().wait()

    async def stream(_model: Model, _context: Context, _options: object | None = None):
        nonlocal attempts
        attempts += 1
        yield _failure("transient")

    agent = Agent(
        Model("fault-smoke"),
        "",
        [],
        stream,
        provider_retry_policy=ProviderRetryPolicy(max_retries=1, sleep=wait, random_source=lambda: 0.5),
    )
    recorder = TraceRecorder()
    task = asyncio.create_task(recorder.capture_run(agent, lambda: agent.run([UserMessage(case.case_id)])))
    await sleep_started.wait()
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    trace = recorder.last_trace
    assert trace is not None
    trace_store.append(trace)
    return EvalExecution(case.case_id, trace, artifacts={"passed": attempts == 1 and agent.messages == [UserMessage(case.case_id)]})


async def run_infrastructure_smoke(result_root: Path, *, keep_failed_workspace: bool = False) -> InfrastructureSmokeReport:
    """Run the non-formal, deterministic Phase 1 smoke through one result pipeline."""
    store = _ResultStore(Path(result_root))
    store.initialize()
    context_results, request_count = await _run_context_smoke(
        store,
        store.root / ".scratch",
        keep_failed_workspace=keep_failed_workspace,
    )
    tool_results = await _run_tool_smoke(store)
    fault_results = await _run_fault_smoke(store)
    cleanup_root = store.root / ".scratch" / "context"
    workspace_cleanup_verified = not cleanup_root.exists() or not any(cleanup_root.rglob("reference.txt"))
    report = InfrastructureSmokeReport(
        tuple([*context_results, *tool_results, *fault_results]),
        context_provider_request_count=request_count,
        context_result_count=len(context_results),
        tool_result_count=len(tool_results),
        fault_result_count=len(fault_results),
        workspace_cleanup_verified=workspace_cleanup_verified,
    )
    (store.root / "report.md").write_text(
        "# Runtime V1 Infrastructure Smoke\n\n"
        f"- Context Provider requests: {report.context_provider_request_count}\n"
        f"- Eval results: {len(report.results)}\n",
        encoding="utf-8",
    )
    return report
