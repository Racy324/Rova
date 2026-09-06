from __future__ import annotations

import asyncio
import sys
from uuid import UUID

import pytest

from rova.ai.events import Start, StreamDone, StreamError, TextDelta
from rova.ai.messages import AssistantMessage, TextBlock, ToolCall, ToolResultMessage, Usage, UserMessage
from rova.ai.models import Model
from rova.ai.tools import Tool
from rova.agent_core.agent import Agent
from rova.agent_core.hooks import HookRegistry, ToolHookPoint
from rova.agent_core.tools import AgentTool, AgentToolResult, ToolExecutionError
from rova.agent_session.agent_session import AgentSession, SessionPersistenceError
from rova.agent_session.compaction import CompactionPolicy
from rova.agent_session.session_store import SessionStoreError
from rova.app.workspace import AlwaysApprove, AlwaysDeny, DefaultCodingToolPolicy, Workspace, build_controlled_coding_tools
from rova.trace import CompactionStatus, CompactionTrigger, RunStatus, TerminationReason, ToolOutcome, TraceRecorder


def make_test_calc_tool() -> AgentTool:
    async def execute(_tool_call_id: str, _params: dict) -> AgentToolResult:
        return AgentToolResult([TextBlock("ok")])

    return AgentTool(Tool("calc", "Test calculation tool", {"expression": str}), execute)


def build_workspace_agent(*, stream_fn, workspace_root, approval_handler) -> Agent:
    return Agent(
        Model(provider="mock"),
        "",
        build_controlled_coding_tools(Workspace(workspace_root), DefaultCodingToolPolicy(), approval_handler),
        stream_fn,
    )


@pytest.mark.asyncio
async def test_recorder_captures_one_finalized_text_turn_and_reported_usage():
    async def stream(model, context, options):
        yield StreamDone(AssistantMessage([TextBlock("done")], usage=Usage(3, 2, 5)))

    agent = Agent(Model(provider="mock"), "", [], stream)

    messages, trace = await TraceRecorder().capture_run(
        agent,
        lambda: agent.run([UserMessage("hello")]),
    )

    assert messages == [AssistantMessage([TextBlock("done")], usage=Usage(3, 2, 5))]
    assert UUID(trace.run_id)
    assert trace.status is RunStatus.COMPLETED
    assert trace.ended_at is not None
    assert trace.duration_ms is not None and trace.duration_ms >= 0
    assert trace.final_message == messages[-1]
    assert trace.usage == Usage(3, 2, 5)
    assert len(trace.turns) == 1
    assert trace.turns[0].turn_index == 1
    assert trace.turns[0].assistant_message == messages[-1]
    assert trace.turns[0].stop_reason == "stop"
    assert trace.turns[0].usage == Usage(3, 2, 5)
    assert trace.turns[0].tool_call_ids == []
    assert trace.tool_executions == []


@pytest.mark.asyncio
async def test_recorder_links_a_tool_result_to_its_finalized_tool_call_id():
    async def stream(model, context, options):
        if not any(isinstance(message, ToolResultMessage) for message in context.messages):
            yield StreamDone(
                AssistantMessage(
                    [ToolCall("calc-1", "calc", {"expression": "2 * 3"})],
                    stop_reason="tool_calls",
                    usage=Usage(5, 1, 6),
                )
            )
            return
        yield StreamDone(AssistantMessage([TextBlock("6")], usage=Usage(7, 2, 9)))

    agent = Agent(Model(provider="mock"), "", [make_test_calc_tool()], stream)
    _, trace = await TraceRecorder().capture_run(agent, lambda: agent.run([UserMessage("calculate")]))

    assert trace.status is RunStatus.COMPLETED
    assert trace.usage == Usage(12, 3, 15)
    assert [turn.turn_index for turn in trace.turns] == [1, 2]
    assert trace.turns[0].tool_call_ids == ["calc-1"]
    assert len(trace.tool_executions) == 1
    execution = trace.tool_executions[0]
    assert execution.tool_call_id == "calc-1"
    assert execution.tool_name == "calc"
    assert execution.arguments == {"expression": "2 * 3"}
    assert execution.turn_index == 1
    assert execution.result == "ok"
    assert execution.is_error is False
    assert execution.duration_ms is not None and execution.duration_ms >= 0


@pytest.mark.asyncio
async def test_recorder_projects_each_tool_call_lifecycle_into_source_ordered_step_records():
    async def stream(model, context, options):
        if not any(isinstance(message, ToolResultMessage) for message in context.messages):
            yield StreamDone(
                AssistantMessage(
                    [
                        ToolCall("calc-1", "calc", {"expression": "2 * 3"}),
                        ToolCall("unknown-1", "unknown", {}),
                    ],
                    stop_reason="tool_calls",
                )
            )
            return
        yield StreamDone(AssistantMessage([TextBlock("done")]))

    agent = Agent(Model(provider="mock"), "", [make_test_calc_tool()], stream)
    _, trace = await TraceRecorder().capture_run(agent, lambda: agent.run([UserMessage("calculate")]))

    first_step = trace.steps[0]
    assert [item.tool_call_id for item in first_step.tool_calls] == ["calc-1", "unknown-1"]
    assert [item.executed for item in first_step.tool_calls] == [True, False]
    assert [item.committed for item in first_step.tool_calls] == [True, True]
    assert first_step.tool_calls[1].outcome is ToolOutcome.TOOL_INPUT_ERROR
    assert first_step.tool_calls[1].failure_stage == "lookup"


@pytest.mark.asyncio
async def test_recorder_keeps_tool_error_distinct_from_harness_failure():
    async def execute(tool_call_id, params):
        raise ToolExecutionError("file not found")

    tool = AgentTool(
        tool=make_test_calc_tool().tool,
        execute=execute,
    )

    async def stream(model, context, options):
        if not any(isinstance(message, ToolResultMessage) for message in context.messages):
            yield StreamDone(
                AssistantMessage(
                    [ToolCall("missing-1", "calc", {"expression": "1 + 1"})],
                    stop_reason="tool_calls",
                )
            )
            return
        result = next(message for message in context.messages if isinstance(message, ToolResultMessage))
        assert result.is_error is True
        yield StreamDone(AssistantMessage([TextBlock("recovered")]))

    agent = Agent(Model(provider="mock"), "", [tool], stream)
    messages, trace = await TraceRecorder().capture_run(agent, lambda: agent.run([UserMessage("read")]))

    assert messages[-1].text == "recovered"
    assert trace.status is RunStatus.COMPLETED
    assert trace.tool_executions[0].is_error is True
    assert trace.tool_executions[0].result == "file not found"
    assert trace.tool_executions[0].outcome is ToolOutcome.TOOL_EXECUTION_ERROR


@pytest.mark.asyncio
async def test_recorder_keeps_executed_uncommitted_tool_call_after_post_hook_harness_failure():
    hooks = HookRegistry()

    async def execute(_tool_call_id, _params):
        return AgentToolResult([TextBlock("raw")])

    async def fail_post(_context):
        raise RuntimeError("post hook failed")

    hooks.register(ToolHookPoint.POST_TOOL_USE, fail_post, source="test.post")

    async def stream(_model, _context, _options):
        yield StreamDone(
            AssistantMessage([ToolCall("calc-1", "calc", {"expression": "1"})], stop_reason="tool_calls")
        )

    agent = Agent(Model(provider="mock"), "", [AgentTool(make_test_calc_tool().tool, execute)], stream, hook_registry=hooks)
    recorder = TraceRecorder()

    with pytest.raises(RuntimeError, match="Lifecycle hook 'test.post' failed"):
        await recorder.capture_run(agent, lambda: agent.run([UserMessage("calculate")]))

    assert recorder.last_trace is not None
    tool_call = recorder.last_trace.steps[0].tool_calls[0]
    assert tool_call.executed is True
    assert tool_call.committed is False
    assert tool_call.result is None
    assert tool_call.ended_at is not None


@pytest.mark.asyncio
async def test_recorder_closes_and_reraises_harness_exception():
    async def execute(tool_call_id, params):
        raise AssertionError("bug")

    tool = AgentTool(make_test_calc_tool().tool, execute)

    async def stream(model, context, options):
        yield StreamDone(
            AssistantMessage(
                [ToolCall("broken-1", "calc", {"expression": "1 + 1"})],
                stop_reason="tool_calls",
            )
        )

    agent = Agent(Model(provider="mock"), "", [tool], stream)
    recorder = TraceRecorder()

    with pytest.raises(AssertionError, match="bug"):
        await recorder.capture_run(agent, lambda: agent.run([UserMessage("run")]))

    trace = recorder.last_trace
    assert trace is not None
    assert trace.status is RunStatus.HARNESS_ERROR
    assert trace.termination_reason is TerminationReason.HARNESS_ERROR
    assert trace.error is not None and trace.error.error_type == "AssertionError"
    assert trace.error.message == "bug"
    assert trace.ended_at is not None
    assert trace.duration_ms is not None and trace.duration_ms >= 0
    assert agent.listeners == []


@pytest.mark.asyncio
async def test_recorder_closes_raw_stream_exception_and_releases_its_listener():
    async def stream(model, context, options):
        raise RuntimeError("stream crashed")
        yield  # pragma: no cover

    agent = Agent(Model(provider="mock"), "", [], stream)
    recorder = TraceRecorder()

    with pytest.raises(RuntimeError, match="stream crashed"):
        await recorder.capture_run(agent, lambda: agent.run([UserMessage("run")]))

    trace = recorder.last_trace
    assert trace is not None
    assert trace.status is RunStatus.PROVIDER_ERROR
    assert trace.termination_reason is TerminationReason.PROVIDER_ERROR
    assert trace.error is not None and trace.error.error_type == "RuntimeError"
    assert trace.ended_at is not None
    assert trace.duration_ms is not None and trace.duration_ms >= 0
    assert agent.listeners == []


@pytest.mark.asyncio
async def test_listener_failure_is_a_harness_error_not_a_provider_error():
    async def stream(model, context, options):
        partial = AssistantMessage([], partial=True)
        yield Start(partial)
        yield TextDelta("done", AssistantMessage([TextBlock("done")], partial=True))
        yield StreamDone(AssistantMessage([TextBlock("done")]))

    agent = Agent(Model(provider="mock"), "", [], stream)

    def broken_listener(event):
        if event.type == "message_update":
            raise RuntimeError("durable listener failed")

    agent.subscribe(broken_listener)
    recorder = TraceRecorder()
    with pytest.raises(RuntimeError, match="durable listener failed"):
        await recorder.capture_run(agent, lambda: agent.run([UserMessage("run")]))

    assert not any(event.type == "provider_error" for event in agent.events)
    assert recorder.last_trace is not None
    assert recorder.last_trace.status is RunStatus.HARNESS_ERROR
    assert recorder.last_trace.termination_reason is TerminationReason.HARNESS_ERROR


@pytest.mark.asyncio
async def test_recorder_preserves_existing_max_turn_return_semantics():
    async def stream(model, context, options):
        yield StreamDone(
            AssistantMessage(
                [ToolCall(f"calc-{len(context.messages)}", "calc", {"expression": "1 + 1"})],
                stop_reason="tool_calls",
            )
        )

    agent = Agent(Model(provider="mock"), "", [make_test_calc_tool()], stream, max_turns=2)
    messages, trace = await TraceRecorder().capture_run(agent, lambda: agent.run([UserMessage("loop")]))

    assert messages[-1].stop_reason == "error"
    assert trace.status is RunStatus.COMPLETED
    assert trace.termination_reason is TerminationReason.MAX_TURNS
    assert trace.final_message is not None
    assert trace.final_message.stop_reason == "error"
    assert len(trace.turns) == 2
    assert len(trace.tool_executions) == 2


@pytest.mark.asyncio
async def test_recorder_does_not_leave_subscriptions_between_captures():
    async def stream(model, context, options):
        yield StreamDone(AssistantMessage([TextBlock("done")]))

    agent = Agent(Model(provider="mock"), "", [], stream)
    recorder = TraceRecorder()

    _, first = await recorder.capture_run(agent, lambda: agent.run([UserMessage("one")]))
    _, second = await recorder.capture_run(agent, lambda: agent.run([UserMessage("two")]))

    assert len(recorder.traces) == 2
    assert first.run_id != second.run_id
    assert [len(trace.turns) for trace in recorder.traces] == [1, 1]
    assert agent.listeners == []


@pytest.mark.asyncio
async def test_trace_observation_preserves_workspace_agent_result_and_side_effect(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    traced_workspace = tmp_path / "traced-workspace"
    traced_workspace.mkdir()
    command = f'"{sys.executable}" -c "from pathlib import Path; Path(\'marker.txt\').write_text(\'ran\')"'

    async def stream(model, context, options):
        if not any(isinstance(message, ToolResultMessage) for message in context.messages):
            yield StreamDone(
                AssistantMessage([ToolCall("shell-1", "shell", {"command": command})], stop_reason="tool_calls")
            )
            return
        yield StreamDone(AssistantMessage([TextBlock("complete")]))

    plain = build_workspace_agent(
        stream_fn=stream,
        workspace_root=workspace,
        approval_handler=AlwaysApprove(),
    )
    traced = build_workspace_agent(
        stream_fn=stream,
        workspace_root=traced_workspace,
        approval_handler=AlwaysApprove(),
    )

    plain_messages = await plain.run([UserMessage("run")])
    traced_messages, trace = await TraceRecorder().capture_run(traced, lambda: traced.run([UserMessage("run")]))

    assert traced_messages == plain_messages
    assert [type(message) for message in traced.messages] == [type(message) for message in plain.messages]
    assert (workspace / "marker.txt").read_text(encoding="utf-8") == "ran"
    assert (traced_workspace / "marker.txt").read_text(encoding="utf-8") == "ran"
    assert trace.status is RunStatus.COMPLETED
    assert trace.termination_reason is TerminationReason.FINAL_RESPONSE
    assert trace.tool_executions[0].tool_name == "shell"


@pytest.mark.asyncio
async def test_recorder_keeps_multi_tool_workspace_loop_order_and_nonzero_shell_as_tool_outcome(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "app.py").write_text("old value\n", encoding="utf-8")
    command = f'"{sys.executable}" -c "raise SystemExit(1)"'

    async def stream(model, context, options):
        results = [message for message in context.messages if isinstance(message, ToolResultMessage)]
        if not results:
            yield StreamDone(
                AssistantMessage([ToolCall("search-1", "search", {"query": "old"})], stop_reason="tool_calls")
            )
            return
        if len(results) == 1:
            yield StreamDone(
                AssistantMessage([ToolCall("read-1", "read", {"path": "app.py"})], stop_reason="tool_calls")
            )
            return
        if len(results) == 2:
            yield StreamDone(
                AssistantMessage(
                    [ToolCall("edit-1", "edit", {"path": "app.py", "old_text": "old", "new_text": "new"})],
                    stop_reason="tool_calls",
                )
            )
            return
        if len(results) == 3:
            yield StreamDone(
                AssistantMessage([ToolCall("shell-1", "shell", {"command": command})], stop_reason="tool_calls")
            )
            return
        assert "exit_code: 1" in results[-1].text
        yield StreamDone(AssistantMessage([TextBlock("complete")]))

    agent = build_workspace_agent(
        stream_fn=stream,
        workspace_root=workspace,
        approval_handler=AlwaysApprove(),
    )
    _, trace = await TraceRecorder().capture_run(agent, lambda: agent.run([UserMessage("fix it")]))

    assert trace.status is RunStatus.COMPLETED
    assert [turn.turn_index for turn in trace.turns] == [1, 2, 3, 4, 5]
    assert [item.tool_name for item in trace.tool_executions] == ["search", "read", "edit", "shell"]
    assert [item.tool_call_id for item in trace.tool_executions] == ["search-1", "read-1", "edit-1", "shell-1"]
    assert [item.turn_index for item in trace.tool_executions] == [1, 2, 3, 4]
    assert all(item.duration_ms is not None and item.duration_ms >= 0 for item in trace.tool_executions)
    assert trace.tool_executions[-1].is_error is False
    assert trace.tool_executions[-1].outcome is ToolOutcome.COMMAND_NONZERO_EXIT
    assert trace.tool_executions[-1].command == command
    assert trace.tool_executions[-1].exit_code == 1
    assert trace.tool_executions[-1].timed_out is False
    assert "exit_code: 1" in trace.tool_executions[-1].result
    assert (workspace / "app.py").read_text(encoding="utf-8") == "new value\n"


@pytest.mark.asyncio
async def test_recorder_uses_structured_approval_metadata_without_parsing_tool_text(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    async def stream(model, context, options):
        if not any(isinstance(message, ToolResultMessage) for message in context.messages):
            yield StreamDone(
                AssistantMessage([ToolCall("write-1", "write", {"path": "new.txt", "content": "no"})], stop_reason="tool_calls")
            )
            return
        yield StreamDone(AssistantMessage([TextBlock("approval observed")]))

    agent = build_workspace_agent(
        stream_fn=stream,
        workspace_root=workspace,
        approval_handler=AlwaysDeny(),
    )
    _, trace = await TraceRecorder().capture_run(agent, lambda: agent.run([UserMessage("write")]))

    execution = trace.tool_executions[0]
    assert execution.outcome is ToolOutcome.APPROVAL_DENIED
    assert execution.policy_decision == "require_approval"
    assert execution.approval_required is True
    assert execution.approval_decision == "deny"
    assert not (workspace / "new.txt").exists()


@pytest.mark.asyncio
async def test_recorder_omits_partial_stream_messages_and_keeps_final_snapshot_only():
    async def stream(model, context, options):
        partial = AssistantMessage([TextBlock("par")], partial=True)
        yield Start(partial)
        yield TextDelta("tial", partial)
        yield StreamDone(AssistantMessage([TextBlock("final")]))

    agent = Agent(Model(provider="mock"), "", [], stream)
    _, trace = await TraceRecorder().capture_run(agent, lambda: agent.run([UserMessage("hello")]))

    assert len(trace.turns) == 1
    assert trace.turns[0].assistant_message == AssistantMessage([TextBlock("final")])
    assert trace.final_message == AssistantMessage([TextBlock("final")])


@pytest.mark.asyncio
async def test_each_session_capture_contains_only_its_current_run(tmp_path):
    async def stream(model, context, options):
        latest_user = next(message for message in reversed(context.messages) if isinstance(message, UserMessage))
        yield StreamDone(AssistantMessage([TextBlock(f"answer: {latest_user.content}")]))

    agent = Agent(Model(provider="mock"), "", [], stream)
    session = AgentSession.create(agent, session_root=tmp_path / "sessions")
    recorder = TraceRecorder()

    _, first = await recorder.capture_run(
        agent,
        lambda: session.prompt("one"),
        session_id=session.session_id,
    )
    _, second = await recorder.capture_run(
        agent,
        lambda: session.prompt("two"),
        session_id=session.session_id,
    )

    assert first.session_id == session.session_id == second.session_id
    assert [turn.assistant_message.text for turn in first.turns] == ["answer: one"]
    assert [turn.assistant_message.text for turn in second.turns] == ["answer: two"]
    assert first.compactions == second.compactions == []
    assert len(agent.messages) == 4


@pytest.mark.asyncio
async def test_recorder_marks_cancelled_capture_aborted_and_reraises():
    entered_stream = asyncio.Event()

    async def stream(model, context, options):
        entered_stream.set()
        await asyncio.Event().wait()
        yield  # pragma: no cover

    agent = Agent(Model(provider="mock"), "", [], stream)
    recorder = TraceRecorder()
    pending = asyncio.create_task(recorder.capture_run(agent, lambda: agent.run([UserMessage("wait")])))
    await entered_stream.wait()
    pending.cancel()

    with pytest.raises(asyncio.CancelledError):
        await pending

    trace = recorder.last_trace
    assert trace is not None
    assert trace.status is RunStatus.ABORTED
    assert trace.error is not None and trace.error.error_type == "CancelledError"
    assert trace.ended_at is not None
    assert agent.listeners == []


@pytest.mark.asyncio
async def test_recorder_keeps_terminal_provider_error_as_structured_finalized_fact():
    async def stream(model, context, options):
        yield StreamError(
            "error",
            AssistantMessage([TextBlock("provider failure")], stop_reason="error"),
        )

    agent = Agent(Model(provider="mock"), "", [], stream)
    _, trace = await TraceRecorder().capture_run(agent, lambda: agent.run([UserMessage("hello")]))

    assert trace.status is RunStatus.PROVIDER_ERROR
    assert trace.termination_reason is TerminationReason.PROVIDER_ERROR
    assert trace.final_message is not None
    assert trace.final_message.stop_reason == "error"
    assert trace.error is not None
    assert trace.error.error_type == "StreamError"


@pytest.mark.asyncio
async def test_recorder_captures_automatic_compaction_from_agent_events(tmp_path):
    class FixedEstimator:
        def estimate_messages(self, messages):
            return len(messages) * 10

    async def stream(model, context, options):
        yield StreamDone(AssistantMessage([TextBlock("done")], usage=Usage(80, 10, 90)))

    async def summarize(request):
        return "summary"

    agent = Agent(Model("mock", context_window=100), "", [], stream)
    session = AgentSession.create(
        agent,
        session_root=tmp_path,
        compaction_policy=CompactionPolicy(10, 10),
        token_estimator=FixedEstimator(),
        summary_fn=summarize,
    )
    _, trace = await TraceRecorder().capture_run(
        agent,
        lambda: session.prompt("hello"),
        session_id=session.session_id,
    )

    assert trace.session_id == session.session_id
    assert len(trace.compactions) == 1
    compaction = trace.compactions[0]
    assert compaction.trigger is CompactionTrigger.AUTOMATIC
    assert compaction.status is CompactionStatus.COMPLETED
    assert compaction.started_at is not None
    assert compaction.ended_at is not None
    assert compaction.duration_ms is not None and compaction.duration_ms >= 0
    assert compaction.pressure_before == 90


@pytest.mark.asyncio
async def test_recorder_captures_non_durable_compaction_failure_without_faulting_session(tmp_path):
    class FixedEstimator:
        def estimate_messages(self, messages):
            return len(messages) * 10

    async def stream(model, context, options):
        yield StreamDone(AssistantMessage([TextBlock("done")], usage=Usage(80, 10, 90)))

    async def failing_summary(request):
        raise RuntimeError("summary provider unavailable")

    agent = Agent(Model("mock", context_window=100), "", [], stream)
    session = AgentSession.create(
        agent,
        session_root=tmp_path,
        compaction_policy=CompactionPolicy(10, 10),
        token_estimator=FixedEstimator(),
        summary_fn=failing_summary,
    )
    messages, trace = await TraceRecorder().capture_run(
        agent,
        lambda: session.prompt("hello"),
        session_id=session.session_id,
    )

    assert messages[-1].text == "done"
    assert session.faulted is False
    assert isinstance(session.last_maintenance_error, RuntimeError)
    assert len(trace.compactions) == 1
    compaction = trace.compactions[0]
    assert compaction.status is CompactionStatus.FAILED
    assert compaction.error is not None
    assert compaction.error.error_type == "RuntimeError"
    assert compaction.error.message == "summary provider unavailable"


@pytest.mark.asyncio
async def test_session_persistence_failure_overrides_prior_final_response_termination(tmp_path, monkeypatch):
    class FixedEstimator:
        def estimate_messages(self, messages):
            return len(messages) * 10

    async def stream(model, context, options):
        yield StreamDone(AssistantMessage([TextBlock("done")], usage=Usage(80, 10, 90)))

    async def summarize(request):
        return "summary"

    agent = Agent(Model("mock", context_window=100), "", [], stream)
    session = AgentSession.create(
        agent,
        session_root=tmp_path,
        compaction_policy=CompactionPolicy(10, 10),
        token_estimator=FixedEstimator(),
        summary_fn=summarize,
    )
    monkeypatch.setattr(
        session._durable_session.store,
        "append_compaction",
        lambda *_: (_ for _ in ()).throw(SessionStoreError("disk full")),
    )
    recorder = TraceRecorder()

    with pytest.raises(SessionPersistenceError, match="compaction"):
        await recorder.capture_run(
            agent,
            lambda: session.prompt("hello"),
            session_id=session.session_id,
        )

    trace = recorder.last_trace
    assert trace is not None
    assert trace.status is RunStatus.SESSION_PERSISTENCE_ERROR
    assert trace.termination_reason is TerminationReason.SESSION_PERSISTENCE_ERROR
    assert trace.error is not None and trace.error.error_type == "SessionPersistenceError"
