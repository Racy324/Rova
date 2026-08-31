import asyncio
import io
import json
from pathlib import Path

import pytest

from rova.ai.events import TextDelta
from rova.ai.messages import AssistantMessage, TextBlock, UserMessage
from rova.ai.models import Model
from rova.agent_core.events import AgentEvent, AgentTerminationReason
from rova.agent_session.session_store import JsonlSessionStore
from rova.app.workspace.approval import ApprovalDecision, ApprovalRequest
from rova.app import tui_gateway
from rova.app.tui_gateway import GatewayApprovalHandler, TuiGateway


class FakeAgent:
    def __init__(self, messages=None):
        self.model = Model("tui-test", provider="test")
        self.messages = list(messages or [])
        self._listeners = []

    def subscribe(self, listener):
        self._listeners.append(listener)

        def unsubscribe():
            self._listeners.remove(listener)

        return unsubscribe

    async def emit(self, event):
        for listener in list(self._listeners):
            result = listener(event)
            if asyncio.iscoroutine(result):
                await result


class FakeSession:
    def __init__(self, session_id):
        self.session_id = session_id
        self.closed = False

    def close(self):
        self.closed = True


class FakeRuntime:
    def __init__(self, session_id, messages=None):
        self.agent = FakeAgent(messages)
        self.session = FakeSession(session_id)
        self.workspace = None
        self.terminal_backend = type("Terminal", (), {
            "environment": type("Environment", (), {
                "kind": "local",
                "executor": "cmd.exe",
                "cwd": "C:/workspace",
                "is_filesystem_sandboxed": False,
            })(),
        })()
        self.source_store = None
        self.skill_catalog_snapshot = type("Catalog", (), {"skills": ()})()
        self.prompt_calls = []
        self.prompt_started = asyncio.Event()
        self.prompt_release = asyncio.Event()
        self.close_calls = 0

    async def prompt(self, text):
        self.prompt_calls.append(text)
        self.prompt_started.set()
        await self.prompt_release.wait()
        return [AssistantMessage([TextBlock("done")])]

    async def close(self):
        self.close_calls += 1
        self.session.close()


def _request(request_id, method, params=None):
    return json.dumps({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params or {}})


@pytest.mark.asyncio
async def test_gateway_reports_status_and_creates_a_new_runtime(tmp_path):
    frames = []
    created_ids = []

    def factory(session_id, approval_handler):
        created_ids.append(session_id)
        return FakeRuntime(session_id or "new-session")

    gateway = TuiGateway(runtime_factory=factory, session_store=JsonlSessionStore(tmp_path), emit_frame=frames.append)

    await gateway.initialize()
    await gateway.process_line(_request("status", "runtime.status"))
    await gateway.process_line(_request("new", "session.new"))

    assert created_ids == [None, None]
    status = next(frame for frame in frames if frame.get("id") == "status")["result"]
    assert status["session_id"] == "new-session"
    assert status["terminal_backend"] == {
        "kind": "local",
        "executor": "cmd.exe",
        "cwd": "C:/workspace",
        "is_filesystem_sandboxed": False,
    }
    assert next(frame for frame in frames if frame.get("id") == "new")["result"]["session_id"] == "new-session"
    assert any(frame.get("params", {}).get("type") == "session.changed" for frame in frames)


@pytest.mark.asyncio
async def test_gateway_lists_persisted_session_summaries_and_resumes_through_runtime_factory(tmp_path):
    store = JsonlSessionStore(tmp_path)
    durable = store.create()
    durable.append(UserMessage("restore this discussion"))
    frames = []
    restored_messages = [UserMessage("restore this discussion"), AssistantMessage([TextBlock("restored")])]
    factory_calls = []

    def factory(session_id, approval_handler):
        factory_calls.append(session_id)
        return FakeRuntime(session_id or "new-session", restored_messages if session_id else [])

    gateway = TuiGateway(runtime_factory=factory, session_store=store, emit_frame=frames.append)
    await gateway.initialize()
    await gateway.process_line(_request("list", "session.list"))
    await gateway.process_line(_request("resume", "session.resume", {"session_id": durable.session_id}))

    assert next(frame for frame in frames if frame.get("id") == "list")["result"][0]["first_user_preview"] == "restore this discussion"
    assert factory_calls == [None, durable.session_id]
    transcript = next(frame for frame in frames if frame.get("id") == "resume")["result"]["transcript"]
    assert transcript == [
        {"role": "user", "text": "restore this discussion"},
        {"role": "assistant", "text": "restored"},
    ]


@pytest.mark.asyncio
async def test_gateway_accepts_prompt_and_forwards_existing_agent_stream_events(tmp_path):
    frames = []
    runtime = FakeRuntime("session-1")
    gateway = TuiGateway(
        runtime_factory=lambda session_id, approval_handler: runtime,
        session_store=JsonlSessionStore(tmp_path),
        emit_frame=frames.append,
    )
    await gateway.initialize()

    await gateway.process_line(_request("prompt", "prompt.submit", {"text": "hello"}))
    await runtime.prompt_started.wait()
    partial = AssistantMessage([TextBlock("")], partial=True)
    await runtime.agent.emit(AgentEvent("message_start", message=partial))
    await runtime.agent.emit(AgentEvent("message_update", message=partial, assistant_message_event=TextDelta("hi", partial)))
    await runtime.agent.emit(AgentEvent("tool_execution_start", tool_call_id="call-1", tool_name="read", args={"path": "a.py"}))
    await runtime.agent.emit(AgentEvent("tool_execution_end", tool_call_id="call-1", tool_name="read", result="content", is_error=False))
    await runtime.agent.emit(AgentEvent("agent_end", termination_reason=AgentTerminationReason.FINAL_RESPONSE))
    runtime.prompt_release.set()
    await gateway.wait_for_prompt()

    assert runtime.prompt_calls == ["hello"]
    assert next(frame for frame in frames if frame.get("id") == "prompt")["result"] == {"accepted": True}
    event_types = [
        frame["params"]["type"]
        for frame in frames
        if frame.get("method") == "event" and frame["params"]["type"] != "session.changed"
    ]
    assert event_types == ["assistant.start", "assistant.delta", "tool.start", "tool.end", "run.finished"]


@pytest.mark.asyncio
async def test_gateway_cancels_the_active_prompt_and_allows_a_follow_up_prompt(tmp_path):
    frames = []
    runtime = FakeRuntime("session-1")
    gateway = TuiGateway(
        runtime_factory=lambda session_id, approval_handler: runtime,
        session_store=JsonlSessionStore(tmp_path),
        emit_frame=frames.append,
    )
    await gateway.initialize()

    await gateway.process_line(_request("first", "prompt.submit", {"text": "cancel this"}))
    await runtime.prompt_started.wait()
    await gateway.process_line(_request("cancel", "prompt.cancel"))
    runtime.prompt_release.set()
    await gateway.wait_for_prompt()

    assert next(frame for frame in frames if frame.get("id") == "cancel")["result"] == {"cancelled": True}
    assert any(frame.get("params", {}).get("type") == "run.cancelled" for frame in frames)

    await gateway.process_line(_request("second", "prompt.submit", {"text": "next turn"}))
    assert next(frame for frame in frames if frame.get("id") == "second")["result"] == {"accepted": True}
    await gateway.wait_for_prompt()


@pytest.mark.asyncio
async def test_gateway_approval_resolves_only_matching_request_and_eof_fails_closed():
    events = []
    handler = GatewayApprovalHandler(lambda event_type, payload: events.append((event_type, payload)))
    request = ApprovalRequest("shell", {"command": "python -V"}, "requires approval", "Run command:\npython -V\ncwd:\nC:/work")

    pending = asyncio.create_task(handler.request_approval(request))
    await asyncio.sleep(0)
    request_id = events[0][1]["request_id"]

    assert handler.respond("unknown", "allow") is False
    assert handler.respond(request_id, "allow") is True
    assert await pending is ApprovalDecision.APPROVE

    pending = asyncio.create_task(handler.request_approval(request))
    await asyncio.sleep(0)
    handler.deny_pending()
    assert await pending is ApprovalDecision.DENY


@pytest.mark.asyncio
async def test_gateway_approval_cancellation_propagates_to_the_active_turn():
    handler = GatewayApprovalHandler(lambda _event_type, _payload: None)
    request = ApprovalRequest("shell", {"command": "python -V"}, "requires approval", "Run command")
    pending = asyncio.create_task(handler.request_approval(request))
    await asyncio.sleep(0)

    pending.cancel()

    with pytest.raises(asyncio.CancelledError):
        await pending
    assert handler._pending == {}


@pytest.mark.asyncio
async def test_gateway_returns_json_rpc_errors_for_malformed_and_unknown_requests(tmp_path):
    frames = []
    gateway = TuiGateway(
        runtime_factory=lambda session_id, approval_handler: FakeRuntime("session-1"),
        session_store=JsonlSessionStore(tmp_path),
        emit_frame=frames.append,
    )
    await gateway.initialize()

    await gateway.process_line("not json")
    await gateway.process_line(_request("unknown", "unknown.method"))

    assert frames[-2]["error"]["code"] == -32700
    assert frames[-1]["error"]["code"] == -32601


@pytest.mark.asyncio
async def test_gateway_close_denies_pending_approval_and_closes_the_idle_session(tmp_path):
    frames = []
    runtime = FakeRuntime("session-1")
    gateway = TuiGateway(
        runtime_factory=lambda session_id, approval_handler: runtime,
        session_store=JsonlSessionStore(tmp_path),
        emit_frame=frames.append,
    )
    await gateway.initialize()
    pending = asyncio.create_task(gateway.approval_handler.request_approval(
        ApprovalRequest("edit", {"path": "a.py"}, "requires approval", "Edit file")
    ))
    await asyncio.sleep(0)

    await gateway.process_line(_request("close", "runtime.close"))

    assert await pending is ApprovalDecision.DENY
    assert runtime.session.closed is True
    assert runtime.close_calls == 1
    assert frames[-1]["result"] == {"closed": True}


@pytest.mark.asyncio
async def test_gateway_routes_approval_response_and_eof_shutdown_fails_closed(tmp_path):
    frames = []
    runtime = FakeRuntime("session-1")
    gateway = TuiGateway(
        runtime_factory=lambda session_id, approval_handler: runtime,
        session_store=JsonlSessionStore(tmp_path),
        emit_frame=frames.append,
    )
    await gateway.initialize()
    request = ApprovalRequest("edit", {"path": "a.py"}, "requires approval", "Edit file")
    pending = asyncio.create_task(gateway.approval_handler.request_approval(request))
    await asyncio.sleep(0)
    approval_event = next(frame for frame in frames if frame.get("params", {}).get("type") == "approval.request")

    await gateway.process_line(_request("approve", "approval.respond", {
        "request_id": approval_event["params"]["payload"]["request_id"], "decision": "allow",
    }))

    assert await pending is ApprovalDecision.APPROVE
    pending = asyncio.create_task(gateway.approval_handler.request_approval(request))
    await asyncio.sleep(0)
    await gateway.shutdown()
    assert await pending is ApprovalDecision.DENY


def test_gateway_stdout_writer_emits_one_utf8_json_rpc_frame_only(monkeypatch):
    stdout = io.StringIO()
    monkeypatch.setattr(tui_gateway.sys, "stdout", stdout)

    tui_gateway._write_jsonrpc_frame({"jsonrpc": "2.0", "method": "event", "params": {"type": "assistant.delta", "payload": {"text": "中文 ✅"}}})

    assert json.loads(stdout.getvalue()) == {
        "jsonrpc": "2.0", "method": "event", "params": {"type": "assistant.delta", "payload": {"text": "中文 ✅"}},
    }


def test_gateway_main_configures_protocol_stdio_before_serving(monkeypatch):
    calls: list[str] = []

    monkeypatch.setattr(tui_gateway, "_configure_gateway_stdio", lambda: calls.append("configured"), raising=False)
    monkeypatch.setattr(tui_gateway.asyncio, "run", lambda awaitable: awaitable.close())

    tui_gateway.main()

    assert calls == ["configured"]


def test_gateway_protocol_stdio_uses_utf8_for_all_three_standard_streams(monkeypatch):
    configured: list[tuple[str, str, str]] = []

    class Stream:
        def __init__(self, name: str) -> None:
            self.name = name

        def reconfigure(self, *, encoding: str, errors: str) -> None:
            configured.append((self.name, encoding, errors))

    monkeypatch.setattr(tui_gateway.sys, "stdin", Stream("stdin"))
    monkeypatch.setattr(tui_gateway.sys, "stdout", Stream("stdout"))
    monkeypatch.setattr(tui_gateway.sys, "stderr", Stream("stderr"))

    tui_gateway._configure_gateway_stdio()

    assert configured == [
        ("stdin", "utf-8", "backslashreplace"),
        ("stdout", "utf-8", "backslashreplace"),
        ("stderr", "utf-8", "backslashreplace"),
    ]
