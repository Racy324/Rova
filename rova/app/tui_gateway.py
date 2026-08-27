"""Thin stdio JSON-RPC adapter between the Ink TUI and ``RovaRuntime``."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
from typing import Any, Optional
from uuid import uuid4

from rova.ai.events import TextDelta
from rova.ai.messages import AssistantMessage, ToolResultMessage, UserMessage
from rova.agent_core.events import AgentEvent
from rova.agent_session.session_store import JsonlSessionStore

from .workspace.approval import ApprovalDecision, ApprovalHandler, ApprovalRequest
from .runtime import RovaRuntime
from .settings import AppSettings


JSONRPC_VERSION = "2.0"


class GatewayApprovalHandler:
    """An ``ApprovalHandler`` that delegates one decision to the connected TUI."""

    def __init__(self, emit_event: Callable[[str, dict[str, Any]], None]) -> None:
        self._emit_event = emit_event
        self._pending: dict[str, asyncio.Future[ApprovalDecision]] = {}

    async def request_approval(self, request: ApprovalRequest) -> ApprovalDecision:
        request_id = uuid4().hex
        future: asyncio.Future[ApprovalDecision] = asyncio.get_running_loop().create_future()
        self._pending[request_id] = future
        self._emit_event("approval.request", _approval_payload(request_id, request))
        try:
            return await future
        except asyncio.CancelledError:
            return ApprovalDecision.DENY
        finally:
            self._pending.pop(request_id, None)

    def respond(self, request_id: str, decision: str) -> bool:
        future = self._pending.get(request_id)
        if future is None or future.done() or decision not in {"allow", "deny"}:
            return False
        future.set_result(ApprovalDecision.APPROVE if decision == "allow" else ApprovalDecision.DENY)
        return True

    def deny_pending(self) -> None:
        for future in list(self._pending.values()):
            if not future.done():
                future.set_result(ApprovalDecision.DENY)


RuntimeFactory = Callable[[Optional[str], Optional[ApprovalHandler]], RovaRuntime]
FrameWriter = Callable[[dict[str, Any]], None]


class TuiGateway:
    """One-process adapter; it owns no Agent, Session, Tool, or policy semantics."""

    def __init__(
        self,
        *,
        runtime_factory: RuntimeFactory,
        session_store: JsonlSessionStore,
        emit_frame: FrameWriter,
        permission_mode: str = "ask",
    ) -> None:
        self._runtime_factory = runtime_factory
        self._session_store = session_store
        self._emit_frame = emit_frame
        self._permission_mode = permission_mode
        self.runtime: RovaRuntime | None = None
        self.approval_handler = GatewayApprovalHandler(self._emit_event)
        self._unsubscribe_agent: Callable[[], None] | None = None
        self._prompt_task: asyncio.Task[None] | None = None
        self._closed = False

    async def initialize(self) -> None:
        await self._replace_runtime(None)

    async def process_line(self, line: str) -> None:
        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            self._emit_error(None, -32700, "Parse error")
            return
        if not isinstance(request, dict) or request.get("jsonrpc") != JSONRPC_VERSION or not isinstance(request.get("method"), str):
            self._emit_error(request.get("id") if isinstance(request, dict) else None, -32600, "Invalid Request")
            return
        request_id = request.get("id")
        params = request.get("params", {})
        if not isinstance(params, dict):
            self._emit_error(request_id, -32602, "Invalid params")
            return
        try:
            result = await self._dispatch(request["method"], params)
        except _GatewayRequestError as error:
            self._emit_error(request_id, error.code, error.message)
            return
        except Exception as error:
            self._emit_event("error", {"error_type": type(error).__name__, "message": _safe_error_message(error)})
            self._emit_error(request_id, -32603, "Internal error")
            return
        if request_id is not None:
            self._emit_frame({"jsonrpc": JSONRPC_VERSION, "id": request_id, "result": result})

    async def wait_for_prompt(self) -> None:
        if self._prompt_task is not None:
            await self._prompt_task

    async def shutdown(self) -> None:
        self.approval_handler.deny_pending()
        if self._prompt_task is not None and not self._prompt_task.done():
            await self._prompt_task
        self._close_runtime()
        self._closed = True

    async def _dispatch(self, method: str, params: dict[str, Any]) -> Any:
        if method == "runtime.status":
            return self._status()
        if method == "prompt.submit":
            text = params.get("text")
            if not isinstance(text, str) or not text.strip():
                raise _GatewayRequestError(-32602, "text must be a non-empty string")
            if self._prompt_task is not None and not self._prompt_task.done():
                raise _GatewayRequestError(-32000, "a prompt is already active")
            self._prompt_task = asyncio.create_task(self._run_prompt(text))
            return {"accepted": True}
        if method == "session.list":
            limit = params.get("limit")
            if limit is not None and (not isinstance(limit, int) or isinstance(limit, bool) or limit < 0):
                raise _GatewayRequestError(-32602, "limit must be a non-negative integer")
            return [asdict(summary) for summary in self._session_store.list_sessions(limit)]
        if method == "session.new":
            self._ensure_idle()
            await self._replace_runtime(None)
            return self._status()
        if method == "session.resume":
            session_id = params.get("session_id")
            if not isinstance(session_id, str) or not session_id:
                raise _GatewayRequestError(-32602, "session_id must be a non-empty string")
            self._ensure_idle()
            try:
                await self._replace_runtime(session_id)
            except Exception as error:
                raise _GatewayRequestError(-32004, _safe_error_message(error)) from error
            return {"status": self._status(), "transcript": _transcript(self._runtime().agent.messages)}
        if method == "approval.respond":
            request_id = params.get("request_id")
            decision = params.get("decision")
            if not isinstance(request_id, str) or decision not in {"allow", "deny"}:
                raise _GatewayRequestError(-32602, "request_id and allow|deny decision are required")
            if not self.approval_handler.respond(request_id, decision):
                raise _GatewayRequestError(-32001, "approval request is unavailable")
            return {"accepted": True}
        if method == "runtime.close":
            self._ensure_idle()
            self.approval_handler.deny_pending()
            self._close_runtime()
            self._closed = True
            return {"closed": True}
        raise _GatewayRequestError(-32601, "Method not found")

    async def _run_prompt(self, text: str) -> None:
        try:
            await self._runtime().prompt(text)
        except Exception as error:
            self._emit_event("error", {"error_type": type(error).__name__, "message": _safe_error_message(error)})

    async def _replace_runtime(self, session_id: str | None) -> None:
        self._close_runtime()
        self.approval_handler = GatewayApprovalHandler(self._emit_event)
        approval_handler: ApprovalHandler | None = self.approval_handler if self._permission_mode == "ask" else None
        self.runtime = self._runtime_factory(session_id, approval_handler)
        self._unsubscribe_agent = self.runtime.agent.subscribe(self._on_agent_event)
        self._emit_event("session.changed", {"session_id": self.runtime.session.session_id})

    def _close_runtime(self) -> None:
        if self._unsubscribe_agent is not None:
            self._unsubscribe_agent()
            self._unsubscribe_agent = None
        if self.runtime is not None:
            self.runtime.session.close()
            self.runtime = None

    def _ensure_idle(self) -> None:
        if self._prompt_task is not None and not self._prompt_task.done():
            raise _GatewayRequestError(-32000, "a prompt is already active")

    def _runtime(self) -> RovaRuntime:
        if self.runtime is None:
            raise _GatewayRequestError(-32002, "runtime is not initialized")
        return self.runtime

    def _status(self) -> dict[str, Any]:
        runtime = self._runtime()
        return {
            "model": runtime.agent.model.model,
            "workspace": str(runtime.workspace.root) if runtime.workspace is not None else None,
            "web_enabled": runtime.source_store is not None,
            "permission_mode": self._permission_mode,
            "session_id": runtime.session.session_id,
            "skill_count": len(runtime.skill_catalog_snapshot.skills),
        }

    def _on_agent_event(self, event: AgentEvent) -> None:
        if event.type == "message_start":
            self._emit_event("assistant.start", {})
        elif event.type == "message_update" and isinstance(event.assistant_message_event, TextDelta):
            self._emit_event("assistant.delta", {"text": event.assistant_message_event.delta})
        elif event.type == "message_end" and event.message is not None:
            self._emit_event("assistant.end", {"text": event.message.text})
        elif event.type == "tool_execution_start":
            self._emit_event("tool.start", {
                "tool_call_id": event.tool_call_id,
                "tool_name": event.tool_name,
                "summary": _tool_summary(event.tool_name, event.args or {}),
            })
        elif event.type == "tool_execution_end":
            self._emit_event("tool.end", {
                "tool_call_id": event.tool_call_id,
                "tool_name": event.tool_name,
                "status": "error" if event.is_error else "success",
                "result_preview": _preview(event.result or ""),
            })
        elif event.type == "provider_error":
            self._emit_event("error", {
                "error_type": event.error_type or "ProviderError",
                "message": event.error_message or "Provider error",
            })
        elif event.type == "agent_end":
            self._emit_event("run.finished", {
                "termination_reason": event.termination_reason.value if event.termination_reason is not None else None,
            })

    def _emit_event(self, event_type: str, payload: dict[str, Any]) -> None:
        self._emit_frame({"jsonrpc": JSONRPC_VERSION, "method": "event", "params": {"type": event_type, "payload": payload}})

    def _emit_error(self, request_id: Any, code: int, message: str) -> None:
        self._emit_frame({"jsonrpc": JSONRPC_VERSION, "id": request_id, "error": {"code": code, "message": message}})


class _GatewayRequestError(RuntimeError):
    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def _approval_payload(request_id: str, request: ApprovalRequest) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "request_id": request_id,
        "tool_name": request.tool_name,
        "summary": request.summary,
        "policy_reason": request.policy_reason,
    }
    if isinstance(request.arguments.get("command"), str):
        payload["command"] = request.arguments["command"]
    if isinstance(request.arguments.get("path"), str):
        payload["path"] = request.arguments["path"]
    cwd = request.summary.partition("\ncwd:\n")[2]
    if cwd:
        payload["cwd"] = cwd
    return payload


def _tool_summary(tool_name: str | None, arguments: Mapping[str, Any]) -> str:
    for key in ("command", "path", "query", "source_id", "name"):
        value = arguments.get(key)
        if isinstance(value, str) and value:
            return _preview(value)
    return tool_name or "tool"


def _transcript(messages: list[Any]) -> list[dict[str, Any]]:
    rendered: list[dict[str, Any]] = []
    for message in messages:
        if isinstance(message, UserMessage):
            rendered.append({"role": "user", "text": message.content})
        elif isinstance(message, AssistantMessage):
            if message.text:
                rendered.append({"role": "assistant", "text": message.text})
            for tool_call in message.tool_calls:
                rendered.append({"role": "tool", "tool_name": tool_call.name, "status": "called", "summary": _tool_summary(tool_call.name, tool_call.arguments)})
        elif isinstance(message, ToolResultMessage):
            rendered.append({
                "role": "tool",
                "tool_name": message.tool_name,
                "status": "error" if message.is_error else "success",
                "summary": _preview(message.text),
            })
    return rendered


def _preview(value: str, *, max_length: int = 240) -> str:
    normalized = " ".join(value.split())
    return normalized if len(normalized) <= max_length else f"{normalized[:max_length - 1]}…"


def _safe_error_message(error: Exception) -> str:
    message = str(error).strip()
    return message or type(error).__name__


async def _serve_default_gateway() -> None:
    from .cli import _build_runtime_from_args, parse_rova_cli_args

    raw_args = os.environ.get("ROVA_TUI_ARGS_JSON", "[]")
    try:
        argv = json.loads(raw_args)
    except json.JSONDecodeError as error:
        raise RuntimeError("ROVA_TUI_ARGS_JSON is invalid") from error
    if not isinstance(argv, list) or not all(isinstance(item, str) for item in argv):
        raise RuntimeError("ROVA_TUI_ARGS_JSON must contain a string argument list")
    args = parse_rova_cli_args(argv)
    app_settings = AppSettings.from_env()
    gateway = TuiGateway(
        runtime_factory=lambda session_id, approval_handler: _build_runtime_from_args(
            args,
            approval_handler=approval_handler,
            session_id=session_id,
            app_settings=app_settings,
        ),
        session_store=gateway_session_store(args.data_dir, app_settings),
        emit_frame=_write_jsonrpc_frame,
        permission_mode=args.permission,
    )
    await gateway.initialize()
    try:
        while not gateway._closed:
            line = await asyncio.to_thread(sys.stdin.readline)
            if line == "":
                break
            await gateway.process_line(line)
    finally:
        await gateway.shutdown()


def gateway_session_store(data_dir: Path | None, app_settings: AppSettings) -> JsonlSessionStore:
    return JsonlSessionStore(app_settings.data_paths(data_dir).sessions)


def _write_jsonrpc_frame(frame: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(frame, ensure_ascii=False, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def _configure_gateway_stdio() -> None:
    """The TUI JSON-RPC transport is always UTF-8, even on Windows GBK consoles."""
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if not callable(reconfigure):
            continue
        try:
            reconfigure(encoding="utf-8", errors="backslashreplace")
        except (AttributeError, OSError, ValueError):
            continue


def main() -> None:
    _configure_gateway_stdio()
    try:
        asyncio.run(_serve_default_gateway())
    except Exception as error:
        sys.stderr.write(f"Rova TUI gateway error: {_safe_error_message(error)}\n")
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
