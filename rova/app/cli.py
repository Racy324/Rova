from __future__ import annotations

import argparse
import asyncio
from collections.abc import Callable, Sequence
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from time import monotonic

from rova.ai.events import TextDelta
from rova.ai.messages import AssistantMessage
from rova.ai.stream import stream_simple

from .context.local import load_local_research_context
from .web.artifact_output import render_research_artifact
from .web.backends import create_web_backends
from .web.settings import WebSettings
from .web.sources import ResearchSourceStore
from .runtime import DEFAULT_MAX_TURNS, MAX_PRODUCT_TURNS, build_rova_runtime
from .settings import AppSettings
from .vision import OpenAICompatibleVisionClient, VisionSettings
from .workspace.terminal import TerminalEnvironment
from .workspace.sandbox import SandboxError


_REPL_EXIT_COMMANDS = frozenset({"exit", "quit", "/q"})


def _configure_console_encoding() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if not callable(reconfigure):
            continue
        try:
            reconfigure(encoding="utf-8", errors="backslashreplace")
        except (AttributeError, OSError, ValueError):
            continue


def _console_print(
    *values: object,
    sep: str = " ",
    end: str = "\n",
    file=None,
    flush: bool = False,
) -> None:
    stream = sys.stdout if file is None else file
    try:
        print(*values, sep=sep, end=end, file=stream, flush=flush)
    except UnicodeEncodeError:
        encoding = getattr(stream, "encoding", None) or "utf-8"
        rendered = sep.join(str(value) for value in values)
        safe_rendered = rendered.encode(encoding, errors="backslashreplace").decode(encoding)
        stream.write(f"{safe_rendered}{end}")
        if flush:
            stream.flush()


def parse_rova_cli_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the local Rova agent")
    parser.add_argument("--workspace", type=Path, help="enable filesystem and shell tools within this workspace")
    parser.add_argument(
        "--environment",
        choices=("local", "sandbox"),
        default=None,
        help="execution environment: local trusted Host workspace or isolated Sandbox",
    )
    parser.add_argument("--sandbox-image", help="Docker image for --environment sandbox")
    parser.add_argument(
        "--terminal-backend",
        choices=("local", "docker"),
        default=None,
        help="terminal execution backend override",
    )
    parser.add_argument(
        "--docker-image",
        help="required Linux image when --terminal-backend docker is selected",
    )
    parser.add_argument("--mcp-config", type=Path, default=None, help="explicit MCP server TOML configuration")
    parser.add_argument("--web", action="store_true", help="enable public-web search and fetch tools")
    parser.add_argument("--tui", action="store_true", help="launch the local Ink terminal interface")
    parser.add_argument(
        "--context-path",
        action="append",
        type=Path,
        default=[],
        dest="context_paths",
        help="explicit UTF-8 local context file; repeat for multiple files",
    )
    parser.add_argument("--save", action="store_true", help="save a completed answer as a runtime artifact")
    parser.add_argument(
        "--data-dir",
        type=Path,
        help="store runtime sessions and artifacts beneath this directory",
    )
    parser.add_argument(
        "--permission",
        choices=("ask", "full"),
        default="ask",
        help="approval handling for policy-required operations (default: ask)",
    )
    parser.add_argument(
        "--max-turns",
        type=_max_turns_argument,
        default=DEFAULT_MAX_TURNS,
        help=f"maximum provider turns per request (1-{MAX_PRODUCT_TURNS}, default: {DEFAULT_MAX_TURNS})",
    )
    parser.add_argument("prompt_parts", nargs="*", help="initial request")
    parsed = parser.parse_args(argv)
    parsed.prompt = " ".join(parsed.prompt_parts)
    if parsed.tui and parsed.save:
        parser.error("--save is not available in TUI mode")
    if parsed.tui and parsed.prompt:
        parser.error("an initial prompt is not available in TUI mode; submit it from the composer")
    return parsed


async def run_rova_cli(
    argv: Sequence[str] | None = None,
    *,
    input_fn: Callable[[str], str] = input,
    args: argparse.Namespace | None = None,
) -> None:
    args = parse_rova_cli_args(argv) if args is None else args
    runtime = _build_runtime_from_args(args)
    start_mcp_discovery = getattr(runtime, "start_mcp_discovery", None)
    if callable(start_mcp_discovery):
        start_mcp_discovery()
    _render_permission_mode(args.permission)
    _render_recovery_report(runtime)
    _render_execution_environment(runtime)
    terminal_backend = getattr(runtime, "terminal_backend", None)
    _subscribe_console_renderer(
        runtime.agent,
        workspace_root=runtime.workspace.root if runtime.workspace is not None else None,
        source_store=runtime.source_store,
        terminal_environment=(
            terminal_backend.environment if terminal_backend is not None else None
        ),
    )
    if args.prompt:
        try:
            responses = await runtime.prompt(args.prompt)
            if args.save:
                reference = _save_final_response(runtime, args.prompt, responses[-1])
                if reference is not None:
                    _console_print(f"[artifact] saved {reference.artifact_id}")
        finally:
            await runtime.close()
        return
    while True:
        try:
            prompt = await asyncio.to_thread(input_fn, "You> ")
        except (EOFError, KeyboardInterrupt):
            await _close_repl_session(runtime)
            return
        normalized_prompt = prompt.strip()
        if normalized_prompt in _REPL_EXIT_COMMANDS:
            await _close_repl_session(runtime)
            return
        if not normalized_prompt:
            continue
        if normalized_prompt.startswith("/sandbox"):
            runtime = await _run_sandbox_command(runtime, normalized_prompt, args, input_fn)
            continue
        try:
            await runtime.prompt(prompt)
        except SandboxError as error:
            _console_print(f"Request blocked: {error}")


def _build_runtime_from_args(
    args: argparse.Namespace,
    *,
    approval_handler=None,
    session_id: str | None = None,
    app_settings: AppSettings | None = None,
):
    local_context = load_local_research_context(args.context_paths)
    app_settings = app_settings or AppSettings.from_env()
    web_search_backend = None
    webpage_fetcher = None
    if args.web:
        web_search_backend, webpage_fetcher = create_web_backends(WebSettings.from_env())
    vision_client = None
    if args.workspace is not None:
        vision_settings = VisionSettings.from_env()
        if vision_settings.is_configured:
            vision_client = OpenAICompatibleVisionClient(vision_settings)
    data_paths = app_settings.data_paths(args.data_dir)
    session_root = data_paths.sessions
    artifact_root = data_paths.artifacts if args.data_dir is not None else app_settings.artifact_root or data_paths.artifacts
    memory_root = data_paths.memory if args.data_dir is not None else getattr(app_settings, "memory_root", None) or data_paths.memory
    skill_root = data_paths.skills
    experience_root = data_paths.experience
    environment_kind, sandbox_image = _resolve_execution_environment(args, app_settings)
    resolved_terminal_backend = "docker" if environment_kind == "sandbox" else "local"
    runtime = build_rova_runtime(
        model=app_settings.to_model(),
        stream_fn=stream_simple,
        workspace_root=args.workspace,
        approval_handler=approval_handler,
        permission_mode=args.permission,
        web_search_backend=web_search_backend,
        webpage_fetcher=webpage_fetcher,
        local_context=local_context,
        compaction_policy=app_settings.to_compaction_policy(),
        session_root=session_root,
        artifact_root=artifact_root,
        max_turns=args.max_turns,
        session_id=session_id,
        memory_root=memory_root,
        skill_root=skill_root,
        memory_model=(
            app_settings.to_memory_model()
            if hasattr(app_settings, "to_memory_model")
            else app_settings.to_model()
        ),
        memory_max_chars=getattr(app_settings, "memory_max_chars", 6_000),
        experience_review_enabled=getattr(app_settings, "experience_review_enabled", True),
        experience_review_tool_threshold=getattr(app_settings, "experience_review_tool_threshold", 10),
        experience_review_task_threshold=getattr(app_settings, "experience_review_task_threshold", 5),
        experience_root=experience_root,
        vision_client=vision_client,
        terminal_backend=resolved_terminal_backend,
        docker_image=sandbox_image,
        isolated_sandbox=environment_kind == "sandbox",
        sandbox_root=data_paths.sandboxes,
        mcp_config_path=(
            getattr(args, "mcp_config", None)
            if getattr(args, "mcp_config", None) is not None
            else getattr(app_settings, "mcp_config_path", None)
        ),
    )
    return runtime


def _resolve_terminal_settings(args: argparse.Namespace, app_settings: AppSettings) -> tuple[str, str | None]:
    backend = args.terminal_backend or getattr(app_settings, "terminal_backend", None) or "local"
    image = args.docker_image or getattr(app_settings, "docker_image", None)
    if backend == "docker":
        if args.workspace is None:
            raise ValueError("terminal backend 'docker' requires --workspace")
        if not image:
            raise ValueError("terminal backend 'docker' requires --docker-image or ROVA_DOCKER_IMAGE")
        return backend, image
    if args.docker_image is not None:
        raise ValueError("--docker-image requires terminal backend 'docker'")
    return backend, None


def _resolve_execution_environment(args: argparse.Namespace, app_settings: AppSettings) -> tuple[str, str | None]:
    """Resolve the public environment contract without exposing direct-bind Docker."""
    legacy_backend = getattr(args, "terminal_backend", None)
    legacy_image = getattr(args, "docker_image", None)
    if legacy_backend == "docker" or legacy_image is not None:
        raise ValueError(
            "The legacy Host-bound Docker terminal is not a public isolation mode; "
            "use --environment sandbox --sandbox-image IMAGE instead."
        )
    configured = getattr(app_settings, "execution_environment", None)
    if configured is None and getattr(app_settings, "terminal_backend", None) == "docker":
        raise ValueError(
            "ROVA_TERMINAL_BACKEND=docker is a legacy Host-bound setting; "
            "use ROVA_EXECUTION_ENVIRONMENT=sandbox and ROVA_SANDBOX_IMAGE instead."
        )
    environment = getattr(args, "environment", None) or configured or "local"
    image = getattr(args, "sandbox_image", None) or getattr(app_settings, "sandbox_image", None)
    if environment == "sandbox":
        if args.workspace is None:
            raise ValueError("--environment sandbox requires --workspace")
        if not image:
            raise ValueError("--environment sandbox requires --sandbox-image or ROVA_SANDBOX_IMAGE")
        return "sandbox", image
    if getattr(args, "sandbox_image", None) is not None:
        raise ValueError("--sandbox-image requires --environment sandbox")
    return "local", None


async def _close_repl_session(runtime) -> None:
    await runtime.close()
    _console_print(f"Session closed: {runtime.session.session_id}")
    _console_print("Goodbye.")


def _save_final_response(runtime, request: str, final_response: AssistantMessage):
    if final_response.stop_reason in {"error", "aborted"} or final_response.tool_calls:
        return None
    source_store = runtime.source_store or ResearchSourceStore()
    content = render_research_artifact(
        request,
        final_response.text,
        source_store,
        local_context=runtime.local_context,
    )
    return runtime.artifact_store.write_text_artifact(
        content,
        artifact_kind="user_requested_output",
        media_type="text/markdown; charset=utf-8",
        run_id=None,
        session_id=runtime.session.session_id,
    )


def _subscribe_console_renderer(
    agent,
    *,
    workspace_root: Path | None = None,
    source_store: ResearchSourceStore | None = None,
    terminal_environment: TerminalEnvironment | None = None,
) -> None:
    streamed_text = False
    tool_started_at: dict[str, float] = {}

    def render(event) -> None:
        nonlocal streamed_text
        if event.type == "tool_execution_start":
            if event.tool_call_id is not None:
                tool_started_at[event.tool_call_id] = monotonic()
            _console_print(f"[tool:start] {event.tool_name}")
            for line in _tool_display_lines(
                event.tool_name,
                event.args or {},
                workspace_root,
                terminal_environment,
            ):
                _console_print(line)
        elif event.type == "tool_execution_end":
            started_at = tool_started_at.pop(event.tool_call_id, None)
            _console_print(f"[tool:end] {event.tool_name}")
            _console_print(f"status: {'error' if event.is_error else 'success'}")
            if started_at is not None:
                _console_print(f"duration: {monotonic() - started_at:.1f}s")
        elif event.type == "message_start":
            streamed_text = False
        elif event.type == "message_update" and isinstance(event.assistant_message_event, TextDelta):
            streamed_text = True
            _console_print(event.assistant_message_event.delta, end="", flush=True)
        elif event.type == "message_end" and event.message and not event.message.tool_calls:
            if not streamed_text and event.message.text:
                _console_print(event.message.text)
            elif streamed_text:
                _console_print()
            _render_cited_sources(event.message.text, source_store)

    agent.subscribe(render)


def _tool_display_lines(
    tool_name: str | None,
    arguments: dict,
    workspace_root: Path | None,
    terminal_environment: TerminalEnvironment | None = None,
) -> list[str]:
    if "command" in arguments:
        lines = [f"Command:\n{arguments['command']}"]
        if tool_name == "shell" and workspace_root is not None:
            if terminal_environment is not None:
                lines.extend([f"backend:\n{terminal_environment.kind}", f"cwd:\n{terminal_environment.cwd}"])
            else:
                lines.append(f"cwd:\n{workspace_root}")
        return lines
    if "path" in arguments:
        return [f"File:\n{arguments['path']}"]
    if "query" in arguments:
        return [f"Query:\n{arguments['query']}"]
    if "source_id" in arguments:
        return [f"Source:\n{arguments['source_id']}"]
    return []


def _render_cited_sources(answer: str, source_store: ResearchSourceStore | None) -> None:
    source_ids = tuple(dict.fromkeys(re.findall(r"\[(S\d+)\]", answer)))
    if not source_ids:
        return
    _console_print()
    _console_print("Sources:")
    for index, source_id in enumerate(source_ids):
        if index:
            _console_print()
        _console_print(f"[{source_id}]")
        try:
            source = source_store.get(source_id) if source_store is not None else None
        except KeyError:
            source = None
        if source is None:
            _console_print("Source unavailable")
            continue
        status = "fetched" if source.content is not None else "search_only"
        _console_print(f"Title:\n{source.title}")
        _console_print(f"URL:\n{source.url}")
        _console_print(f"Status:\n{status}")


def _render_permission_mode(permission_mode: str) -> None:
    _console_print(f"Permission mode: {permission_mode}")
    if permission_mode == "full":
        _console_print("Warning: Full mode automatically approves tool operations that require approval.")
        _console_print("Warning: Shell commands are not sandboxed.")


def _render_execution_environment(runtime) -> None:
    control = getattr(runtime, "sandbox_control", None)
    if control is None:
        if runtime.workspace is not None:
            _console_print("Environment: Local (trusted Host). File and shell tools modify the Host Workspace directly.")
        return
    status = control.status()
    _console_print(
        f"Environment: Sandbox (isolated) | state: {status.sandbox_state.value} | id: {status.sandbox_id}. "
        "Host Workspace remains unchanged until explicit Apply."
    )
    if status.container_recreated_on_resume:
        _console_print("Sandbox files were preserved; the execution container was recreated, so container-local state may need recreation.")


async def _run_sandbox_command(runtime, command: str, args: argparse.Namespace, input_fn: Callable[[str], str]):
    """REPL control-plane commands; deliberately separate from Agent tools."""
    control = getattr(runtime, "sandbox_control", None)
    action = command.removeprefix("/sandbox").strip().lower() or "status"
    if control is None:
        _console_print("Sandbox controls are unavailable in Local mode. Local mode modifies the Host Workspace directly.")
        return runtime
    if action == "status":
        status = control.status()
        _console_print(
            f"Environment: Sandbox (isolated) | state: {status.sandbox_state.value} | "
            f"id: {status.sandbox_id} | changed paths: {status.changed_path_count if status.changed_path_count is not None else 'unavailable'}"
        )
        return runtime
    if action == "diff":
        changed = control.diff()
        _console_print(f"Sandbox diff: {len(changed.changes)} changed path(s)")
        for item in changed.changes:
            _console_print(f"- {item.kind.value}: {item.path}")
        return runtime
    if action == "apply":
        plan = control.apply_plan()
        if plan.conflicts:
            _console_print("Apply blocked: Host Workspace changed since the Sandbox baseline.")
            for conflict in plan.conflicts:
                _console_print(f"- {conflict.path}: {conflict.reason}")
            return runtime
        if input_fn(f"Apply {len(plan.changed_set.changes)} Sandbox change(s) to the Host Workspace? [y/N] ").strip().lower() != "y":
            _console_print("Apply cancelled. Host Workspace was not modified.")
            return runtime
        report = control.apply(confirm=lambda _plan: True)
        _console_print("Host changes applied successfully." if report.applied else f"Apply did not complete: {report.error or report.state}")
        return runtime
    if action == "discard":
        changed = control.diff()
        if input_fn(f"Discard {len(changed.changes)} Sandbox change(s)? Host Workspace remains unchanged. [y/N] ").strip().lower() != "y":
            _console_print("Discard cancelled.")
            return runtime
        report = control.discard(confirm=lambda _plan: True)
        _console_print("Sandbox changes were discarded. Host Workspace was not modified." if report.discarded else f"Discard did not complete: {report.error or report.state.value}")
        return runtime
    if action == "restore":
        operation_id = control.metadata().active_apply_id
        if not operation_id:
            _console_print("No unfinished Apply requires preimage recovery.")
            return runtime
        if input_fn("Restore Host preimages touched by the unfinished Apply? [y/N] ").strip().lower() != "y":
            _console_print("Recovery cancelled.")
            return runtime
        report = control.restore_preimages(operation_id, confirm=lambda _report: True)
        _console_print(f"Apply recovery state: {report.state.value if report.state else 'unknown'}")
        return runtime
    if action == "new":
        if input_fn("Create a new Sandbox baseline from the current Host Workspace? [y/N] ").strip().lower() != "y":
            _console_print("New Sandbox creation cancelled.")
            return runtime
        metadata = control.create_new()
        session_id = runtime.session.session_id
        await runtime.close()
        replacement = _build_runtime_from_args(args, session_id=session_id)
        _console_print(f"Sandbox created ({metadata.sandbox_id[:8]}). Host Workspace remains unchanged until explicit Apply.")
        return replacement
    _console_print("Sandbox commands: /sandbox status | diff | apply | discard | restore | new")
    return runtime


def _render_recovery_report(runtime) -> None:
    report = getattr(runtime, "recovery_report", getattr(runtime.session, "recovery_report", None))
    if report is None or report.recovered_count == 0:
        return
    unknown_count = sum(item.side_effects_unknown for item in report.items)
    message = f"Session recovery: committed {report.recovered_count} pending tool result(s)."
    if unknown_count:
        message += f" {unknown_count} may have unknown side effects; inspect the current state before continuing."
    _console_print(message)


def _max_turns_argument(value: str) -> int:
    try:
        max_turns = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("max turns must be an integer") from error
    if not 1 <= max_turns <= MAX_PRODUCT_TURNS:
        raise argparse.ArgumentTypeError(f"max turns must be between 1 and {MAX_PRODUCT_TURNS}")
    return max_turns


def _tui_entry_path() -> Path:
    return Path(__file__).resolve().parents[2] / "ui-tui" / "dist" / "entry.js"


def _launch_tui(args: argparse.Namespace) -> None:
    _configure_tui_console_encoding()
    entry = _tui_entry_path()
    if not entry.is_file():
        raise SystemExit("TUI is not built. Run: cd ui-tui; pnpm install; pnpm build")
    node = shutil.which("node")
    if node is None:
        raise SystemExit("Node.js was not found on PATH; install Node.js, then run: cd ui-tui; pnpm install; pnpm build")
    environment = os.environ.copy()
    environment["ROVA_TUI_PYTHON"] = sys.executable
    environment["ROVA_TUI_ARGS_JSON"] = json.dumps(_tui_gateway_argv(args), ensure_ascii=False)
    process = subprocess.Popen([node, str(entry)], env=environment, shell=False)
    try:
        return_code = process.wait()
    except KeyboardInterrupt:
        # Ink receives this idle Ctrl+C too and closes the Gateway through its
        # existing normal exit path. Keep the parent alive until that cleanup
        # finishes instead of leaking a Python traceback to the user.
        return_code = process.wait()
    if return_code:
        raise SystemExit(return_code)


def _configure_tui_console_encoding() -> None:
    """Prepare the attached Windows console for Ink's UTF-8 ANSI rendering."""
    if os.name != "nt":
        return
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleOutputCP(65001)
        kernel32.SetConsoleCP(65001)

        stdout_handle = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
        console_mode = ctypes.c_uint()
        if stdout_handle not in (0, -1) and kernel32.GetConsoleMode(stdout_handle, ctypes.byref(console_mode)):
            enable_virtual_terminal_processing = 0x0004
            kernel32.SetConsoleMode(
                stdout_handle,
                console_mode.value | enable_virtual_terminal_processing,
            )
    except (AttributeError, OSError):
        return


def _supports_tui_terminal() -> bool:
    """Ink needs both streams attached to a real ANSI-capable terminal."""
    for stream in (sys.stdin, sys.stdout):
        isatty = getattr(stream, "isatty", None)
        try:
            if not callable(isatty) or not isatty():
                return False
        except (OSError, ValueError):
            return False
    return True


def _tui_gateway_argv(args: argparse.Namespace) -> list[str]:
    argv: list[str] = []
    if args.workspace is not None:
        argv.extend(["--workspace", str(args.workspace)])
    if getattr(args, "environment", None) is not None:
        argv.extend(["--environment", args.environment])
    if getattr(args, "sandbox_image", None) is not None:
        argv.extend(["--sandbox-image", args.sandbox_image])
    if args.terminal_backend is not None:
        argv.extend(["--terminal-backend", args.terminal_backend])
    if args.docker_image is not None:
        argv.extend(["--docker-image", args.docker_image])
    if getattr(args, "mcp_config", None) is not None:
        argv.extend(["--mcp-config", str(args.mcp_config)])
    if args.web:
        argv.append("--web")
    for context_path in args.context_paths:
        argv.extend(["--context-path", str(context_path)])
    if args.data_dir is not None:
        argv.extend(["--data-dir", str(args.data_dir)])
    argv.extend(["--permission", args.permission, "--max-turns", str(args.max_turns)])
    return argv


def main() -> None:
    _configure_console_encoding()
    args = parse_rova_cli_args()
    if args.tui:
        if _supports_tui_terminal():
            _launch_tui(args)
        else:
            _console_print("TUI requires an interactive terminal; using the scrolling CLI transcript instead.")
            asyncio.run(run_rova_cli(args=args))
        return
    asyncio.run(run_rova_cli(args=args))
