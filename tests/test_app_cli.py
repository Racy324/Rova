from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from rova.ai.events import StreamDone, TextDelta
from rova.ai.messages import AssistantMessage, TextBlock
from rova.ai.models import Model
from rova.agent_core.events import AgentEvent
from rova.agent_session.agent_session import RecoveryItem, RecoveryReport
from rova.app import cli
from rova.app.cli import _resolve_terminal_settings, _subscribe_console_renderer, _tool_display_lines, parse_rova_cli_args, run_rova_cli
from rova.app.paths import RovaDataPaths
from rova.app.context.local import LocalContextItem, LocalResearchContext
from rova.app.web.sources import ResearchSourceStore, SearchHit
from rova.artifacts import FileArtifactStore
from rova.app.workspace.terminal import TerminalEnvironment
from rova.app.skill_proposals import FileSkillProposalStore, SkillProposal
from rova.app.skill_candidates import FileSkillCandidateStore


@pytest.fixture(autouse=True)
def _isolate_default_rova_data_dir(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("ROVA_DATA_DIR", str(tmp_path / "rova-data"))


class _LegacyGbkStream:
    encoding = "gbk"

    def __init__(self) -> None:
        self.output: list[str] = []

    def write(self, text: str) -> int:
        text.encode(self.encoding)
        self.output.append(text)
        return len(text)

    def flush(self) -> None:
        return None


class _FakeAppSettings:
    data_dir: Path | None = None
    artifact_root: Path | None = None
    memory_root: Path | None = None

    def data_paths(self, explicit_data_dir: Path | None = None) -> RovaDataPaths:
        return RovaDataPaths.resolve(explicit_data_dir or self.data_dir)


def test_unified_cli_parses_explicit_tools_context_and_prompt() -> None:
    args = parse_rova_cli_args([
        "--workspace", "project",
        "--web",
        "--context-path", "notes/one.md",
        "--context-path", "notes/two.md",
        "--save",
        "--data-dir", "runtime-data",
        "--max-turns", "16",
        "compare", "the", "sources",
    ])

    assert args.workspace == Path("project")
    assert args.web is True
    assert args.context_paths == [Path("notes/one.md"), Path("notes/two.md")]
    assert args.save is True
    assert args.data_dir == Path("runtime-data")
    assert args.max_turns == 16
    assert args.prompt == "compare the sources"


def test_unified_cli_defaults_permission_to_ask() -> None:
    assert parse_rova_cli_args([]).permission == "ask"
    assert parse_rova_cli_args([]).data_dir is None


@pytest.mark.asyncio
async def test_skills_proposals_lists_pending_proposals_without_building_runtime(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    data_dir = tmp_path / "runtime-data"
    proposal = FileSkillProposalStore(data_dir / "skill-proposals").save(
        3,
        [
            SkillProposal(
                action="create",
                name="paper-review",
                content="---\nname: paper-review\ndescription: Review papers.\n---\n",
                rationale="Capture a reusable workflow.",
            )
        ],
    )[0]

    def runtime_must_not_be_created(*_args, **_kwargs):
        raise AssertionError("skills management must not build a Runtime")

    monkeypatch.setattr(cli, "_build_runtime_from_args", runtime_must_not_be_created)

    await run_rova_cli(["--data-dir", str(data_dir), "skills", "proposals"])

    rendered = capsys.readouterr().out
    assert proposal.proposal_id in rendered
    assert "create paper-review" in rendered


@pytest.mark.asyncio
async def test_skills_candidate_create_materializes_pending_proposal_without_runtime(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    data_dir = tmp_path / "runtime-data"
    proposal = FileSkillProposalStore(data_dir / "skill-proposals").save(
        3,
        [
            SkillProposal(
                action="create",
                name="paper-review",
                content="---\nname: paper-review\ndescription: Review papers.\n---\n",
                rationale="Capture a reusable workflow.",
            )
        ],
    )[0]
    monkeypatch.setattr(
        cli,
        "_build_runtime_from_args",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("must not build Runtime")),
    )

    await run_rova_cli([
        "--data-dir", str(data_dir), "skills", "candidate", "create", proposal.proposal_id,
    ])

    candidates = FileSkillCandidateStore(data_dir / "skill-candidates").list()
    assert len(candidates) == 1
    assert candidates[0].proposal_id == proposal.proposal_id
    assert candidates[0].candidate_id in capsys.readouterr().out


@pytest.mark.asyncio
async def test_skills_candidate_list_and_show_use_candidate_store_and_review_service(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    data_dir = tmp_path / "runtime-data"
    proposal = FileSkillProposalStore(data_dir / "skill-proposals").save(
        3,
        [
            SkillProposal(
                action="create",
                name="paper-review",
                content="---\nname: paper-review\ndescription: Review papers.\n---\n# paper-review\n",
                rationale="Capture a reusable workflow.",
            )
        ],
    )[0]
    active_store = cli.FileSkillStore(data_dir / "skills")
    candidate_store = FileSkillCandidateStore(data_dir / "skill-candidates")
    candidate = cli.CandidateMaterializer(
        proposal_store=FileSkillProposalStore(data_dir / "skill-proposals"),
        active_skill_store=active_store,
        candidate_store=candidate_store,
    ).materialize(proposal.proposal_id)
    monkeypatch.setattr(
        cli,
        "_build_runtime_from_args",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("must not build Runtime")),
    )

    await run_rova_cli(["--data-dir", str(data_dir), "skills", "candidate", "list"])
    listed = capsys.readouterr().out
    await run_rova_cli([
        "--data-dir", str(data_dir), "skills", "candidate", "show", candidate.candidate_id,
    ])
    shown = capsys.readouterr().out

    assert candidate.candidate_id in listed
    assert "ready create paper-review" in listed
    assert candidate.candidate_id in shown
    assert "Capture a reusable workflow." in shown
    assert "target_absent" in shown
    assert "Validation errors: none" in shown
    assert "# paper-review" in shown


@pytest.mark.asyncio
async def test_skills_candidate_promote_requires_yes_when_stdin_is_noninteractive(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    data_dir = tmp_path / "runtime-data"
    proposal = FileSkillProposalStore(data_dir / "skill-proposals").save(
        1,
        [
            SkillProposal(
                action="create",
                name="paper-review",
                content="---\nname: paper-review\ndescription: Review papers.\n---\n",
                rationale="Capture a reusable workflow.",
            )
        ],
    )[0]
    active_store = cli.FileSkillStore(data_dir / "skills")
    candidate_store = FileSkillCandidateStore(data_dir / "skill-candidates")
    candidate = cli.CandidateMaterializer(
        proposal_store=FileSkillProposalStore(data_dir / "skill-proposals"),
        active_skill_store=active_store,
        candidate_store=candidate_store,
    ).materialize(proposal.proposal_id)

    class NonInteractiveInput:
        def isatty(self) -> bool:
            return False

    monkeypatch.setattr(cli.sys, "stdin", NonInteractiveInput())
    monkeypatch.setattr(
        cli,
        "_build_runtime_from_args",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("must not build Runtime")),
    )

    await run_rova_cli([
        "--data-dir", str(data_dir), "skills", "candidate", "promote", candidate.candidate_id,
    ])

    assert not (active_store.root / "paper-review").exists()
    assert candidate_store.read(candidate.candidate_id).state.value == "ready"
    assert "requires --yes in non-interactive mode" in capsys.readouterr().out


@pytest.mark.asyncio
async def test_skills_candidate_reject_forwards_reason_without_building_runtime(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    data_dir = tmp_path / "runtime-data"
    proposal = FileSkillProposalStore(data_dir / "skill-proposals").save(
        1,
        [
            SkillProposal(
                action="create",
                name="paper-review",
                content="---\nname: paper-review\ndescription: Review papers.\n---\n",
                rationale="Capture a reusable workflow.",
            )
        ],
    )[0]
    active_store = cli.FileSkillStore(data_dir / "skills")
    candidate_store = FileSkillCandidateStore(data_dir / "skill-candidates")
    candidate = cli.CandidateMaterializer(
        proposal_store=FileSkillProposalStore(data_dir / "skill-proposals"),
        active_skill_store=active_store,
        candidate_store=candidate_store,
    ).materialize(proposal.proposal_id)
    monkeypatch.setattr(
        cli,
        "_build_runtime_from_args",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("must not build Runtime")),
    )

    await run_rova_cli([
        "--data-dir", str(data_dir), "skills", "candidate", "reject", candidate.candidate_id,
        "--reason", "Needs a narrower scope.",
    ])

    assert candidate_store.read(candidate.candidate_id).state.value == "rejected"
    assert candidate_store.read(candidate.candidate_id).rejection_reason == "Needs a narrower scope."
    assert not (active_store.root / "paper-review").exists()
    assert "Candidate rejected" in capsys.readouterr().out


@pytest.mark.asyncio
@pytest.mark.parametrize(("response", "promoted"), [("y", True), ("Y", False)])
async def test_skills_candidate_promote_interactive_accepts_only_exact_y(
    monkeypatch, tmp_path: Path, capsys, response: str, promoted: bool
) -> None:
    data_dir = tmp_path / "runtime-data"
    proposal = FileSkillProposalStore(data_dir / "skill-proposals").save(
        1,
        [
            SkillProposal(
                action="create",
                name="paper-review",
                content="---\nname: paper-review\ndescription: Review papers.\n---\n",
                rationale="Capture a reusable workflow.",
            )
        ],
    )[0]
    active_store = cli.FileSkillStore(data_dir / "skills")
    candidate_store = FileSkillCandidateStore(data_dir / "skill-candidates")
    candidate = cli.CandidateMaterializer(
        proposal_store=FileSkillProposalStore(data_dir / "skill-proposals"),
        active_skill_store=active_store,
        candidate_store=candidate_store,
    ).materialize(proposal.proposal_id)

    class InteractiveInput:
        def isatty(self) -> bool:
            return True

    monkeypatch.setattr(cli.sys, "stdin", InteractiveInput())
    monkeypatch.setattr(
        cli,
        "_build_runtime_from_args",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("must not build Runtime")),
    )

    await run_rova_cli(
        ["--data-dir", str(data_dir), "skills", "candidate", "promote", candidate.candidate_id],
        input_fn=lambda _prompt: response,
    )

    assert (active_store.root / "paper-review").exists() is promoted
    assert candidate_store.read(candidate.candidate_id).state.value == (
        "promoted" if promoted else "ready"
    )
    output = capsys.readouterr().out
    assert "Target status: target_absent" in output
    assert ("Candidate promoted" if promoted else "Promotion cancelled") in output


@pytest.mark.asyncio
async def test_skills_candidate_promote_yes_confirms_without_interactive_input(
    monkeypatch, tmp_path: Path
) -> None:
    data_dir = tmp_path / "runtime-data"
    proposal = FileSkillProposalStore(data_dir / "skill-proposals").save(
        1,
        [
            SkillProposal(
                action="create",
                name="paper-review",
                content="---\nname: paper-review\ndescription: Review papers.\n---\n",
                rationale="Capture a reusable workflow.",
            )
        ],
    )[0]
    active_store = cli.FileSkillStore(data_dir / "skills")
    candidate_store = FileSkillCandidateStore(data_dir / "skill-candidates")
    candidate = cli.CandidateMaterializer(
        proposal_store=FileSkillProposalStore(data_dir / "skill-proposals"),
        active_skill_store=active_store,
        candidate_store=candidate_store,
    ).materialize(proposal.proposal_id)
    monkeypatch.setattr(
        cli,
        "_build_runtime_from_args",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("must not build Runtime")),
    )

    await run_rova_cli([
        "--data-dir", str(data_dir), "skills", "candidate", "promote", candidate.candidate_id, "--yes",
    ])

    assert active_store.read_main_document("paper-review").startswith("---\nname: paper-review")
    assert candidate_store.read(candidate.candidate_id).state.value == "promoted"


@pytest.mark.asyncio
@pytest.mark.parametrize("reason_args", [(), ("--reason", "")])
async def test_skills_candidate_reject_requires_nonempty_reason(
    monkeypatch, tmp_path: Path, capsys, reason_args: tuple[str, ...]
) -> None:
    data_dir = tmp_path / "runtime-data"
    proposal = FileSkillProposalStore(data_dir / "skill-proposals").save(
        1,
        [
            SkillProposal(
                action="create",
                name="paper-review",
                content="---\nname: paper-review\ndescription: Review papers.\n---\n",
                rationale="Capture a reusable workflow.",
            )
        ],
    )[0]
    active_store = cli.FileSkillStore(data_dir / "skills")
    candidate_store = FileSkillCandidateStore(data_dir / "skill-candidates")
    candidate = cli.CandidateMaterializer(
        proposal_store=FileSkillProposalStore(data_dir / "skill-proposals"),
        active_skill_store=active_store,
        candidate_store=candidate_store,
    ).materialize(proposal.proposal_id)
    monkeypatch.setattr(
        cli,
        "_build_runtime_from_args",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("must not build Runtime")),
    )

    await run_rova_cli([
        "--data-dir", str(data_dir), "skills", "candidate", "reject", candidate.candidate_id, *reason_args,
    ])

    assert candidate_store.read(candidate.candidate_id).state.value == "ready"
    assert not (active_store.root / "paper-review").exists()
    assert "requires --reason" in capsys.readouterr().out


def test_cli_renders_only_recovery_counts_and_side_effect_warning(capsys) -> None:
    runtime = SimpleNamespace(
        session=SimpleNamespace(
            recovery_report=RecoveryReport((
                RecoveryItem("shell", 0, "execution_interrupted", side_effects_unknown=True),
                RecoveryItem("read_file", 1, "execution_not_started", side_effects_unknown=False),
            )),
        ),
    )

    cli._render_recovery_report(runtime)

    rendered = capsys.readouterr().out
    assert "Session recovery: committed 2 pending tool result(s)." in rendered
    assert "1 may have unknown side effects" in rendered
    assert "shell" not in rendered
    assert "read_file" not in rendered


def test_unified_cli_accepts_mcp_config_and_forwards_it_to_tui() -> None:
    args = parse_rova_cli_args(["--mcp-config", "configs/mcp.toml", "--tui"])

    assert args.mcp_config == Path("configs/mcp.toml")
    assert "--mcp-config" in cli._tui_gateway_argv(args)


def test_unified_cli_accepts_explicit_docker_terminal_backend_and_forwards_it_to_tui() -> None:
    args = parse_rova_cli_args([
        "--tui",
        "--workspace", "project",
        "--terminal-backend", "docker",
        "--docker-image", "rova-test:latest",
    ])

    assert args.terminal_backend == "docker"
    assert args.docker_image == "rova-test:latest"
    assert cli._tui_gateway_argv(args) == [
        "--workspace", "project",
        "--terminal-backend", "docker",
        "--docker-image", "rova-test:latest",
        "--permission", "ask", "--max-turns", "16",
    ]


def test_terminal_settings_resolve_cli_then_persistent_config_then_default() -> None:
    settings = SimpleNamespace(terminal_backend="docker", docker_image="config:image")
    assert _resolve_terminal_settings(parse_rova_cli_args(["--workspace", "project"]), settings) == ("docker", "config:image")
    assert _resolve_terminal_settings(parse_rova_cli_args(["--workspace", "project", "--terminal-backend", "local"]), settings) == ("local", None)
    assert _resolve_terminal_settings(parse_rova_cli_args(["--workspace", "project", "--docker-image", "cli:image"]), settings) == ("docker", "cli:image")
    assert _resolve_terminal_settings(parse_rova_cli_args(["--workspace", "project"]), SimpleNamespace(terminal_backend=None, docker_image=None)) == ("local", None)


@pytest.mark.parametrize("argv", [
    ["--terminal-backend", "docker", "--docker-image", "rova-test:latest"],
    ["--workspace", "project", "--terminal-backend", "docker"],
    ["--workspace", "project", "--docker-image", "rova-test:latest"],
])
def test_unified_cli_rejects_invalid_terminal_backend_combinations(argv: list[str]) -> None:
    with pytest.raises(ValueError):
        _resolve_terminal_settings(
            parse_rova_cli_args(argv),
            SimpleNamespace(terminal_backend=None, docker_image=None),
        )


def test_cli_shell_display_uses_the_actual_terminal_backend_cwd(tmp_path: Path) -> None:
    lines = _tool_display_lines(
        "shell",
        {"command": "python task.py"},
        tmp_path,
        TerminalEnvironment("docker", "docker (/bin/sh)", "/workspace", True),
    )

    assert lines == ["Command:\npython task.py", "backend:\ndocker", "cwd:\n/workspace"]


def test_unified_cli_accepts_tui_without_changing_standard_options() -> None:
    args = parse_rova_cli_args(["--tui", "--workspace", "project", "--web"])

    assert args.tui is True
    assert args.workspace == Path("project")
    assert args.web is True


def test_public_environment_settings_prefer_explicit_sandbox_over_persistent_default() -> None:
    settings = SimpleNamespace(execution_environment="local", sandbox_image="config:image")
    args = parse_rova_cli_args(["--workspace", "project", "--environment", "sandbox", "--sandbox-image", "cli:image"])

    assert cli._resolve_execution_environment(args, settings) == ("sandbox", "cli:image")


def test_legacy_host_bound_docker_cli_is_rejected_with_a_sandbox_migration_message() -> None:
    args = parse_rova_cli_args(["--workspace", "project", "--terminal-backend", "docker", "--docker-image", "old:image"])

    with pytest.raises(ValueError, match="--environment sandbox"):
        cli._resolve_execution_environment(args, SimpleNamespace(execution_environment=None, sandbox_image=None))


def test_tui_terminal_support_requires_real_input_and_output_ttys(monkeypatch) -> None:
    class Stream:
        def __init__(self, is_tty: bool) -> None:
            self._is_tty = is_tty

        def isatty(self) -> bool:
            return self._is_tty

    monkeypatch.setattr(cli.sys, "stdin", Stream(True))
    monkeypatch.setattr(cli.sys, "stdout", Stream(False))

    assert cli._supports_tui_terminal() is False


def test_cli_main_falls_back_to_scrolling_repl_without_a_real_tty(monkeypatch, capsys) -> None:
    args = parse_rova_cli_args(["--tui"])
    seen: list[object] = []

    async def run_repl(*, args):
        seen.append(args)

    monkeypatch.setattr(cli, "_configure_console_encoding", lambda: None)
    monkeypatch.setattr(cli, "parse_rova_cli_args", lambda: args)
    monkeypatch.setattr(cli, "_supports_tui_terminal", lambda: False)
    monkeypatch.setattr(cli, "run_rova_cli", run_repl)

    cli.main()

    assert seen == [args]
    assert "using the scrolling CLI transcript" in capsys.readouterr().out


def test_tui_launcher_uses_node_with_an_absolute_python_and_json_argument_list(monkeypatch, tmp_path: Path) -> None:
    entry = tmp_path / "ui-tui" / "dist" / "entry.js"
    entry.parent.mkdir(parents=True)
    entry.write_text("// built", encoding="utf-8")
    captured = {}
    args = parse_rova_cli_args(["--tui", "--workspace", "project", "--max-turns", "16"])

    monkeypatch.setattr(cli, "_tui_entry_path", lambda: entry)
    monkeypatch.setattr(cli.shutil, "which", lambda name: "C:/Program Files/nodejs/node.exe" if name == "node" else None)
    monkeypatch.setattr(
        cli.subprocess,
        "Popen",
        lambda command, **kwargs: captured.update(command=command, **kwargs) or SimpleNamespace(wait=lambda: 0),
    )

    cli._launch_tui(args)

    assert captured["command"] == ["C:/Program Files/nodejs/node.exe", str(entry)]
    assert captured["shell"] is False
    assert captured["env"]["ROVA_TUI_PYTHON"] == sys.executable
    assert json.loads(captured["env"]["ROVA_TUI_ARGS_JSON"]) == [
        "--workspace", "project", "--permission", "ask", "--max-turns", "16",
    ]


def test_tui_launcher_configures_the_attached_console_before_starting_ink(monkeypatch, tmp_path: Path) -> None:
    entry = tmp_path / "ui-tui" / "dist" / "entry.js"
    entry.parent.mkdir(parents=True)
    entry.write_text("// built", encoding="utf-8")
    calls: list[str] = []
    args = parse_rova_cli_args(["--tui"])

    monkeypatch.setattr(cli, "_tui_entry_path", lambda: entry)
    monkeypatch.setattr(cli, "_configure_tui_console_encoding", lambda: calls.append("configured"), raising=False)
    monkeypatch.setattr(cli.shutil, "which", lambda _name: "node.exe")
    monkeypatch.setattr(cli.subprocess, "run", lambda *_args, **_kwargs: SimpleNamespace(returncode=0))

    cli._launch_tui(args)

    assert calls == ["configured"]


def test_tui_console_configuration_enables_utf8_and_virtual_terminal_output(monkeypatch) -> None:
    import ctypes

    calls: list[tuple[str, int, int | None]] = []

    class Kernel32:
        def SetConsoleOutputCP(self, code_page: int) -> int:
            calls.append(("output_cp", code_page, None))
            return 1

        def SetConsoleCP(self, code_page: int) -> int:
            calls.append(("input_cp", code_page, None))
            return 1

        def GetStdHandle(self, handle: int) -> int:
            calls.append(("stdout_handle", handle, None))
            return 42

        def GetConsoleMode(self, handle: int, mode) -> int:
            calls.append(("get_mode", handle, None))
            mode._obj.value = 0x0001
            return 1

        def SetConsoleMode(self, handle: int, mode: int) -> int:
            calls.append(("set_mode", handle, mode))
            return 1

    fake_ctypes = SimpleNamespace(
        c_uint=ctypes.c_uint,
        byref=ctypes.byref,
        windll=SimpleNamespace(kernel32=Kernel32()),
    )
    monkeypatch.setattr(cli.os, "name", "nt")
    monkeypatch.setitem(sys.modules, "ctypes", fake_ctypes)

    cli._configure_tui_console_encoding()

    assert ("output_cp", 65001, None) in calls
    assert ("input_cp", 65001, None) in calls
    assert ("set_mode", 42, 0x0005) in calls


def test_tui_launcher_waits_for_ink_to_close_after_one_idle_keyboard_interrupt(monkeypatch, tmp_path: Path) -> None:
    entry = tmp_path / "ui-tui" / "dist" / "entry.js"
    entry.parent.mkdir(parents=True)
    entry.write_text("// built", encoding="utf-8")
    args = parse_rova_cli_args(["--tui"])

    class Process:
        def __init__(self) -> None:
            self.wait_calls = 0

        def wait(self) -> int:
            self.wait_calls += 1
            if self.wait_calls == 1:
                raise KeyboardInterrupt
            return 0

    process = Process()
    monkeypatch.setattr(cli, "_tui_entry_path", lambda: entry)
    monkeypatch.setattr(cli, "_configure_tui_console_encoding", lambda: None, raising=False)
    monkeypatch.setattr(cli.shutil, "which", lambda _name: "node.exe")
    monkeypatch.setattr(cli.subprocess, "Popen", lambda *_args, **_kwargs: process, raising=False)
    monkeypatch.setattr(cli.subprocess, "run", lambda *_args, **_kwargs: SimpleNamespace(returncode=0))

    cli._launch_tui(args)

    assert process.wait_calls == 2


def test_cli_configures_standard_streams_for_utf8_at_startup(monkeypatch) -> None:
    configured: list[tuple[str, str]] = []

    class Stream:
        def __init__(self, name: str) -> None:
            self.name = name

        def reconfigure(self, *, encoding: str, errors: str) -> None:
            configured.append((self.name, f"{encoding}/{errors}"))

    monkeypatch.setattr(sys, "stdout", Stream("stdout"))
    monkeypatch.setattr(sys, "stderr", Stream("stderr"))

    cli._configure_console_encoding()

    assert configured == [("stdout", "utf-8/backslashreplace"), ("stderr", "utf-8/backslashreplace")]


def test_cli_safely_degrades_unicode_for_a_legacy_console_stream() -> None:
    stream = _LegacyGbkStream()

    cli._console_print("completed ✅", file=stream)

    assert "completed" in "".join(stream.output)
    assert "\\u2705" in "".join(stream.output)


def test_cli_console_print_preserves_ascii_output() -> None:
    stream = _LegacyGbkStream()

    cli._console_print("completed", file=stream)

    assert "".join(stream.output) == "completed\n"


def test_cli_streaming_renderer_does_not_raise_for_unicode_on_a_legacy_console(monkeypatch) -> None:
    listeners = []
    stream = _LegacyGbkStream()

    class FakeAgent:
        def subscribe(self, listener):
            listeners.append(listener)
            return lambda: None

    monkeypatch.setattr(sys, "stdout", stream)
    _subscribe_console_renderer(FakeAgent())

    listeners[0](AgentEvent("message_start"))
    partial = AssistantMessage([TextBlock("hello world 🚀")])
    listeners[0](AgentEvent("message_update", assistant_message_event=TextDelta("hello", partial)))
    listeners[0](AgentEvent("message_update", assistant_message_event=TextDelta(" world 🚀", partial)))
    listeners[0](AgentEvent("message_end", message=AssistantMessage([TextBlock("hello world 🚀")])))

    assert "".join(stream.output) == "hello world \\U0001f680\n"


@pytest.mark.asyncio
async def test_console_encoding_configuration_failure_does_not_block_repl_close(monkeypatch, tmp_path: Path) -> None:
    runtime = _runtime(tmp_path, response="unused")

    class FailingStream:
        def reconfigure(self, **_kwargs) -> None:
            raise OSError("console configuration unavailable")

        def write(self, _text: str) -> int:
            return 0

        def flush(self) -> None:
            return None

    class Settings(_FakeAppSettings):
        artifact_root = tmp_path / "artifacts"

        def to_model(self):
            return Model(provider="mock")

        def to_compaction_policy(self):
            return None

    monkeypatch.setattr(sys, "stdout", FailingStream())
    monkeypatch.setattr(sys, "stderr", FailingStream())
    monkeypatch.setattr(cli, "load_local_research_context", lambda paths: LocalResearchContext(()), raising=False)
    monkeypatch.setattr(cli, "AppSettings", type("AppSettings", (), {"from_env": classmethod(lambda cls: Settings())}), raising=False)
    monkeypatch.setattr(cli, "build_rova_runtime", lambda **kwargs: runtime, raising=False)

    cli._configure_console_encoding()
    await run_rova_cli([], input_fn=_repl_inputs(iter(["/q"])))

    assert runtime.session.close_calls == 1
    assert runtime.close_calls == 1


@pytest.mark.parametrize("mode", ["ask", "full"])
def test_unified_cli_accepts_explicit_permission_mode(mode: str) -> None:
    assert parse_rova_cli_args(["--permission", mode]).permission == mode


def test_unified_cli_rejects_unknown_permission_mode() -> None:
    with pytest.raises(SystemExit):
        parse_rova_cli_args(["--permission", "unknown"])


@pytest.mark.parametrize("value", ["0", "-1", "33"])
def test_unified_cli_rejects_max_turns_outside_product_limit(value: str) -> None:
    with pytest.raises(SystemExit):
        parse_rova_cli_args(["--max-turns", value, "continue"])


@pytest.mark.asyncio
async def test_unified_cli_composes_one_runtime_from_explicit_options(monkeypatch, tmp_path: Path, capsys) -> None:
    captured: dict[str, object] = {}
    local_context = LocalResearchContext((
        LocalContextItem("L1", "notes.md", 5, "a" * 64, "notes"),
    ))
    runtime = _runtime(tmp_path, response="Completed.")

    class Settings(_FakeAppSettings):
        artifact_root = tmp_path / "artifacts"

        def to_model(self):
            return Model(provider="mock")

        def to_compaction_policy(self):
            return None

    def build_runtime(**kwargs):
        captured.update(kwargs)
        return runtime

    monkeypatch.setattr(cli, "load_local_research_context", lambda paths: local_context, raising=False)
    monkeypatch.setattr(cli, "AppSettings", type("AppSettings", (), {"from_env": classmethod(lambda cls: Settings())}), raising=False)
    monkeypatch.setattr(cli, "WebSettings", type("WebSettings", (), {"from_env": classmethod(lambda cls: object())}), raising=False)
    monkeypatch.setattr(cli, "create_web_backends", lambda settings: ("search", "fetch"), raising=False)
    monkeypatch.setattr(cli, "build_rova_runtime", build_runtime, raising=False)

    workspace = tmp_path / "workspace"
    data_dir = tmp_path / "runtime-data"
    await run_rova_cli([
        "--workspace", str(workspace), "--web", "--context-path", "notes.md", "--data-dir", str(data_dir), "--max-turns", "16", "--permission", "full", "summarize",
    ])

    assert captured["workspace_root"] == workspace
    assert captured["web_search_backend"] == "search"
    assert captured["webpage_fetcher"] == "fetch"
    assert captured["local_context"] is local_context
    assert captured["permission_mode"] == "full"
    assert captured["session_root"] == data_dir / "sessions"
    assert captured["artifact_root"] == data_dir / "artifacts"
    assert captured["memory_root"] == data_dir / "memory"
    assert captured["skill_root"] == data_dir / "skills"
    assert captured["max_turns"] == 16
    assert runtime.session.prompts == ["summarize"]
    rendered = capsys.readouterr().out
    assert "Permission mode: full" in rendered
    assert "Shell commands are not sandboxed." in rendered


@pytest.mark.asyncio
async def test_unified_cli_data_dir_isolates_session_and_artifact_writes(monkeypatch, tmp_path: Path) -> None:
    async def stream(_model, _context, _options):
        yield StreamDone(AssistantMessage([TextBlock("saved answer")]))

    configured_artifact_root = tmp_path / "configured-artifacts"

    class Settings(_FakeAppSettings):
        artifact_root = configured_artifact_root

        def to_model(self):
            return Model(provider="mock")

        def to_compaction_policy(self):
            return None

    monkeypatch.setattr(cli, "stream_simple", stream)
    monkeypatch.setattr(cli, "AppSettings", type("AppSettings", (), {"from_env": classmethod(lambda cls: Settings())}))
    data_dir = tmp_path / "isolated-data"

    await run_rova_cli(["--data-dir", str(data_dir), "--save", "save", "this"])

    assert list((data_dir / "sessions").glob("*.jsonl"))
    artifacts = list((data_dir / "artifacts").glob("*.json"))
    assert len(artifacts) == 1
    assert not configured_artifact_root.exists()


@pytest.mark.asyncio
async def test_unified_cli_without_data_dir_preserves_runtime_default_roots(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}
    runtime = _runtime(tmp_path, response="done")

    class Settings(_FakeAppSettings):
        artifact_root = None

        def to_model(self):
            return Model(provider="mock")

        def to_compaction_policy(self):
            return None

    def build_runtime(**kwargs):
        captured.update(kwargs)
        return runtime

    monkeypatch.setattr(cli, "load_local_research_context", lambda paths: LocalResearchContext(()), raising=False)
    monkeypatch.setattr(cli, "AppSettings", type("AppSettings", (), {"from_env": classmethod(lambda cls: Settings())}))
    monkeypatch.setattr(cli, "build_rova_runtime", build_runtime)

    await run_rova_cli(["hello"])

    default_paths = RovaDataPaths.resolve()
    assert captured["session_root"] == default_paths.sessions
    assert captured["artifact_root"] == default_paths.artifacts
    assert captured["memory_root"] == default_paths.memory
    assert captured["skill_root"] == default_paths.skills


@pytest.mark.asyncio
async def test_unified_cli_repl_reuses_the_same_runtime(monkeypatch, tmp_path: Path) -> None:
    runtime = _runtime(tmp_path, response="Acknowledged.")
    inputs = iter(["coding", "second request", "/q"])

    class Settings(_FakeAppSettings):
        artifact_root = tmp_path / "artifacts"

        def to_model(self):
            return Model(provider="mock")

        def to_compaction_policy(self):
            return None

    monkeypatch.setattr(cli, "load_local_research_context", lambda paths: LocalResearchContext(()), raising=False)
    monkeypatch.setattr(cli, "AppSettings", type("AppSettings", (), {"from_env": classmethod(lambda cls: Settings())}), raising=False)
    monkeypatch.setattr(cli, "build_rova_runtime", lambda **kwargs: runtime, raising=False)

    await run_rova_cli([], input_fn=_repl_inputs(inputs))

    assert runtime.session.prompts == ["coding", "second request"]
    assert runtime.session.close_calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("exit_command", ["exit", "quit", "/q"])
async def test_unified_cli_repl_exit_commands_close_without_prompting(monkeypatch, tmp_path: Path, capsys, exit_command: str) -> None:
    runtime = _runtime(tmp_path, response="unused")

    class Settings(_FakeAppSettings):
        artifact_root = tmp_path / "artifacts"

        def to_model(self):
            return Model(provider="mock")

        def to_compaction_policy(self):
            return None

    monkeypatch.setattr(cli, "load_local_research_context", lambda paths: LocalResearchContext(()), raising=False)
    monkeypatch.setattr(cli, "AppSettings", type("AppSettings", (), {"from_env": classmethod(lambda cls: Settings())}), raising=False)
    monkeypatch.setattr(cli, "build_rova_runtime", lambda **kwargs: runtime, raising=False)

    await run_rova_cli([], input_fn=_repl_inputs(iter([exit_command])))

    assert runtime.session.prompts == []
    assert runtime.session.close_calls == 1
    output = capsys.readouterr().out
    assert "Session closed: session-1" in output
    assert "Goodbye." in output


@pytest.mark.asyncio
async def test_unified_cli_repl_eof_closes_without_traceback(monkeypatch, tmp_path: Path, capsys) -> None:
    runtime = _runtime(tmp_path, response="unused")

    class Settings(_FakeAppSettings):
        artifact_root = tmp_path / "artifacts"

        def to_model(self):
            return Model(provider="mock")

        def to_compaction_policy(self):
            return None

    def raise_eof(_prompt: str) -> str:
        raise EOFError

    monkeypatch.setattr(cli, "load_local_research_context", lambda paths: LocalResearchContext(()), raising=False)
    monkeypatch.setattr(cli, "AppSettings", type("AppSettings", (), {"from_env": classmethod(lambda cls: Settings())}), raising=False)
    monkeypatch.setattr(cli, "build_rova_runtime", lambda **kwargs: runtime, raising=False)

    await run_rova_cli([], input_fn=raise_eof)

    assert runtime.session.prompts == []
    assert runtime.session.close_calls == 1
    output = capsys.readouterr().out
    assert "Session closed: session-1" in output
    assert "Traceback" not in output


@pytest.mark.asyncio
async def test_unified_cli_repl_idle_keyboard_interrupt_closes_session(monkeypatch, tmp_path: Path, capsys) -> None:
    runtime = _runtime(tmp_path, response="unused")

    class Settings(_FakeAppSettings):
        artifact_root = tmp_path / "artifacts"

        def to_model(self):
            return Model(provider="mock")

        def to_compaction_policy(self):
            return None

    def raise_interrupt(_prompt: str) -> str:
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "load_local_research_context", lambda paths: LocalResearchContext(()), raising=False)
    monkeypatch.setattr(cli, "AppSettings", type("AppSettings", (), {"from_env": classmethod(lambda cls: Settings())}), raising=False)
    monkeypatch.setattr(cli, "build_rova_runtime", lambda **kwargs: runtime, raising=False)

    await run_rova_cli([], input_fn=raise_interrupt)

    assert runtime.session.prompts == []
    assert runtime.session.close_calls == 1
    assert "Goodbye." in capsys.readouterr().out


@pytest.mark.asyncio
async def test_unified_cli_repl_ignores_empty_input_and_keeps_session_open(monkeypatch, tmp_path: Path) -> None:
    runtime = _runtime(tmp_path, response="Acknowledged.")
    inputs = iter(["", "hello", "/q"])

    class Settings(_FakeAppSettings):
        artifact_root = tmp_path / "artifacts"

        def to_model(self):
            return Model(provider="mock")

        def to_compaction_policy(self):
            return None

    monkeypatch.setattr(cli, "load_local_research_context", lambda paths: LocalResearchContext(()), raising=False)
    monkeypatch.setattr(cli, "AppSettings", type("AppSettings", (), {"from_env": classmethod(lambda cls: Settings())}), raising=False)
    monkeypatch.setattr(cli, "build_rova_runtime", lambda **kwargs: runtime, raising=False)

    await run_rova_cli([], input_fn=_repl_inputs(inputs))

    assert runtime.session.prompts == ["hello"]
    assert runtime.session.close_calls == 1


@pytest.mark.asyncio
async def test_unified_cli_save_writes_a_user_requested_output_after_final_response(monkeypatch, tmp_path: Path) -> None:
    source_store = ResearchSourceStore()
    source = source_store.register(SearchHit("Evidence", "https://example.test/evidence", "snippet"))
    source_store.set_content(source.source_id, "fetched evidence")
    runtime = _runtime(tmp_path, response="Evidence supports the answer. [S1]", source_store=source_store)

    class Settings(_FakeAppSettings):
        artifact_root = tmp_path / "artifacts"

        def to_model(self):
            return Model(provider="mock")

        def to_compaction_policy(self):
            return None

    monkeypatch.setattr(cli, "load_local_research_context", lambda paths: LocalResearchContext(()), raising=False)
    monkeypatch.setattr(cli, "AppSettings", type("AppSettings", (), {"from_env": classmethod(lambda cls: Settings())}), raising=False)
    monkeypatch.setattr(cli, "WebSettings", type("WebSettings", (), {"from_env": classmethod(lambda cls: object())}), raising=False)
    monkeypatch.setattr(cli, "create_web_backends", lambda settings: ("search", "fetch"), raising=False)
    monkeypatch.setattr(cli, "build_rova_runtime", lambda **kwargs: runtime, raising=False)

    await run_rova_cli(["--web", "--save", "summarize", "evidence"])

    files = list((tmp_path / "artifacts").glob("*.json"))
    assert len(files) == 1
    envelope = json.loads(files[0].read_text(encoding="utf-8"))
    assert envelope["artifact_kind"] == "user_requested_output"
    assert "tool_call_id" not in envelope
    assert "## Source Snapshot" in envelope["raw_output"]
    assert "[S1] Evidence" in envelope["raw_output"]
    assert "URL: https://example.test/evidence" in envelope["raw_output"]
    assert "status: fetched" in envelope["raw_output"]
    assert "fetched evidence" not in envelope["raw_output"]


@pytest.mark.asyncio
async def test_unified_cli_workspace_smoke_injects_environment_into_fake_provider(monkeypatch, tmp_path: Path) -> None:
    received_system_prompts: list[str] = []

    async def stream(_model, context, _options):
        received_system_prompts.append(context.system_prompt)
        yield StreamDone(AssistantMessage([TextBlock("done")]))

    class Settings(_FakeAppSettings):
        artifact_root = tmp_path / "artifacts"

        def to_model(self):
            return Model(provider="mock")

        def to_compaction_policy(self):
            return None

    monkeypatch.setattr(cli, "stream_simple", stream)
    monkeypatch.setattr(cli, "AppSettings", type("AppSettings", (), {"from_env": classmethod(lambda cls: Settings())}))
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    await run_rova_cli(["--workspace", str(workspace), "inspect", "environment"])

    provider_context = received_system_prompts[0]
    assert "Runtime facts:" in provider_context
    assert "Current time:" in provider_context
    assert "OS:" in provider_context
    assert "Python version:" not in provider_context
    assert "Runtime Python:" not in provider_context
    assert f"Workspace shell cwd: {workspace.resolve()}" in provider_context


def test_unified_cli_renderer_displays_tool_start_end_and_local_duration(monkeypatch, tmp_path: Path, capsys) -> None:
    listeners = []

    class FakeAgent:
        def subscribe(self, listener):
            listeners.append(listener)
            return lambda: None

    clock = iter([10.0, 12.3, 13.0, 13.5])
    monkeypatch.setattr(cli, "monotonic", lambda: next(clock), raising=False)
    _subscribe_console_renderer(FakeAgent(), workspace_root=tmp_path)

    listeners[0](AgentEvent("tool_execution_start", tool_call_id="shell-1", tool_name="shell", args={"command": "python snake.py"}))
    listeners[0](AgentEvent("tool_execution_end", tool_call_id="shell-1", tool_name="shell", is_error=False))
    listeners[0](AgentEvent("tool_execution_start", tool_call_id="read-1", tool_name="read", args={"path": "src/main.py"}))
    listeners[0](AgentEvent("tool_execution_end", tool_call_id="read-1", tool_name="read", is_error=True))

    output = capsys.readouterr().out
    assert output.index("[tool:start] shell") < output.index("[tool:end] shell")
    assert "Command:\npython snake.py" in output
    assert f"cwd:\n{tmp_path}" in output
    assert "status: success" in output
    assert "duration: 2.3s" in output
    assert "[tool:start] read" in output
    assert "status: error" in output


def test_unified_cli_renderer_appends_fetched_citation_source_details(capsys) -> None:
    listeners = []
    source_store = ResearchSourceStore()
    source = source_store.register(SearchHit("LangGraph documentation", "https://example.test/langgraph", "snippet"))
    source_store.set_content(source.source_id, "fetched body")

    class FakeAgent:
        def subscribe(self, listener):
            listeners.append(listener)
            return lambda: None

    _subscribe_console_renderer(FakeAgent(), source_store=source_store)
    listeners[0](AgentEvent("message_end", message=AssistantMessage([TextBlock("LangGraph uses StateGraph [S1].")])) )

    output = capsys.readouterr().out
    assert "LangGraph uses StateGraph [S1]." in output
    assert "Sources:" in output
    assert "[S1]" in output
    assert "Title:\nLangGraph documentation" in output
    assert "URL:\nhttps://example.test/langgraph" in output
    assert "Status:\nfetched" in output
    assert "fetched body" not in output


def test_unified_cli_renderer_marks_search_only_and_missing_citations_without_failing(capsys) -> None:
    listeners = []
    source_store = ResearchSourceStore()
    source_store.register(SearchHit("Discovery result", "https://example.test/discovery", "snippet"))

    class FakeAgent:
        def subscribe(self, listener):
            listeners.append(listener)
            return lambda: None

    _subscribe_console_renderer(FakeAgent(), source_store=source_store)
    listeners[0](AgentEvent("message_end", message=AssistantMessage([TextBlock("Discovery [S1], unavailable [S99].")])) )

    output = capsys.readouterr().out
    assert "Status:\nsearch_only" in output
    assert "verified evidence" not in output
    assert "[S99]\nSource unavailable" in output


def test_unified_cli_renderer_does_not_append_sources_without_citations(capsys) -> None:
    listeners = []

    class FakeAgent:
        def subscribe(self, listener):
            listeners.append(listener)
            return lambda: None

    _subscribe_console_renderer(FakeAgent(), source_store=ResearchSourceStore())
    listeners[0](AgentEvent("message_end", message=AssistantMessage([TextBlock("hello")])) )

    output = capsys.readouterr().out
    assert output == "hello\n"
    assert "Sources:" not in output


def _runtime(tmp_path: Path, *, response: str, source_store: ResearchSourceStore | None = None):
    class FakeAgent:
        def subscribe(self, listener):
            return lambda: None

    class FakeSession:
        session_id = "session-1"

        def __init__(self):
            self.prompts: list[str] = []
            self.close_calls = 0

        async def prompt(self, prompt: str):
            self.prompts.append(prompt)
            return [AssistantMessage([TextBlock(response)])]

        def close(self):
            self.close_calls += 1

    runtime = SimpleNamespace(
        agent=FakeAgent(),
        session=FakeSession(),
        artifact_store=FileArtifactStore(tmp_path / "artifacts"),
        workspace=None,
        source_store=source_store,
        local_context=LocalResearchContext(()),
        close_calls=0,
    )

    async def prompt(text: str):
        return await runtime.session.prompt(text)

    runtime.prompt = prompt

    async def close() -> None:
        runtime.close_calls += 1
        runtime.session.close()

    runtime.close = close
    return runtime


def _repl_inputs(inputs):
    def read(_prompt: str) -> str:
        try:
            return next(inputs)
        except StopIteration as error:
            raise EOFError from error

    return read
