from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from rova.app.memory import (
    FileMemoryStore,
    MemoryDocumentAction,
    MemoryDocumentUpdate,
    MemorySnapshot,
    MemoryUpdate,
    create_memory_tools,
)
from rova.ai.messages import ToolCall
from rova.agent_core.tools import ToolRegistry
from rova.app.workspace.instructions import load_workspace_instruction


def _update(*, user: tuple[MemoryDocumentAction, str] = (MemoryDocumentAction.NOOP, ""), memory: tuple[MemoryDocumentAction, str] = (MemoryDocumentAction.NOOP, "")) -> MemoryUpdate:
    return MemoryUpdate(
        user=MemoryDocumentUpdate(*user),
        memory=MemoryDocumentUpdate(*memory),
    )


def test_file_memory_store_creates_empty_root_and_loads_missing_documents(tmp_path: Path) -> None:
    root = tmp_path / "memory"

    snapshot = FileMemoryStore(root).load_snapshot()

    assert root.is_dir()
    assert snapshot == MemorySnapshot()
    assert not (root / "USER.md").exists()
    assert not (root / "MEMORY.md").exists()


def test_memory_snapshot_is_data_only_and_does_not_render_provider_context() -> None:
    snapshot = MemorySnapshot(
        user_markdown="## Coding\n\n- Prefer simple composition.",
        memory_markdown="## Project Facts\n\n- Uses Python.",
    )

    assert snapshot.user_markdown == "## Coding\n\n- Prefer simple composition."
    assert snapshot.memory_markdown == "## Project Facts\n\n- Uses Python."
    assert not hasattr(snapshot, "render_for_provider")


def test_workspace_instruction_uses_strict_single_file_precedence(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "Claude.md").write_text("claude", encoding="utf-8")
    (workspace / "AGENTS.md").write_text("agents", encoding="utf-8")
    (workspace / "Hermes.md").write_text("hermes", encoding="utf-8")

    instruction = load_workspace_instruction(workspace)

    assert instruction.filename == "Hermes.md"
    assert instruction.content == "hermes"


def test_missing_workspace_instruction_is_normal(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    assert load_workspace_instruction(workspace).content == ""


def test_malformed_memory_document_degrades_to_empty_snapshot(tmp_path: Path) -> None:
    (tmp_path / "USER.md").write_bytes(b"\xff\xfe")

    snapshot = FileMemoryStore(tmp_path).load_snapshot()

    assert snapshot.user_markdown == ""


@pytest.mark.asyncio
async def test_file_memory_store_reloads_latest_document_inside_lock_before_update(tmp_path: Path) -> None:
    store = FileMemoryStore(tmp_path / "memory")
    await store.update(lambda _snapshot: _immediate(_update(memory=(MemoryDocumentAction.ADD, "## Facts\n\n- initial"))), max_chars=200)

    async def append(label: str) -> None:
        async def build(snapshot: MemorySnapshot) -> MemoryUpdate:
            return _update(memory=(MemoryDocumentAction.UPDATE, f"{snapshot.memory_markdown}\n- {label}"))

        await store.update(build, max_chars=200)

    await asyncio.gather(append("first"), append("second"))

    content = store.load_snapshot().memory_markdown
    assert "- initial" in content
    assert "- first" in content
    assert "- second" in content


@pytest.mark.asyncio
async def test_file_memory_store_rejects_oversized_update_without_replacing_existing_file(tmp_path: Path) -> None:
    store = FileMemoryStore(tmp_path / "memory")
    await store.update(lambda _snapshot: _immediate(_update(user=(MemoryDocumentAction.ADD, "## Communication\n\n- concise"))), max_chars=80)

    with pytest.raises(ValueError, match="maximum length"):
        await store.update(
            lambda _snapshot: _immediate(_update(user=(MemoryDocumentAction.UPDATE, "x" * 81))),
            max_chars=80,
        )

    assert store.load_snapshot().user_markdown == "## Communication\n\n- concise"


@pytest.mark.asyncio
async def test_memory_store_applies_update_and_delete_to_independent_documents(tmp_path: Path) -> None:
    store = FileMemoryStore(tmp_path / "memory")
    await store.update(
        lambda _snapshot: _immediate(_update(
            user=(MemoryDocumentAction.ADD, "## Preferences\n\n- Python"),
            memory=(MemoryDocumentAction.ADD, "## Facts\n\n- Old fact"),
        )),
        max_chars=200,
    )

    result = await store.update(
        lambda _snapshot: _immediate(_update(
            user=(MemoryDocumentAction.UPDATE, "## Preferences\n\n- Rust"),
            memory=(MemoryDocumentAction.DELETE, ""),
        )),
        max_chars=200,
    )

    assert result.changed_documents == ("USER.md", "MEMORY.md")
    assert store.load_snapshot().user_markdown == "## Preferences\n\n- Rust"
    assert store.load_snapshot().memory_markdown == ""
    assert not (store.root / "MEMORY.md").exists()


@pytest.mark.asyncio
async def test_memory_manage_immediately_applies_a_complete_explicit_update(tmp_path: Path) -> None:
    store = FileMemoryStore(tmp_path / "memory")
    registry = ToolRegistry(create_memory_tools(store, max_chars=200))

    result = await registry.execute(
        ToolCall(
            "memory-1",
            "memory_manage",
            {
                "user_action": "ADD",
                "user_markdown": "- Prefer concise reports",
                "memory_action": "ADD",
                "memory_markdown": "- Baseline uses seed 0",
            },
        )
    )

    assert not result.is_error
    assert result.text == "Updated memory: USER.md, MEMORY.md"
    assert store.load_snapshot() == MemorySnapshot(
        user_markdown="- Prefer concise reports",
        memory_markdown="- Baseline uses seed 0",
    )


@pytest.mark.asyncio
async def test_memory_manage_rejects_invalid_action_without_partial_write(tmp_path: Path) -> None:
    store = FileMemoryStore(tmp_path / "memory")
    await store.update(
        lambda _snapshot: _immediate(_update(user=(MemoryDocumentAction.ADD, "- Existing preference"))),
        max_chars=200,
    )
    registry = ToolRegistry(create_memory_tools(store, max_chars=200))

    result = await registry.execute(
        ToolCall(
            "memory-2",
            "memory_manage",
            {
                "user_action": "UPDATE",
                "user_markdown": "- This must not persist",
                "memory_action": "BAD",
                "memory_markdown": "- invalid",
            },
        )
    )

    assert result.is_error
    assert result.metadata["outcome"] == "tool_execution_error"
    assert store.load_snapshot().user_markdown == "- Existing preference"
    assert store.load_snapshot().memory_markdown == ""


async def _immediate(value: MemoryUpdate) -> MemoryUpdate:
    return value
