from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path

import pytest

from rova.app import memory as memory_module
from rova.app.memory import (
    FileMemoryStore,
    MemoryDocumentAction,
    MemoryDocumentUpdate,
    MemorySnapshot,
    MemoryUpdate,
    create_memory_tools,
)
from rova.ai.messages import ToolCall
from rova.ai.providers.openai_compatible import to_provider_tools
from rova.agent_core.tools import ToolRegistry, ToolRuntime
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


@pytest.mark.skip(reason="Deferred Hardening: Entry identity and sidecar metadata are outside V2 Core.")
def test_entry_snapshot_uses_h2_sections_and_keeps_legacy_markdown_model_visible(tmp_path: Path) -> None:
    root = tmp_path / "memory"
    root.mkdir()
    original = "## Preferences\n\n- Prefer concise reports.\n\n## Background\n\n- Studies agent systems."
    (root / "USER.md").write_text(original, encoding="utf-8")

    api = _v2_memory_api()
    entry_snapshot = FileMemoryStore(root).load_entry_snapshot()

    assert entry_snapshot.snapshot.user_markdown == original
    assert tuple(entry.content for entry in entry_snapshot.user_entries) == (
        "## Preferences\n\n- Prefer concise reports.",
        "## Background\n\n- Studies agent systems.",
    )
    assert all(entry.entry_id.startswith("legacy:USER.md:") for entry in entry_snapshot.user_entries)
    assert all(entry.revision == 0 for entry in entry_snapshot.user_entries)
    assert "legacy:USER.md:" not in entry_snapshot.snapshot.user_markdown
    assert not (root / ".memory-entries.json").exists()


@pytest.mark.skip(reason="Deferred Hardening: manual-edit stale-write reconciliation is outside V2 Core.")
def test_entry_snapshot_preserves_sidecar_identity_and_treats_manual_edit_as_stale(tmp_path: Path) -> None:
    root = tmp_path / "memory"
    root.mkdir()
    original = "## Preferences\n\n- Prefer concise reports."
    (root / "USER.md").write_text(original, encoding="utf-8")
    digest = hashlib.sha256(original.encode("utf-8")).hexdigest()
    (root / ".memory-entries.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "documents": {
                    "USER.md": [
                        {
                            "entry_id": "user-pref-1",
                            "revision": 4,
                            "content_sha256": digest,
                        }
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
    store = FileMemoryStore(root)

    api = _v2_memory_api()
    before = store.load_entry_snapshot()
    assert before.user_entries[0].entry_id == "user-pref-1"
    assert before.user_entries[0].revision == 4

    (root / "USER.md").write_text("## Preferences\n\n- Prefer detailed reports.", encoding="utf-8")
    after = store.load_entry_snapshot()

    assert after.snapshot.user_markdown == "## Preferences\n\n- Prefer detailed reports."
    assert after.user_entries[0].entry_id.startswith("manual:USER.md:")
    assert after.user_entries[0].revision == 0
    with pytest.raises(api.MemoryOperationError, match="unknown or stale"):
        api.validate_memory_operations(
            after,
            [
                api.MemoryOperation(
                    document=api.MemoryDocument.USER,
                    action=MemoryDocumentAction.UPDATE,
                    target_entry_id="user-pref-1",
                    expected_revision=4,
                    content="## Preferences\n\n- Prefer short reports.",
                )
            ],
            max_chars=200,
        )


@pytest.mark.skip(reason="Deferred Hardening: Entry target/revision CAS is outside V2 Core.")
def test_entry_operations_require_a_precise_target_and_expected_revision(tmp_path: Path) -> None:
    api = _v2_memory_api()
    store = FileMemoryStore(tmp_path / "memory")
    entry_snapshot = store.load_entry_snapshot()

    for operation in (
        api.MemoryOperation(
            document=api.MemoryDocument.MEMORY,
            action=MemoryDocumentAction.UPDATE,
            content="## Environment\n\n- Uses Python.",
        ),
        api.MemoryOperation(
            document=api.MemoryDocument.MEMORY,
            action=MemoryDocumentAction.DELETE,
            target_entry_id="missing-revision",
        ),
    ):
        with pytest.raises(api.MemoryOperationError, match="target_entry_id and expected_revision"):
            api.validate_memory_operations(entry_snapshot, [operation], max_chars=200)


@pytest.mark.skip(reason="Deferred Hardening: Entry duplicate and capacity validation is outside V2 Core.")
def test_entry_operations_reject_duplicate_add_workspace_scope_and_capacity_without_writing(tmp_path: Path) -> None:
    api = _v2_memory_api()
    root = tmp_path / "memory"
    root.mkdir()
    original = "## Environment\n\n- Uses Python."
    (root / "MEMORY.md").write_text(original, encoding="utf-8")
    entry_snapshot = FileMemoryStore(root).load_entry_snapshot()

    duplicate = api.MemoryOperation(
        document=api.MemoryDocument.MEMORY,
        action=MemoryDocumentAction.ADD,
        content=original,
    )
    with pytest.raises(api.MemoryOperationError, match="duplicate"):
        api.validate_memory_operations(entry_snapshot, [duplicate], max_chars=200)

    workspace_scoped = api.MemoryOperation(
        document=api.MemoryDocument.MEMORY,
        action=MemoryDocumentAction.ADD,
        content="## Repository\n\n- Uses a private package index.",
        scope=api.MemoryEntryScope.WORKSPACE,
    )
    with pytest.raises(api.MemoryOperationError, match="workspace-specific"):
        api.validate_memory_operations(entry_snapshot, [workspace_scoped], max_chars=200)

    oversized = api.MemoryOperation(
        document=api.MemoryDocument.MEMORY,
        action=MemoryDocumentAction.ADD,
        content="## Long\n\n" + "x" * 200,
    )
    with pytest.raises(api.MemoryOperationError, match="maximum length"):
        api.validate_memory_operations(entry_snapshot, [oversized], max_chars=40)

    assert (root / "MEMORY.md").read_text(encoding="utf-8") == original
    assert not (root / ".memory-entries.json").exists()


def _v2_memory_api():
    required = (
        "MemoryDocument",
        "MemoryEntryScope",
        "MemoryOperation",
        "MemoryOperationError",
        "validate_memory_operations",
    )
    missing = [name for name in required if not hasattr(memory_module, name)]
    assert not missing, f"Task 1 Memory V2 contract API is missing: {', '.join(missing)}"
    return memory_module


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

    result = await ToolRuntime(registry).execute(
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

    result = await ToolRuntime(registry).execute(
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
    assert result.metadata["outcome"] == "tool_input_error"
    assert store.load_snapshot().user_markdown == "- Existing preference"
    assert store.load_snapshot().memory_markdown == ""


def test_memory_manage_provider_schema_uses_runtime_action_enum() -> None:
    tool = create_memory_tools(FileMemoryStore(), max_chars=200)[0].tool
    provider_schema = to_provider_tools([tool])[0]["function"]["parameters"]
    expected = [action.value for action in MemoryDocumentAction]

    assert provider_schema["properties"]["user_action"]["enum"] == expected
    assert provider_schema["properties"]["memory_action"]["enum"] == expected
    assert "USER.md" in provider_schema["properties"]["user_action"]["description"]
    assert "MEMORY.md" in provider_schema["properties"]["memory_action"]["description"]


@pytest.mark.asyncio
@pytest.mark.parametrize("action", [action.value for action in MemoryDocumentAction])
async def test_memory_manage_accepts_every_declared_action(tmp_path: Path, action: str) -> None:
    registry = ToolRegistry(create_memory_tools(FileMemoryStore(tmp_path / "memory"), max_chars=200))
    arguments = {
        "user_action": action,
        "memory_action": "NOOP",
    }
    if action in {"ADD", "UPDATE"}:
        arguments["user_markdown"] = "- Stable user information"

    result = await ToolRuntime(registry).execute(ToolCall("memory-valid", "memory_manage", arguments))

    assert not result.is_error


@pytest.mark.asyncio
@pytest.mark.parametrize("field, invalid", [("user_action", "remember"), ("memory_action", "save"), ("memory_action", "UNKNOWN")])
async def test_memory_manage_reports_invalid_action_with_allowed_values(tmp_path: Path, field: str, invalid: str) -> None:
    registry = ToolRegistry(create_memory_tools(FileMemoryStore(tmp_path / "memory"), max_chars=200))
    arguments = {
        "user_action": "NOOP",
        "memory_action": "NOOP",
    }
    arguments[field] = invalid

    result = await ToolRuntime(registry).execute(ToolCall("memory-invalid", "memory_manage", arguments))

    assert result.is_error
    assert field in result.text
    assert invalid in result.text
    assert "ADD, UPDATE, DELETE, NOOP" in result.text


async def _immediate(value: MemoryUpdate) -> MemoryUpdate:
    return value
