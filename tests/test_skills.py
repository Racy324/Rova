from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import time

import pytest

from rova.ai.messages import ToolCall
from rova.agent_core.tools import ToolRegistry
from rova.app.file_lock import FileLock
from rova.app.skills import FileSkillStore, SkillStoreError, create_skill_tools


def _skill_markdown(name: str, description: str, body: str = "## Procedure\n\nFollow the checks.") -> str:
    return f"---\nname: {name}\ndescription: {description}\n---\n\n# {name}\n\n{body}\n"


def test_catalog_discovery_handles_missing_empty_and_multiple_skills(tmp_path: Path) -> None:
    store = FileSkillStore(tmp_path / "skills")

    assert store.discover_catalog().skills == ()

    (store.root / "zeta").mkdir()
    (store.root / "zeta" / "SKILL.md").write_text(_skill_markdown("zeta", "Zeta method."), encoding="utf-8")
    (store.root / "alpha").mkdir()
    (store.root / "alpha" / "SKILL.md").write_text(_skill_markdown("alpha", "Alpha method."), encoding="utf-8")

    snapshot = store.discover_catalog()

    assert [(item.name, item.description) for item in snapshot.skills] == [
        ("alpha", "Alpha method."),
        ("zeta", "Zeta method."),
    ]
    assert not hasattr(snapshot, "render_for_provider")


def test_catalog_discovery_skips_malformed_skill_without_hiding_valid_skill(tmp_path: Path) -> None:
    store = FileSkillStore(tmp_path / "skills")
    (store.root / "valid").mkdir(parents=True)
    (store.root / "valid" / "SKILL.md").write_text(_skill_markdown("valid", "Valid method."), encoding="utf-8")
    (store.root / "broken").mkdir()
    (store.root / "broken" / "SKILL.md").write_text("# no frontmatter", encoding="utf-8")

    with pytest.warns(RuntimeWarning, match="broken"):
        snapshot = store.discover_catalog()

    assert [(item.name, item.description) for item in snapshot.skills] == [("valid", "Valid method.")]


def test_store_reads_main_file_and_skill_relative_attachments(tmp_path: Path) -> None:
    store = FileSkillStore(tmp_path / "skills")
    store.create("code-review", _skill_markdown("code-review", "Review code."))
    reference = store.root / "code-review" / "references" / "cpp-guidelines.md"
    reference.parent.mkdir()
    reference.write_text("Check error paths.", encoding="utf-8")

    assert "# code-review" in store.read("code-review")
    assert store.read("code-review", "references/cpp-guidelines.md") == "Check error paths."


@pytest.mark.parametrize("action", ["create", "edit", "delete"])
def test_skill_mutation_waits_for_another_process_holding_the_store_lock(tmp_path: Path, action: str) -> None:
    store = FileSkillStore(tmp_path / "skills")
    store.discover_catalog()
    initial_content = _skill_markdown("code-review", "Initial review.")
    changed_content = _skill_markdown("code-review", "Updated review.")
    if action != "create":
        store.create("code-review", initial_content)
    statement = {
        "create": "store.create('code-review', sys.argv[2])",
        "edit": "store.edit('code-review', sys.argv[2])",
        "delete": "store.delete('code-review')",
    }[action]
    script = (
        "from pathlib import Path\n"
        "from rova.app.skills import FileSkillStore\n"
        "import sys\n"
        "store = FileSkillStore(Path(sys.argv[1]))\n"
        "print('ready', flush=True)\n"
        f"{statement}\n"
    )

    with FileLock(store.root / ".skills.lock"):
        process = subprocess.Popen(
            [sys.executable, "-c", script, str(store.root), changed_content],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        assert process.stdout is not None
        assert process.stdout.readline().strip() == "ready"
        time.sleep(0.15)
        if action == "create":
            assert not (store.root / "code-review").exists()
        else:
            assert (store.root / "code-review" / "SKILL.md").read_text(encoding="utf-8") == initial_content

    _stdout, stderr = process.communicate(timeout=5)
    assert process.returncode == 0, stderr
    if action == "delete":
        assert not (store.root / "code-review").exists()
    else:
        expected_content = changed_content
        assert (store.root / "code-review" / "SKILL.md").read_text(encoding="utf-8") == expected_content


def test_store_edits_skill_main_file_with_matching_frontmatter(tmp_path: Path) -> None:
    store = FileSkillStore(tmp_path / "skills")
    store.create("code-review", _skill_markdown("code-review", "Review code."))

    store.edit("code-review", _skill_markdown("code-review", "Review changes carefully.", "Updated procedure."))

    assert [(item.name, item.description) for item in store.discover_catalog().skills] == [
        ("code-review", "Review changes carefully."),
    ]
    assert "Updated procedure." in store.read("code-review")


@pytest.mark.parametrize("path", ["../outside.md", "/absolute.md", "C:/absolute.md"])
def test_store_rejects_attachment_paths_outside_the_skill(tmp_path: Path, path: str) -> None:
    store = FileSkillStore(tmp_path / "skills")
    store.create("code-review", _skill_markdown("code-review", "Review code."))

    with pytest.raises(SkillStoreError, match="path"):
        store.read("code-review", path)


def test_store_rejects_skill_name_escape_and_missing_files(tmp_path: Path) -> None:
    store = FileSkillStore(tmp_path / "skills")

    with pytest.raises(SkillStoreError, match="skill name"):
        store.create("../escape", _skill_markdown("escape", "Invalid."))
    with pytest.raises(SkillStoreError, match="not found"):
        store.read("missing")


def test_store_rejects_symlink_attachment_escape_when_supported(tmp_path: Path) -> None:
    store = FileSkillStore(tmp_path / "skills")
    store.create("code-review", _skill_markdown("code-review", "Review code."))
    outside = tmp_path / "outside.md"
    outside.write_text("outside", encoding="utf-8")
    link = store.root / "code-review" / "references-link.md"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("symlink creation is unavailable on this platform")

    with pytest.raises(SkillStoreError, match="path"):
        store.read("code-review", "references-link.md")


@pytest.mark.asyncio
async def test_skill_view_returns_a_normal_tool_result(tmp_path: Path) -> None:
    store = FileSkillStore(tmp_path / "skills")
    store.create("code-review", _skill_markdown("code-review", "Review code."))
    registry = ToolRegistry(create_skill_tools(store))

    result = await registry.execute(ToolCall("view-1", "skill_view", {"name": "code-review"}))

    assert result.is_error is False
    assert result.tool_name == "skill_view"
    assert "Skill: code-review" in result.text
    assert "# code-review" in result.text


@pytest.mark.asyncio
async def test_skill_manage_create_edit_and_delete_use_normal_tool_results(tmp_path: Path) -> None:
    store = FileSkillStore(tmp_path / "skills")
    registry = ToolRegistry(create_skill_tools(store))
    content = _skill_markdown("code-review", "Review code.")

    created = await registry.execute(ToolCall("create-1", "skill_manage", {
        "action": "create", "name": "code-review", "content": content,
    }))
    duplicate = await registry.execute(ToolCall("create-2", "skill_manage", {
        "action": "create", "name": "code-review", "content": content,
    }))
    edited = await registry.execute(ToolCall("edit-1", "skill_manage", {
        "action": "edit", "name": "code-review", "content": "Updated.", "path": "references/checks.md",
    }))
    assert edited.is_error is False
    assert (store.root / "code-review" / "references" / "checks.md").read_text(encoding="utf-8") == "Updated.\n"
    deleted = await registry.execute(ToolCall("delete-1", "skill_manage", {
        "action": "delete", "name": "code-review",
    }))
    missing = await registry.execute(ToolCall("delete-2", "skill_manage", {
        "action": "delete", "name": "code-review",
    }))

    assert created.is_error is False
    assert duplicate.is_error is True
    assert "already exists" in duplicate.text
    assert deleted.is_error is False
    assert missing.is_error is True
    assert "not found" in missing.text
    assert not list(store.root.rglob("*.tmp"))


@pytest.mark.asyncio
async def test_skill_manage_rejects_invalid_action_and_attachment_escape(tmp_path: Path) -> None:
    store = FileSkillStore(tmp_path / "skills")
    store.create("code-review", _skill_markdown("code-review", "Review code."))
    registry = ToolRegistry(create_skill_tools(store))

    invalid = await registry.execute(ToolCall("bad-1", "skill_manage", {
        "action": "unknown", "name": "code-review",
    }))
    escaped = await registry.execute(ToolCall("bad-2", "skill_manage", {
        "action": "edit", "name": "code-review", "path": "../escape.md", "content": "no",
    }))

    assert invalid.is_error is True
    assert "unsupported action" in invalid.text
    assert escaped.is_error is True
    assert "path" in escaped.text
