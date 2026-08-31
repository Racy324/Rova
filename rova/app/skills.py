from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
import re
import shutil
from typing import Callable, Final
from uuid import uuid4
import warnings

from rova.ai.messages import TextBlock
from rova.ai.tools import Tool
from rova.agent_core.tools import AgentTool, AgentToolResult, ToolExecutionError

from .file_lock import FileLock, FileLockError
from .paths import RovaDataPaths


_SKILL_NAME: Final[re.Pattern[str]] = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")
_SKILL_FILENAME: Final[str] = "SKILL.md"
_SKILL_DIRECTORY_TEMPLATE: Final[str] = "${ROVA_SKILL_DIR}"


class SkillStoreError(RuntimeError):
    """Expected local Skill storage failure exposed through normal tool errors."""


@dataclass(frozen=True)
class SkillMetadata:
    name: str
    description: str


@dataclass(frozen=True)
class SkillCatalogSnapshot:
    skills: tuple[SkillMetadata, ...] = ()


class FileSkillStore:
    """Small filesystem store for progressively disclosed local Skills."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = RovaDataPaths.resolve().skills if root is None else Path(root)

    def discover_catalog(self) -> SkillCatalogSnapshot:
        self._ensure_root()
        skills: list[SkillMetadata] = []
        try:
            directories = sorted(self.root.iterdir(), key=lambda path: path.name)
        except OSError as error:
            raise SkillStoreError("could not read skills directory") from error
        for directory in directories:
            if not directory.is_dir() or directory.is_symlink():
                continue
            skill_file = directory / _SKILL_FILENAME
            if not skill_file.is_file() or skill_file.is_symlink():
                continue
            try:
                metadata = _parse_skill_metadata(_read_utf8(skill_file), directory.name)
            except SkillStoreError as error:
                warnings.warn(f"Ignoring invalid Skill '{directory.name}': {error}", RuntimeWarning, stacklevel=2)
                continue
            skills.append(metadata)
        return SkillCatalogSnapshot(tuple(skills))

    def read(
        self,
        name: str,
        path: str | None = None,
        *,
        skill_directory_renderer: Callable[[Path], str] | None = None,
    ) -> str:
        skill_directory = self._skill_directory(name)
        target = self._resolve_skill_file(skill_directory, path)
        renderer = skill_directory_renderer or _host_skill_directory
        return _substitute_skill_directory(_read_utf8(target), renderer(skill_directory))

    def resolved_directory(self, name: str) -> Path:
        """Return the installed package directory for an explicitly loaded Skill."""
        return self._skill_directory(name).resolve()

    def create(self, name: str, content: str) -> None:
        _validate_skill_name(name)
        _parse_skill_metadata(_validate_content(content), name)
        self._ensure_root()
        try:
            with FileLock(self.root / ".skills.lock"):
                directory = self.root / name
                if directory.exists() or directory.is_symlink():
                    raise SkillStoreError(f"Skill already exists: {name}")
                created = False
                try:
                    directory.mkdir()
                    created = True
                    _write_utf8_atomically(directory / _SKILL_FILENAME, content)
                except OSError:
                    if created:
                        try:
                            directory.rmdir()
                        except OSError:
                            pass
                    raise
        except FileLockError as error:
            raise SkillStoreError("could not acquire Skill store lock") from error
        except OSError as error:
            raise SkillStoreError(f"could not create Skill: {name}") from error

    def edit(self, name: str, content: str, path: str | None = None) -> None:
        normalized = _validate_content(content)
        if path is None:
            _parse_skill_metadata(normalized, name)
        try:
            self._ensure_root()
            with FileLock(self.root / ".skills.lock"):
                skill_directory = self._skill_directory(name)
                target = self._resolve_skill_file(skill_directory, path, allow_missing=True)
                _write_utf8_atomically(target, normalized)
        except FileLockError as error:
            raise SkillStoreError("could not acquire Skill store lock") from error
        except OSError as error:
            raise SkillStoreError(f"could not update Skill file: {name}") from error

    def delete(self, name: str) -> None:
        try:
            self._ensure_root()
            with FileLock(self.root / ".skills.lock"):
                directory = self._skill_directory(name)
                shutil.rmtree(directory)
        except FileLockError as error:
            raise SkillStoreError("could not acquire Skill store lock") from error
        except OSError as error:
            raise SkillStoreError(f"could not delete Skill: {name}") from error

    def _ensure_root(self) -> None:
        try:
            self.root.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            raise SkillStoreError("could not create skills directory") from error
        try:
            self.root.chmod(0o700)
        except OSError:
            pass

    def _skill_directory(self, name: str) -> Path:
        _validate_skill_name(name)
        self._ensure_root()
        directory = self.root / name
        if not directory.is_dir() or directory.is_symlink():
            raise SkillStoreError(f"Skill not found: {name}")
        try:
            directory.resolve().relative_to(self.root.resolve())
        except ValueError as error:
            raise SkillStoreError("Skill path is outside the skills directory") from error
        return directory

    def _resolve_skill_file(self, directory: Path, path: str | None, *, allow_missing: bool = False) -> Path:
        if path is None:
            target = directory / _SKILL_FILENAME
        else:
            relative = Path(path)
            if relative.is_absolute() or ".." in relative.parts:
                raise SkillStoreError("Skill path must stay within the Skill directory")
            target = directory / relative
        try:
            target.resolve().relative_to(directory.resolve())
        except ValueError as error:
            raise SkillStoreError("Skill path must stay within the Skill directory") from error
        if target.is_symlink() or (not allow_missing and not target.is_file()):
            raise SkillStoreError(f"Skill file not found: {path or _SKILL_FILENAME}")
        if target.exists() and not target.is_file():
            raise SkillStoreError(f"Skill path is not a file: {path or _SKILL_FILENAME}")
        return target


def create_skill_tools(
    store: FileSkillStore,
    *,
    skill_directory_renderer: Callable[[Path], str] | None = None,
) -> list[AgentTool]:
    async def view(_tool_call_id: str, params: dict) -> AgentToolResult:
        name = params["name"]
        path = params.get("path")
        try:
            skill_directory = store.resolved_directory(name)
            renderer = skill_directory_renderer or _host_skill_directory
            rendered_directory = renderer(skill_directory)
            content = store.read(name, path, skill_directory_renderer=renderer)
        except SkillStoreError as error:
            raise ToolExecutionError(str(error), metadata={"outcome": "tool_input_error"}) from error
        label = path or _SKILL_FILENAME
        return AgentToolResult([TextBlock(f"Skill: {name}\nPath: {label}\nSkill directory: {rendered_directory}\n\n{content}")])

    async def manage(_tool_call_id: str, params: dict) -> AgentToolResult:
        action = params["action"]
        name = params["name"]
        path = params.get("path")
        try:
            if action == "create":
                store.create(name, _required_content(params))
                text = f"Created Skill: {name}"
            elif action == "edit":
                store.edit(name, _required_content(params), path)
                text = f"Updated Skill file: {name}/{path or _SKILL_FILENAME}"
            elif action == "delete":
                store.delete(name)
                text = f"Deleted Skill: {name}"
            else:
                raise SkillStoreError(f"unsupported action: {action}")
        except SkillStoreError as error:
            raise ToolExecutionError(str(error), metadata={"outcome": "tool_execution_error"}) from error
        return AgentToolResult([TextBlock(text)])

    return [
        AgentTool(
            Tool(
                "skill_view",
                "Read one local Skill or Skill-relative attachment. Avoid reloading a Skill when its full content is still in the current conversation; reload it after compaction or when needed again.",
                {"name": str, "path": str},
                required=("name",),
            ),
            view,
        ),
        AgentTool(
            Tool(
                "skill_manage",
                "Create, edit, or delete an explicit local Skill. Create or update a Skill only when its method is reusable; do not save one-off task state or debug output as a Skill.",
                {"action": str, "name": str, "content": str, "path": str},
                required=("action", "name"),
            ),
            manage,
        ),
    ]


def _parse_skill_metadata(content: str, expected_name: str) -> SkillMetadata:
    lines = content.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    if not lines or lines[0] != "---":
        raise SkillStoreError("SKILL.md must begin with frontmatter")
    try:
        closing = lines.index("---", 1)
    except ValueError as error:
        raise SkillStoreError("SKILL.md frontmatter is not closed") from error
    fields: dict[str, str] = {}
    for line in lines[1:closing]:
        if not line.strip():
            continue
        key, separator, value = line.partition(":")
        if not separator or not key.strip() or not value.strip():
            raise SkillStoreError("SKILL.md frontmatter must use key: value lines")
        fields[key.strip()] = value.strip().strip('"').strip("'")
    name = fields.get("name", "")
    description = fields.get("description", "")
    _validate_skill_name(name)
    if name != expected_name:
        raise SkillStoreError("SKILL.md name must match its directory name")
    if not description:
        raise SkillStoreError("SKILL.md requires a description")
    return SkillMetadata(name, description)


def _validate_skill_name(name: str) -> None:
    if not isinstance(name, str) or not _SKILL_NAME.fullmatch(name):
        raise SkillStoreError("invalid skill name")


def _validate_content(content: object) -> str:
    if not isinstance(content, str) or not content.strip():
        raise SkillStoreError("Skill content is required")
    return content.replace("\r\n", "\n").replace("\r", "\n").rstrip() + "\n"


def _required_content(params: dict) -> str:
    return _validate_content(params.get("content"))


def _read_utf8(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise SkillStoreError(f"could not read Skill file: {path.name}") from error


def _host_skill_directory(directory: Path) -> str:
    return str(directory.resolve())


def _substitute_skill_directory(content: str, rendered_directory: str) -> str:
    return content.replace(_SKILL_DIRECTORY_TEMPLATE, rendered_directory)


def _write_utf8_atomically(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid4().hex}.tmp"
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
