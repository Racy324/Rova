from __future__ import annotations

import os
import tempfile
from pathlib import Path

from rova.agent_core.tools import ToolExecutionError


class WorkspaceError(ToolExecutionError):
    """A user-facing failure while resolving a workspace path."""


class CodingToolError(ToolExecutionError):
    """A user-facing failure from a coding tool operation."""


class Workspace:
    """The injected, canonical filesystem boundary shared by coding tools."""

    def __init__(self, root: Path) -> None:
        try:
            resolved_root = Path(root).resolve()
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            raise WorkspaceError(f"invalid workspace root: {error}") from error
        if not resolved_root.exists():
            raise WorkspaceError("workspace root does not exist")
        if not resolved_root.is_dir():
            raise WorkspaceError("workspace root is not a directory")
        self.root = resolved_root

    def resolve(self, requested_path: str) -> Path:
        if not isinstance(requested_path, str):
            raise WorkspaceError("path must be text")
        candidate = Path(requested_path)
        try:
            resolved = candidate.resolve() if candidate.is_absolute() else (self.root / candidate).resolve()
            resolved.relative_to(self.root)
        except (OSError, RuntimeError, ValueError) as error:
            raise WorkspaceError("path escapes workspace root") from error
        return resolved

    def display_path(self, path: Path) -> str:
        try:
            return Path(path).resolve().relative_to(self.root).as_posix()
        except (OSError, RuntimeError, ValueError) as error:
            raise WorkspaceError("path escapes workspace root") from error

    def read_text(self, path: Path) -> str:
        if not path.exists():
            raise CodingToolError("file not found")
        if not path.is_file():
            raise CodingToolError("not a file")
        try:
            data = path.read_bytes()
        except OSError as error:
            raise CodingToolError(f"unable to read file: {error}") from error
        if b"\x00" in data:
            raise CodingToolError("file is not UTF-8 text")
        try:
            return data.decode("utf-8")
        except UnicodeDecodeError as error:
            raise CodingToolError("file is not UTF-8 text") from error

    def write_text(self, path: Path, content: str) -> None:
        if not isinstance(content, str):
            raise CodingToolError("content must be text")
        parent = path.parent
        if not parent.exists():
            raise CodingToolError("parent directory does not exist")
        if not parent.is_dir():
            raise CodingToolError("parent is not a directory")
        if path.exists() and not path.is_file():
            raise CodingToolError("target is not a regular file")
        try:
            descriptor, temporary_name = tempfile.mkstemp(prefix=".Rova-", dir=parent)
            temporary_path = Path(temporary_name)
            try:
                with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as temporary_file:
                    temporary_file.write(content)
                    temporary_file.flush()
                    os.fsync(temporary_file.fileno())
                temporary_path.replace(path)
            except Exception:
                temporary_path.unlink(missing_ok=True)
                raise
        except CodingToolError:
            raise
        except OSError as error:
            raise CodingToolError(f"unable to write file: {error}") from error
