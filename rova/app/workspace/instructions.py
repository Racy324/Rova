from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class WorkspaceInstructionSnapshot:
    filename: str = ""
    content: str = ""


def load_workspace_instruction(workspace_root: Path | None) -> WorkspaceInstructionSnapshot:
    if workspace_root is None:
        return WorkspaceInstructionSnapshot()
    root = Path(workspace_root)
    for filename in ("Hermes.md", "AGENTS.md", "Claude.md"):
        path = root / filename
        if not path.is_file():
            continue
        try:
            return WorkspaceInstructionSnapshot(filename, path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError):
            return WorkspaceInstructionSnapshot()
    return WorkspaceInstructionSnapshot()
