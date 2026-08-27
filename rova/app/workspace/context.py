from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .workspace import Workspace


@dataclass
class WorkspaceContext:
    """Runtime-local summary of successful explicit Coding tool file operations."""

    workspace: Workspace
    files_read: set[str] = field(default_factory=set)
    files_modified: set[str] = field(default_factory=set)
    files_created: set[str] = field(default_factory=set)

    def record_read(self, path: Path) -> None:
        self.files_read.add(self.workspace.display_path(path))

    def record_write(self, path: Path, *, existed_before: bool) -> None:
        display_path = self.workspace.display_path(path)
        if existed_before:
            if display_path not in self.files_created:
                self.files_modified.add(display_path)
            return
        self.files_created.add(display_path)

    def record_edit(self, path: Path) -> None:
        display_path = self.workspace.display_path(path)
        if display_path not in self.files_created:
            self.files_modified.add(display_path)

    def render_for_provider(self) -> str:
        sections = [
            ("Read", self.files_read),
            ("Modified", self.files_modified),
            ("Created", self.files_created),
        ]
        rendered = ["Current workspace changes:"]
        for heading, paths in sections:
            if paths:
                rendered.append(f"{heading}:")
                rendered.extend(f"- {path}" for path in sorted(paths))
        return "\n".join(rendered) if len(rendered) > 1 else ""
