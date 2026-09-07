from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RovaDataPaths:
    """The local directories owned by one Rova installation."""

    root: Path

    @classmethod
    def resolve(cls, data_dir: Path | None = None) -> "RovaDataPaths":
        root = data_dir
        if root is None:
            configured = os.environ.get("ROVA_DATA_DIR")
            root = Path(configured).expanduser() if configured else Path.home() / ".rova"
        return cls(Path(root).expanduser())

    @property
    def sessions(self) -> Path:
        return self.root / "sessions"

    @property
    def artifacts(self) -> Path:
        return self.root / "artifacts"

    @property
    def traces(self) -> Path:
        return self.root / "traces"

    @property
    def memory(self) -> Path:
        return self.root / "memory"

    @property
    def skills(self) -> Path:
        return self.root / "skills"

    @property
    def extensions(self) -> Path:
        return self.root / "extensions"

    @property
    def experience(self) -> Path:
        return self.root / "experience"

    @property
    def sandboxes(self) -> Path:
        return self.root / "sandboxes"
