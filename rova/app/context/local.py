from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Sequence


MAX_CONTEXT_FILES = 8
MAX_CONTEXT_FILE_BYTES = 64 * 1024
MAX_CONTEXT_TOTAL_BYTES = 256 * 1024
_TEXT_SUFFIXES = frozenset({".txt", ".md", ".rst", ".py", ".json", ".toml", ".yaml", ".yml", ".csv"})


class LocalContextError(ValueError):
    pass


@dataclass(frozen=True)
class LocalContextItem:
    label: str
    filename: str
    byte_count: int
    sha256: str
    content: str


@dataclass(frozen=True)
class LocalResearchContext:
    items: tuple[LocalContextItem, ...] = ()

    def render_user_attachment(self) -> str:
        if not self.items:
            return ""
        lines = [
            "--- Attached Context ---",
            "These materials were explicitly selected by the user for this request.",
            "They are local context, not fetched public-web sources. Use [S#] only for fetched external sources.",
        ]
        for item in self.items:
            lines.extend(["", f"[{item.label}] {item.filename}", item.content])
        return "\n".join(lines)


def load_local_research_context(paths: Sequence[Path]) -> LocalResearchContext:
    if len(paths) > MAX_CONTEXT_FILES:
        raise LocalContextError(f"at most {MAX_CONTEXT_FILES} local context files are allowed")
    items: list[LocalContextItem] = []
    seen: set[Path] = set()
    total_bytes = 0
    for index, original_path in enumerate(paths, start=1):
        path = Path(original_path).expanduser().resolve()
        if path in seen:
            raise LocalContextError(f"duplicate local context path: {path.name}")
        seen.add(path)
        if not path.is_file():
            raise LocalContextError(f"local context path must be a regular file: {path}")
        if path.suffix.lower() not in _TEXT_SUFFIXES:
            raise LocalContextError(f"unsupported local context file type: {path.suffix or '<none>'}")
        size = path.stat().st_size
        if size > MAX_CONTEXT_FILE_BYTES:
            raise LocalContextError(f"local context file is too large: {path.name}")
        if total_bytes + size > MAX_CONTEXT_TOTAL_BYTES:
            raise LocalContextError("local context total size is too large")
        try:
            raw = path.read_bytes()
            content = raw.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")
        except UnicodeDecodeError as error:
            raise LocalContextError(f"local context file must be UTF-8: {path.name}") from error
        total_bytes += len(raw)
        items.append(LocalContextItem(f"L{index}", path.name, len(raw), hashlib.sha256(raw).hexdigest(), content))
    return LocalResearchContext(tuple(items))
