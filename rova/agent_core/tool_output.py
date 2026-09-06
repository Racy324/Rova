from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Protocol, Sequence

from rova.ai.messages import TextBlock


class ArtifactStoreError(RuntimeError):
    """Raw tool output could not be durably persisted."""


class ToolOutputStrategy(str, Enum):
    HEAD = "head"
    TAIL = "tail"
    HEAD_TAIL = "head_tail"


@dataclass(frozen=True)
class ToolOutputLimits:
    max_lines: int = 500
    max_bytes: int = 50 * 1024

    def __post_init__(self) -> None:
        for name, value in (("max_lines", self.max_lines), ("max_bytes", self.max_bytes)):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.max_bytes < 32:
            raise ValueError("max_bytes must leave room for a truncation marker")


@dataclass(frozen=True)
class ArtifactReference:
    artifact_id: str
    media_type: str
    byte_count: int
    sha256: str
    created_at: datetime
    run_id: str | None = None
    session_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "media_type": self.media_type,
            "byte_count": self.byte_count,
            "sha256": self.sha256,
            "created_at": self.created_at.isoformat(),
            "run_id": self.run_id,
            "session_id": self.session_id,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ArtifactReference":
        return cls(
            artifact_id=str(value["artifact_id"]),
            media_type=str(value["media_type"]),
            byte_count=int(value["byte_count"]),
            sha256=str(value["sha256"]),
            created_at=datetime.fromisoformat(str(value["created_at"])),
            run_id=value.get("run_id"),
            session_id=value.get("session_id"),
        )


@dataclass(frozen=True)
class ToolOutputMetadata:
    schema_version: int
    # `truncated` is retained for backward-compatible consumers.  It describes
    # only the preview, not whether the raw result was durably retained.
    truncated: bool
    externalized: bool
    artifact_ref: str
    original_size_chars: int
    preview_truncated: bool
    preview_size_chars: int
    strategy: ToolOutputStrategy
    original_line_count: int
    original_byte_count: int
    preview_line_count: int
    preview_byte_count: int
    omitted_line_count: int
    omitted_byte_count: int
    truncated_line_count: int
    artifact: ArtifactReference

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "truncated": self.truncated,
            "externalized": self.externalized,
            "artifact_ref": self.artifact_ref,
            "original_size_chars": self.original_size_chars,
            "preview_truncated": self.preview_truncated,
            "preview_size_chars": self.preview_size_chars,
            "strategy": self.strategy.value,
            "original_line_count": self.original_line_count,
            "original_byte_count": self.original_byte_count,
            "preview_line_count": self.preview_line_count,
            "preview_byte_count": self.preview_byte_count,
            "omitted_line_count": self.omitted_line_count,
            "omitted_byte_count": self.omitted_byte_count,
            "truncated_line_count": self.truncated_line_count,
            "artifact": self.artifact.to_dict(),
        }


@dataclass(frozen=True)
class ProcessedToolOutput:
    preview: list[TextBlock]
    metadata: ToolOutputMetadata


class ArtifactStore(Protocol):
    def write_text(
        self,
        raw_text: str,
        *,
        tool_call_id: str,
        tool_name: str,
        is_error: bool,
        run_id: str | None,
        session_id: str | None,
    ) -> ArtifactReference: ...


@dataclass(frozen=True)
class ToolOutputScope:
    run_id: str | None = None
    session_id: str | None = None


class ToolOutputProcessor:
    def __init__(self, store: ArtifactStore, limits: ToolOutputLimits | None = None) -> None:
        self._store = store
        self._limits = limits or ToolOutputLimits()

    def process(
        self,
        tool_call_id: str,
        tool_name: str,
        content: str | Sequence[TextBlock],
        metadata: dict[str, Any],
        *,
        is_error: bool,
        scope: ToolOutputScope | None = None,
    ) -> ProcessedToolOutput:
        raw_text = content if isinstance(content, str) else "".join(block.text for block in content)
        scope = scope or ToolOutputScope()
        artifact = self._store.write_text(
            raw_text,
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            is_error=is_error,
            run_id=scope.run_id,
            session_id=scope.session_id,
        )
        strategy = _strategy_for(tool_name)
        preview, truncated_line_count = _preview(raw_text, self._limits, strategy, artifact.artifact_id)
        original_bytes = len(raw_text.encode("utf-8"))
        preview_bytes = len(preview.encode("utf-8"))
        raw_lines = _line_count(raw_text)
        preview_lines = _line_count(preview)
        truncated = preview != raw_text
        metadata_value = ToolOutputMetadata(
            schema_version=1,
            truncated=truncated,
            externalized=True,
            artifact_ref=artifact.artifact_id,
            original_size_chars=len(raw_text),
            preview_truncated=truncated,
            preview_size_chars=len(preview),
            strategy=strategy,
            original_line_count=raw_lines,
            original_byte_count=original_bytes,
            preview_line_count=preview_lines,
            preview_byte_count=preview_bytes,
            omitted_line_count=max(0, raw_lines - _retained_raw_line_count(preview, raw_text)),
            omitted_byte_count=max(0, original_bytes - _raw_bytes_visible(preview, raw_text)),
            truncated_line_count=truncated_line_count,
            artifact=artifact,
        )
        return ProcessedToolOutput([TextBlock(preview)], metadata_value)


def _strategy_for(tool_name: str) -> ToolOutputStrategy:
    return {
        "read": ToolOutputStrategy.HEAD,
        "shell": ToolOutputStrategy.TAIL,
        "search": ToolOutputStrategy.HEAD_TAIL,
    }.get(tool_name, ToolOutputStrategy.HEAD)


def _preview(raw: str, limits: ToolOutputLimits, strategy: ToolOutputStrategy, artifact_id: str) -> tuple[str, int]:
    if _line_count(raw) <= limits.max_lines and len(raw.encode("utf-8")) <= limits.max_bytes:
        return raw, 0
    marker = f"\n[tool output truncated: artifact={artifact_id}]\n"
    marker_bytes = len(marker.encode("utf-8"))
    available = limits.max_bytes - marker_bytes
    if available < 1:
        raise ValueError("max_bytes is too small for the truncation marker")
    lines = raw.splitlines()
    if not lines and raw:
        lines = [raw]
    selected = _selected_lines(lines, limits.max_lines, strategy)
    rendered, partial = _fit_lines(selected, available, strategy)
    if strategy is ToolOutputStrategy.TAIL:
        preview = marker + rendered
    elif strategy is ToolOutputStrategy.HEAD_TAIL and len(selected) > 1:
        split = (len(selected) + 1) // 2
        preview = "\n".join(selected[:split])
        # Refit after injecting the marker between head and tail.
        rendered, partial = _fit_lines(selected, available, strategy)
        split = (len(selected) + 1) // 2
        preview = "\n".join(rendered.split("\n")[:split]) + marker + "\n".join(rendered.split("\n")[split:])
        if len(preview.encode("utf-8")) > limits.max_bytes:
            preview = rendered + marker
    else:
        preview = rendered + marker
    while len(preview.encode("utf-8")) > limits.max_bytes:
        preview = _utf8_prefix(preview, limits.max_bytes)
    return preview, partial


def _selected_lines(lines: list[str], max_lines: int, strategy: ToolOutputStrategy) -> list[str]:
    if len(lines) <= max_lines:
        return list(lines)
    if strategy is ToolOutputStrategy.HEAD:
        return lines[:max_lines]
    if strategy is ToolOutputStrategy.TAIL:
        return lines[-max_lines:]
    head = (max_lines + 1) // 2
    tail = max_lines - head
    return lines[:head] + lines[-tail:] if tail else lines[:head]


def _fit_lines(lines: list[str], available: int, strategy: ToolOutputStrategy) -> tuple[str, int]:
    kept: list[str] = []
    iterable = reversed(lines) if strategy is ToolOutputStrategy.TAIL else iter(lines)
    partial = 0
    used = 0
    for line in iterable:
        prefix = "\n" if kept else ""
        cost = len((prefix + line).encode("utf-8"))
        if used + cost <= available:
            kept.append(line)
            used += cost
            continue
        remaining = max(0, available - used - len(prefix.encode("utf-8")))
        if remaining:
            fragment = _utf8_prefix(line, remaining)
            if fragment:
                kept.append(fragment)
                partial = 1
        break
    if strategy is ToolOutputStrategy.TAIL:
        kept.reverse()
    return "\n".join(kept), partial


def _utf8_prefix(text: str, max_bytes: int) -> str:
    encoded = text.encode("utf-8")[:max_bytes]
    return encoded.decode("utf-8", errors="ignore")


def _line_count(text: str) -> int:
    return len(text.splitlines())


def _retained_raw_line_count(preview: str, raw: str) -> int:
    raw_lines = set(raw.splitlines())
    return sum(1 for line in preview.splitlines() if line in raw_lines)


def _raw_bytes_visible(preview: str, raw: str) -> int:
    # The preview includes a marker. Count only characters which are a prefix/suffix
    # of raw data; metadata remains conservative when repeated lines occur.
    marker_index = preview.find("[tool output truncated:")
    visible = preview if marker_index < 0 else preview[:marker_index]
    return min(len(raw.encode("utf-8")), len(visible.encode("utf-8")))
