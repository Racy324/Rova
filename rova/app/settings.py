from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from rova.ai.models import Model
from rova.agent_session.compaction import CompactionPolicy
from rova.config import load_project_env
from .paths import RovaDataPaths


@dataclass(frozen=True)
class AppSettings:
    provider: str = "mock"
    model: str = "mock"
    base_url: str | None = None
    context_window: int | None = None
    compaction_reserve_tokens: int | None = None
    compaction_keep_recent_tokens: int | None = None
    artifact_root: Path | None = None
    memory_root: Path | None = None
    memory_provider: str | None = None
    memory_model: str | None = None
    memory_base_url: str | None = None
    memory_context_window: int | None = None
    memory_max_chars: int = 6_000
    experience_review_enabled: bool = True
    experience_review_tool_threshold: int = 10
    experience_review_task_threshold: int = 5
    data_dir: Path | None = None
    terminal_backend: str | None = None
    docker_image: str | None = None

    @classmethod
    def from_env(
        cls,
        env: Mapping[str, str] | None = None,
        *,
        dotenv_path: Path | None = None,
    ) -> "AppSettings":
        source = load_project_env(env, dotenv_path=dotenv_path)
        context_window = source.get("ROVA_CONTEXT_WINDOW")
        reserve_tokens = source.get("ROVA_COMPACTION_RESERVE_TOKENS")
        keep_recent_tokens = source.get("ROVA_COMPACTION_KEEP_RECENT_TOKENS")
        artifact_root = source.get("ROVA_ARTIFACT_ROOT")
        memory_root = source.get("ROVA_MEMORY_ROOT")
        memory_context_window = source.get("ROVA_MEMORY_CONTEXT_WINDOW")
        memory_max_chars = source.get("ROVA_MEMORY_MAX_CHARS")
        experience_review_enabled = source.get("ROVA_EXPERIENCE_REVIEW_ENABLED")
        experience_review_tool_threshold = source.get("ROVA_EXPERIENCE_REVIEW_TOOL_THRESHOLD")
        experience_review_task_threshold = source.get("ROVA_EXPERIENCE_REVIEW_TASK_THRESHOLD")
        data_dir = source.get("ROVA_DATA_DIR")
        terminal_backend = source.get("ROVA_TERMINAL_BACKEND", "").strip().lower() or None
        if terminal_backend not in {None, "local", "docker"}:
            raise ValueError("ROVA_TERMINAL_BACKEND must be 'local' or 'docker'")
        return cls(
            provider=source.get("ROVA_PROVIDER", "mock"),
            model=source.get("ROVA_MODEL", "mock"),
            base_url=source.get("ROVA_BASE_URL") or None,
            context_window=int(context_window) if context_window else None,
            compaction_reserve_tokens=int(reserve_tokens) if reserve_tokens else None,
            compaction_keep_recent_tokens=int(keep_recent_tokens) if keep_recent_tokens else None,
            artifact_root=Path(artifact_root).expanduser() if artifact_root else None,
            memory_root=Path(memory_root).expanduser() if memory_root else None,
            memory_provider=source.get("ROVA_MEMORY_PROVIDER") or None,
            memory_model=source.get("ROVA_MEMORY_MODEL") or None,
            memory_base_url=source.get("ROVA_MEMORY_BASE_URL") or None,
            memory_context_window=int(memory_context_window) if memory_context_window else None,
            memory_max_chars=int(memory_max_chars) if memory_max_chars else 6_000,
            experience_review_enabled=_parse_bool(experience_review_enabled, default=True),
            experience_review_tool_threshold=(
                int(experience_review_tool_threshold) if experience_review_tool_threshold else 10
            ),
            experience_review_task_threshold=(
                int(experience_review_task_threshold) if experience_review_task_threshold else 5
            ),
            data_dir=Path(data_dir).expanduser() if data_dir else None,
            terminal_backend=terminal_backend,
            docker_image=source.get("ROVA_DOCKER_IMAGE") or None,
        )

    def to_model(self) -> Model:
        return Model(
            provider=self.provider,
            model=self.model,
            base_url=self.base_url,
            context_window=self.context_window,
        )

    def to_memory_model(self) -> Model:
        return Model(
            provider=self.memory_provider or self.provider,
            model=self.memory_model or self.model,
            base_url=self.memory_base_url if self.memory_base_url is not None else self.base_url,
            context_window=(
                self.memory_context_window
                if self.memory_context_window is not None
                else self.context_window
            ),
        )

    def data_paths(self, explicit_data_dir: Path | None = None) -> RovaDataPaths:
        return RovaDataPaths.resolve(explicit_data_dir or self.data_dir)

    def to_compaction_policy(self) -> CompactionPolicy | None:
        if self.compaction_reserve_tokens is None and self.compaction_keep_recent_tokens is None:
            return None
        if self.compaction_reserve_tokens is None or self.compaction_keep_recent_tokens is None:
            raise ValueError("both compaction reserve and keep-recent settings are required")
        return CompactionPolicy(self.compaction_reserve_tokens, self.compaction_keep_recent_tokens)


def _parse_bool(value: str | None, *, default: bool) -> bool:
    if value is None or not value.strip():
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError("ROVA_EXPERIENCE_REVIEW_ENABLED must be a boolean")
