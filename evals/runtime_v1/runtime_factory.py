from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rova.agent_core.types import StreamFn
from rova.agent_session.compaction import CompactionPolicy
from rova.ai.models import Model
from rova.app.runtime import RovaRuntime, build_rova_runtime
from rova.app.workspace.approval import AlwaysApprove


@dataclass(frozen=True)
class ContextManagementProfile:
    """The only Runtime V1 A/B differences permitted by the evaluation suite."""

    name: str
    compaction_policy: CompactionPolicy | None
    enable_tool_result_externalization: bool
    enable_context_overflow_recovery: bool

    @classmethod
    def base(cls) -> "ContextManagementProfile":
        return cls(
            name="base",
            compaction_policy=None,
            enable_tool_result_externalization=False,
            enable_context_overflow_recovery=False,
        )

    @classmethod
    def full(cls, compaction_policy: CompactionPolicy) -> "ContextManagementProfile":
        return cls(
            name="full",
            compaction_policy=compaction_policy,
            enable_tool_result_externalization=True,
            enable_context_overflow_recovery=True,
        )


def build_context_runtime(
    *,
    profile: ContextManagementProfile,
    model: Model,
    stream_fn: StreamFn,
    workspace_root: Path,
    state_root: Path,
    trace_root: Path | None = None,
    max_turns: int = 6,
    isolated_sandbox: bool = False,
    sandbox_root: Path | None = None,
    sandbox_image: str | None = None,
) -> RovaRuntime:
    """Build one isolated Context A/B runtime through the product assembly path."""
    state_root = Path(state_root)
    return build_rova_runtime(
        model=model,
        stream_fn=stream_fn,
        workspace_root=workspace_root,
        approval_handler=AlwaysApprove(),
        permission_mode="full",
        session_root=state_root / "sessions",
        artifact_root=state_root / "artifacts",
        trace_root=trace_root or state_root / "traces",
        memory_root=state_root / "memory",
        skill_root=state_root / "skills",
        experience_review_enabled=False,
        compaction_policy=profile.compaction_policy,
        enable_tool_result_externalization=profile.enable_tool_result_externalization,
        enable_context_overflow_recovery=profile.enable_context_overflow_recovery,
        max_turns=max_turns,
        terminal_backend="docker" if isolated_sandbox else "local",
        docker_image=sandbox_image,
        isolated_sandbox=isolated_sandbox,
        sandbox_root=sandbox_root,
    )
