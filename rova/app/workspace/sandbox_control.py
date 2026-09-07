"""User-control-plane operations for one selected isolated Sandbox.

This module deliberately contains no AgentTool.  Creating, diffing, applying,
discarding and repairing a Sandbox are user initiated runtime operations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .sandbox import (
    ApplyReport,
    ChangedSet,
    DiscardReport,
    SandboxApplyService,
    SandboxDiffService,
    SandboxMetadata,
    SandboxState,
    SandboxStore,
)
from .workspace import Workspace


@dataclass(frozen=True)
class SandboxStatus:
    """Safe-to-render Sandbox status; it intentionally contains no paths."""

    environment_kind: str
    sandbox_state: SandboxState
    sandbox_id: str
    changed_path_count: int | None
    apply_recovery_required: bool
    host_isolation_active: bool
    container_recreated_on_resume: bool


class SandboxControl:
    """Small application service shared by the public CLI and TUI."""

    def __init__(
        self,
        store: SandboxStore,
        host_workspace: Workspace,
        session_id: str,
        sandbox_id: str,
        *,
        container_recreated_on_resume: bool = False,
    ) -> None:
        self._store = store
        self._host_workspace = host_workspace
        self._session_id = session_id
        self._sandbox_id = sandbox_id
        self._container_recreated_on_resume = container_recreated_on_resume
        self._diff = SandboxDiffService(store)
        self._apply = SandboxApplyService(store)

    @property
    def sandbox_id(self) -> str:
        return self._sandbox_id

    def metadata(self) -> SandboxMetadata:
        return self._store.load_for_execution(self._sandbox_id)

    def status(self) -> SandboxStatus:
        metadata = self.metadata()
        changed_path_count: int | None = None
        if metadata.state is SandboxState.READY:
            changed_path_count = len(self._diff.compute_changed_set(metadata.sandbox_id).changes)
        return SandboxStatus(
            environment_kind="sandbox",
            sandbox_state=metadata.state,
            sandbox_id=metadata.sandbox_id[:8],
            changed_path_count=changed_path_count,
            apply_recovery_required=metadata.state is SandboxState.APPLYING or metadata.active_apply_id is not None,
            host_isolation_active=metadata.state is SandboxState.READY,
            container_recreated_on_resume=self._container_recreated_on_resume,
        )

    def diff(self) -> ChangedSet:
        return self._diff.compute_changed_set(self._sandbox_id)

    def apply_plan(self):
        return self._apply.build_plan(self._sandbox_id)

    def create_new(self) -> SandboxMetadata:
        return self._store.create_new_for_terminal_session(self._host_workspace, self._session_id)

    def apply(self, *, confirm: Callable) -> ApplyReport:
        return self._apply.apply(self._sandbox_id, confirm=confirm)

    def discard(self, *, confirm: Callable) -> DiscardReport:
        return self._store.discard(self._sandbox_id, confirm=confirm)

    def restore_preimages(self, operation_id: str, *, confirm: Callable) -> ApplyReport:
        return self._apply.restore_preimages(self._sandbox_id, operation_id, confirm=confirm)
