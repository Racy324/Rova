from __future__ import annotations

import pytest

from rova.ai.mock import MockProvider
from rova.ai.models import Model
from rova.agent_session.compaction import CompactionPolicy
from rova.app.runtime import build_rova_runtime


@pytest.mark.asyncio
async def test_context_profiles_control_only_the_three_frozen_context_seams(tmp_path) -> None:
    from evals.runtime_v1.runtime_factory import (
        ContextManagementProfile,
        build_context_runtime,
    )

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    model = Model("runtime-v1-smoke", context_window=4_000)
    policy = CompactionPolicy(reserve_tokens=1_000, keep_recent_tokens=500)

    base = build_context_runtime(
        profile=ContextManagementProfile.base(),
        model=model,
        stream_fn=MockProvider(),
        workspace_root=workspace,
        state_root=tmp_path / "base-state",
    )
    full = build_context_runtime(
        profile=ContextManagementProfile.full(policy),
        model=model,
        stream_fn=MockProvider(),
        workspace_root=workspace,
        state_root=tmp_path / "full-state",
    )
    try:
        assert base.agent.model == full.agent.model == model
        assert base.agent.max_turns == full.agent.max_turns
        assert base.agent.tool_execution_mode is full.agent.tool_execution_mode
        assert base.session._compaction_policy is None
        assert full.session._compaction_policy == policy
        assert base.agent.tool_runtime._tool_output_processor is None
        assert full.agent.tool_runtime._tool_output_processor is not None
        assert base.agent._context_overflow_recovery is None
        assert full.agent._context_overflow_recovery is not None
    finally:
        await base.close()
        await full.close()


@pytest.mark.asyncio
async def test_eval_assembly_seam_keeps_normal_runtime_context_behavior_enabled_by_default(tmp_path) -> None:
    runtime = build_rova_runtime(
        model=Model("runtime-default", context_window=4_000),
        stream_fn=MockProvider(),
        session_root=tmp_path / "sessions",
        artifact_root=tmp_path / "artifacts",
        compaction_policy=CompactionPolicy(reserve_tokens=1_000, keep_recent_tokens=500),
        experience_review_enabled=False,
    )
    try:
        assert runtime.agent.tool_runtime._tool_output_processor is not None
        assert runtime.agent._context_overflow_recovery is not None
        assert runtime.session._compaction_policy == CompactionPolicy(
            reserve_tokens=1_000,
            keep_recent_tokens=500,
        )
    finally:
        await runtime.close()
