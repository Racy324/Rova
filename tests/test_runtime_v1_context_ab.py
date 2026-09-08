from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest


def test_dry_run_context_fixtures_materialize_immutable_large_input(tmp_path: Path) -> None:
    from evals.runtime_v1.fixtures import dry_run_context_fixtures, fresh_workspace

    cm01, cm02 = dry_run_context_fixtures()
    with fresh_workspace(cm01, tmp_path / "cm01") as workspace:
        assert (workspace / "reference.txt").stat().st_size >= 80 * 1024
        assert "ROVA_EVAL_TARGET_MAPPING=violet-47" in (workspace / "reference.txt").read_text(encoding="utf-8")
    with fresh_workspace(cm02, tmp_path / "cm02") as workspace:
        constraints = sorted((workspace / "constraints").glob("*.txt"))
        assert len(constraints) == 4
        assert sum(item.stat().st_size for item in constraints) >= 200 * 1024


def test_context_validator_uses_the_execution_environment_filesystem_root_for_sandbox_runs(tmp_path: Path) -> None:
    from evals.runtime_v1.context_ab import execution_workspace_root

    host_root = tmp_path / "host"
    sandbox_root = tmp_path / "sandbox"
    host_root.mkdir()
    sandbox_root.mkdir()
    runtime = SimpleNamespace(
        execution_environment=SimpleNamespace(
            filesystem=SimpleNamespace(resolve=lambda _path: sandbox_root)
        )
    )

    assert execution_workspace_root(runtime, host_root) == sandbox_root


@pytest.mark.asyncio
async def test_context_dry_run_uses_only_profile_seams_and_observes_externalization_and_compaction(
    tmp_path: Path,
) -> None:
    from evals.runtime_v1.context_ab import run_context_dry_run

    report = await run_context_dry_run(tmp_path / "results", use_development_provider=True)

    assert len(report.results) == 4
    by_key = {(item.case_id, item.profile): item for item in report.observations}
    assert by_key[("CM01_large_tool_output_repair", "base")].externalization_count == 0
    assert by_key[("CM01_large_tool_output_repair", "full")].externalization_count >= 1
    assert by_key[("CM02_long_history_followthrough", "base")].compaction_count == 0
    assert by_key[("CM02_long_history_followthrough", "full")].compaction_count >= 1
    assert all(result.task_success for result in report.results)
    # CM01 has read -> write -> final (3 calls) for each profile. CM02 has
    # four acknowledgements plus a write/final-response Tool loop (6 calls),
    # and Full adds one compaction summary request: 3 + 3 + 6 + 7.
    assert report.provider_request_count == 19
    assert not list((tmp_path / "results" / ".scratch").rglob("reference.txt"))
