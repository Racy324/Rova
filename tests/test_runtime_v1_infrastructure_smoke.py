from __future__ import annotations

from pathlib import Path

import pytest


@pytest.mark.asyncio
async def test_infrastructure_smoke_persists_context_tool_and_fault_results_then_cleans_workspaces(
    tmp_path: Path,
) -> None:
    from evals.runtime_v1.runner import run_infrastructure_smoke
    from rova.eval.store import JsonlEvalStore

    report = await run_infrastructure_smoke(tmp_path / "results")

    assert report.context_provider_request_count == 8
    assert report.context_result_count == 4
    assert report.tool_result_count == 4
    assert report.fault_result_count == 12
    assert report.workspace_cleanup_verified is True
    assert all(result.task_success is True for result in report.results)
    assert len(JsonlEvalStore(tmp_path / "results" / "eval-results.jsonl").load_all()) == 20
    assert (tmp_path / "results" / "manifest.json").is_file()
    assert (tmp_path / "results" / "run-manifest.jsonl").is_file()
    assert (tmp_path / "results" / "traces" / "runs.jsonl").is_file()
