from __future__ import annotations

import json
from pathlib import Path

import pytest


def _git_clean(arguments: tuple[str, ...]) -> str:
    if arguments == ("status", "--porcelain"):
        return ""
    if arguments == ("rev-parse", "HEAD"):
        return "e" * 40
    if arguments[:2] == ("merge-base", "--is-ancestor"):
        return ""
    raise AssertionError(f"unexpected git invocation: {arguments}")


def test_v2_manifest_freezes_complete_tool_fault_authority_without_eval_suite_commit(tmp_path: Path) -> None:
    from evals.runtime_v1.freeze_v2 import write_v2_freeze_manifest

    manifest_path = write_v2_freeze_manifest(
        tmp_path / "runtime_v1_evaluation_v2.json",
        runtime_commit="a" * 40,
        frozen_at="2026-09-09T00:00:00Z",
        provider="openai_compatible",
        model="mimo-v2.5",
        sandbox_image="python:3.12-slim",
        sandbox_image_digest="sha256:" + "b" * 64,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["suite_version"] == "runtime_v1_evaluation_v2"
    assert manifest["runtime_commit"] == "a" * 40
    assert "eval_suite_commit" not in manifest
    assert len(manifest["tool_parallelism"]["scenarios"]) == 12
    assert manifest["tool_parallelism"]["scenarios"][0]["per_tool_latency_ms"] == [25]
    assert manifest["fault_injection"]["authority_file_sha256"].keys() >= {
        "evals/runtime_v1/fault_benchmark.py",
        "evals/runtime_v1/runner.py",
    }
    assert manifest["supersedes"]["suite_version"] == "runtime_v1_context_64k_v1"
    assert manifest["supersedes"]["status"] == "pre_formal_superseded"


def test_formal_preflight_writes_distinct_runtime_and_eval_suite_commits(tmp_path: Path) -> None:
    from evals.runtime_v1.formal_preflight import preflight_formal_suite, write_formal_metadata
    from evals.runtime_v1.freeze_v2 import write_v2_freeze_manifest

    manifest_path = write_v2_freeze_manifest(
        tmp_path / "runtime_v1_evaluation_v2.json",
        runtime_commit="a" * 40,
        frozen_at="2026-09-09T00:00:00Z",
        provider="openai_compatible",
        model="mimo-v2.5",
        sandbox_image="python:3.12-slim",
        sandbox_image_digest="sha256:" + "b" * 64,
    )
    preflight = preflight_formal_suite(
        manifest_path,
        git_runner=_git_clean,
        docker_image_digest=lambda _image: "sha256:" + "b" * 64,
    )
    metadata_path = write_formal_metadata(tmp_path / "formal-metadata.json", preflight)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert metadata["suite_version"] == "runtime_v1_evaluation_v2"
    assert metadata["runtime_commit"] == "a" * 40
    assert metadata["eval_suite_commit"] == "e" * 40
    assert metadata["formal_execution_started"] is False


def test_formal_preflight_rejects_dirty_worktree_and_authority_hash_mismatch(tmp_path: Path) -> None:
    from evals.runtime_v1.formal_preflight import FormalPreflightError, preflight_formal_suite
    from evals.runtime_v1.freeze_v2 import write_v2_freeze_manifest

    manifest_path = write_v2_freeze_manifest(
        tmp_path / "runtime_v1_evaluation_v2.json",
        runtime_commit="a" * 40,
        frozen_at="2026-09-09T00:00:00Z",
        provider="openai_compatible",
        model="mimo-v2.5",
        sandbox_image="python:3.12-slim",
        sandbox_image_digest="sha256:" + "b" * 64,
    )

    with pytest.raises(FormalPreflightError, match="working tree is not clean"):
        preflight_formal_suite(
            manifest_path,
            git_runner=lambda arguments: " M evals/runtime_v1/tool_parallel.py" if arguments == ("status", "--porcelain") else _git_clean(arguments),
            docker_image_digest=lambda _image: "sha256:" + "b" * 64,
        )

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["fault_injection"]["authority_file_sha256"]["evals/runtime_v1/runner.py"] = "0" * 64
    mismatched = tmp_path / "mismatched.json"
    mismatched.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(FormalPreflightError, match="authority hash mismatch"):
        preflight_formal_suite(
            mismatched,
            git_runner=_git_clean,
            docker_image_digest=lambda _image: "sha256:" + "b" * 64,
        )
