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


def test_v4_freeze_preserves_v3_benchmark_contract_and_freezes_formal_authority(
    tmp_path: Path,
) -> None:
    from evals.runtime_v1.freeze_v4 import write_v4_freeze_manifest

    v3_path = Path("evals/runtime_v1/frozen/runtime_v1_evaluation_v3.json")
    manifest_path = write_v4_freeze_manifest(
        tmp_path / "runtime_v1_evaluation_v4.json",
        v3_manifest_path=v3_path,
        frozen_at="2026-09-09T00:00:00Z",
    )
    v3 = json.loads(v3_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["suite_version"] == "runtime_v1_evaluation_v4"
    assert manifest["runtime_commit"] == "3d33e093e8e8b22e996a7d33ef0db69cdbb081f6"
    assert "eval_suite_commit" not in manifest
    assert manifest["context_cases"] == v3["context_cases"]
    assert manifest["context_profiles"] == v3["context_profiles"]
    assert manifest["tool_parallelism"]["scenarios"] == v3["tool_parallelism"]["scenarios"]
    assert manifest["fault_injection"]["case_ids"] == v3["fault_injection"]["case_ids"]
    assert manifest["fault_injection"]["repeats"] == 3
    assert "duplicate_side_effect_count" in manifest["fault_injection"]["observation_schema"]
    assert manifest["formal_runs"] == {
        "context_ab": 12,
        "tool_parallelism_warmups": 36,
        "tool_parallelism_measurements": 360,
        "fault_injection": 36,
    }
    assert "evals/runtime_v1/formal.py" in manifest["authority_file_sha256"]
    assert "evals/runtime_v1/formal_preflight_v4.py" in manifest["authority_file_sha256"]
    assert manifest["formal_execution_contract"] == {
        "formal_record_schema_version": 1,
        "logical_identity": [
            "experiment",
            "item_id",
            "profile",
            "repeat_index",
            "sample_index",
            "phase",
        ],
        "execution_states": ["created", "running", "completed", "interrupted", "invalid"],
        "interruption_policy": "interrupted records are diagnostic only; aggregation fails closed",
        "resume_policy": "unsupported; start a new execution_id from the frozen manifest",
        "raw_persistence_layout": [
            "execution.json",
            "frozen-manifest.json",
            "formal-metadata.json",
            "raw/context-results.jsonl",
            "raw/tool-samples.jsonl",
            "raw/fault-results.jsonl",
            "raw/run-index.jsonl",
            "traces/runs.jsonl",
        ],
        "formal_execution_started": False,
    }
    dispositions = {
        item["experiment_id"]: item["aggregation_status"]
        for item in manifest["historical_run_disposition"]
    }
    assert dispositions["runtime_v1_context_64k_v1"] == "pre_formal_superseded"
    assert dispositions["runtime_v1_evaluation_v2"] == "pre_formal_superseded"
    assert dispositions["runtime_v1_evaluation_v3"] == "pre_formal_superseded"
    assert dispositions["simulation-*"] == "simulation_only"


def test_v4_preflight_checks_execution_authority_and_returns_clean_head_metadata(
    tmp_path: Path,
) -> None:
    from evals.runtime_v1.formal_preflight_v4 import (
        FormalPreflightError,
        preflight_formal_suite,
    )
    from evals.runtime_v1.freeze_v4 import write_v4_freeze_manifest

    manifest_path = write_v4_freeze_manifest(
        tmp_path / "runtime_v1_evaluation_v4.json",
        v3_manifest_path=Path("evals/runtime_v1/frozen/runtime_v1_evaluation_v3.json"),
        frozen_at="2026-09-09T00:00:00Z",
    )
    result = preflight_formal_suite(
        manifest_path,
        git_runner=_git_clean,
        docker_image_digest=lambda _image: "sha256:d764629ce0ddd8c71fd371e9901efb324a95789d2315a47db7e4d27e78f1b0e9",
    )
    assert result.eval_suite_commit == "e" * 40
    assert result.formal_execution_started is False
    assert result.execution_counts == {
        "context_ab": 12,
        "tool_parallelism_warmups": 36,
        "tool_parallelism_measurements": 360,
        "fault_injection": 36,
    }

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["authority_file_sha256"]["evals/runtime_v1/formal.py"] = "0" * 64
    mismatched = tmp_path / "mismatched-v4.json"
    mismatched.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(FormalPreflightError, match="authority hash mismatch"):
        preflight_formal_suite(
            mismatched,
            git_runner=_git_clean,
            docker_image_digest=lambda _image: "sha256:d764629ce0ddd8c71fd371e9901efb324a95789d2315a47db7e4d27e78f1b0e9",
        )

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["fault_injection"]["observation_schema"].remove("duplicate_side_effect_count")
    incomplete_observation = tmp_path / "incomplete-observation-v4.json"
    incomplete_observation.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(FormalPreflightError, match="Fault observation schema mismatch"):
        preflight_formal_suite(
            incomplete_observation,
            git_runner=_git_clean,
            docker_image_digest=lambda _image: "sha256:d764629ce0ddd8c71fd371e9901efb324a95789d2315a47db7e4d27e78f1b0e9",
        )
