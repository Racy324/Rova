from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_freeze_manifest_has_immutable_context_profiles_cases_and_formal_suite_contract(tmp_path: Path) -> None:
    from evals.runtime_v1.freeze import write_freeze_manifest

    path = write_freeze_manifest(
        tmp_path / "runtime_v1_context_64k_v1.json",
        git_commit_sha="a" * 40,
        frozen_at="2026-09-08T00:00:00Z",
        provider="openai_compatible",
        model="mimo-v2.5",
        sandbox_image="python:3.12-slim",
        sandbox_image_digest="sha256:" + "b" * 64,
    )

    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert manifest["suite_version"] == "runtime_v1_context_64k_v1"
    assert manifest["git_commit_sha"] == "a" * 40
    assert manifest["frozen_at"] == "2026-09-08T00:00:00Z"
    assert manifest["formal_runs"] == {
        "context_ab": 12,
        "tool_parallelism_batches": 360,
        "fault_injection": 36,
    }
    assert manifest["context_profiles"]["base"] == {
        "compaction": False,
        "tool_result_externalization": False,
        "overflow_recovery": False,
    }
    assert manifest["context_profiles"]["full"] == {
        "compaction": True,
        "tool_result_externalization": True,
        "overflow_recovery": True,
        "compaction_policy": {
            "context_window": 64_000,
            "reserve_tokens": 12_000,
            "keep_recent_tokens": 20_000,
        },
    }
    assert {item["case_id"] for item in manifest["context_cases"]} == {
        "CM01_large_tool_output_repair",
        "CM02_long_history_followthrough",
    }
    assert all(len(item["fixture_sha256"]) == 64 for item in manifest["context_cases"])
    assert all(len(item["prompt_sha256"]) == 64 for item in manifest["context_cases"])
    assert all(len(item["validator_sha256"]) == 64 for item in manifest["context_cases"])
    assert manifest["tool_parallelism"]["scenario_count"] == 12
    assert manifest["tool_parallelism"]["warmup_samples_per_scenario"] == 3
    assert manifest["tool_parallelism"]["measured_samples_per_scenario"] == 30
    assert manifest["fault_injection"]["case_ids"] == [f"FI{index:02d}" for index in range(1, 13)]
    assert manifest["fault_injection"]["repeats"] == 3


def test_freeze_manifest_classifies_pre_freeze_runs_outside_formal_aggregation(tmp_path: Path) -> None:
    from evals.runtime_v1.freeze import write_freeze_manifest

    path = write_freeze_manifest(
        tmp_path / "runtime_v1_context_64k_v1.json",
        git_commit_sha="a" * 40,
        frozen_at="2026-09-08T00:00:00Z",
        provider="openai_compatible",
        model="mimo-v2.5",
        sandbox_image="python:3.12-slim",
        sandbox_image_digest="sha256:" + "b" * 64,
    )

    manifest = json.loads(path.read_text(encoding="utf-8"))
    history = {item["experiment_id"]: item for item in manifest["historical_run_disposition"]}
    assert history["dry-run-live-20260908-r4"]["aggregation_status"] == "invalid"
    assert "Host/Sandbox" in history["dry-run-live-20260908-r4"]["reason"]
    assert history["pre-freeze-calibration-20260908"]["aggregation_status"] == "calibration_only"
    assert manifest["formal_change_lock"] is True


def test_freeze_manifest_is_write_once(tmp_path: Path) -> None:
    from evals.runtime_v1.freeze import write_freeze_manifest

    arguments = {
        "git_commit_sha": "a" * 40,
        "frozen_at": "2026-09-08T00:00:00Z",
        "provider": "openai_compatible",
        "model": "mimo-v2.5",
        "sandbox_image": "python:3.12-slim",
        "sandbox_image_digest": "sha256:" + "b" * 64,
    }
    path = tmp_path / "runtime_v1_context_64k_v1.json"
    write_freeze_manifest(path, **arguments)

    with pytest.raises(FileExistsError):
        write_freeze_manifest(path, **arguments)
