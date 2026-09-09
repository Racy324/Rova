from __future__ import annotations

import hashlib
import json
from pathlib import Path


def _git_clean(arguments: tuple[str, ...]) -> str:
    if arguments == ("status", "--porcelain"):
        return ""
    if arguments == ("rev-parse", "HEAD"):
        return "e" * 40
    if arguments[:2] == ("merge-base", "--is-ancestor"):
        return ""
    raise AssertionError(f"unexpected git invocation: {arguments}")


def test_v6_only_adds_context_runtime_failure_containment_to_v5_contract(tmp_path: Path) -> None:
    from evals.runtime_v1.freeze_v6 import write_v6_freeze_manifest

    v5_path = Path("evals/runtime_v1/frozen/runtime_v1_evaluation_v5.json")
    manifest_path = write_v6_freeze_manifest(
        tmp_path / "runtime_v1_evaluation_v6.json",
        v5_manifest_path=v5_path,
        frozen_at="2026-09-09T00:00:00Z",
    )
    v5 = json.loads(v5_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["suite_version"] == "runtime_v1_evaluation_v6"
    for field in (
        "runtime_commit", "context_cases", "context_profiles", "shared_runtime_settings",
        "model", "sandbox", "tool_parallelism", "fault_injection", "metric_definitions", "formal_runs",
    ):
        assert manifest[field] == v5[field]
    assert manifest["formal_execution_contract"]["formal_record_schema_version"] == 2
    assert manifest["formal_execution_contract"]["context_runtime_failure_boundary"] == {
        "runtime_prompt_exception": "persist_failed_context_record_and_continue",
        "eval_infrastructure_phases": "fail_closed",
    }
    assert "evals/runtime_v1/freeze_v6.py" in manifest["authority_file_sha256"]
    assert "evals/runtime_v1/formal_preflight_v6.py" in manifest["authority_file_sha256"]
    assert any(
        item["experiment_id"] == "runtime_v1_evaluation_v5"
        and item["execution_id"] == "bacd33b52df14f0aaccf18a2d11f35e5"
        and item["aggregation_status"] == "formal_interrupted_diagnostic"
        for item in manifest["historical_run_disposition"]
    )


def test_v6_preflight_shares_file_byte_manifest_identity_with_formal_plan(tmp_path: Path) -> None:
    from evals.runtime_v1.formal import FormalExecutionPlan
    from evals.runtime_v1.formal_preflight_v6 import preflight_formal_suite
    from evals.runtime_v1.freeze_v6 import write_v6_freeze_manifest

    manifest_path = write_v6_freeze_manifest(
        tmp_path / "runtime_v1_evaluation_v6.json",
        v5_manifest_path=Path("evals/runtime_v1/frozen/runtime_v1_evaluation_v5.json"),
        frozen_at="2026-09-09T00:00:00Z",
    )
    preflight = preflight_formal_suite(
        manifest_path,
        git_runner=_git_clean,
        docker_image_digest=lambda _image: "sha256:d764629ce0ddd8c71fd371e9901efb324a95789d2315a47db7e4d27e78f1b0e9",
    )
    plan = FormalExecutionPlan.from_frozen_manifest(manifest_path, eval_suite_commit=preflight.eval_suite_commit)

    expected = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    assert preflight.manifest_sha256 == expected
    assert plan.manifest_sha256 == expected
