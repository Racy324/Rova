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


def test_v5_freeze_preserves_v4_benchmark_contract_and_marks_v4_invalidated(
    tmp_path: Path,
) -> None:
    from evals.runtime_v1.freeze_v5 import write_v4_invalidation_record, write_v5_freeze_manifest

    v4_path = Path("evals/runtime_v1/frozen/runtime_v1_evaluation_v4.json")
    manifest_path = write_v5_freeze_manifest(
        tmp_path / "runtime_v1_evaluation_v5.json",
        v4_manifest_path=v4_path,
        frozen_at="2026-09-09T00:00:00Z",
    )
    invalidation_path = write_v4_invalidation_record(tmp_path / "runtime_v1_evaluation_v4.formal-invalidated.json")
    v4 = json.loads(v4_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    invalidation = json.loads(invalidation_path.read_text(encoding="utf-8"))

    assert manifest["suite_version"] == "runtime_v1_evaluation_v5"
    assert manifest["context_cases"] == v4["context_cases"]
    assert manifest["context_profiles"] == v4["context_profiles"]
    assert manifest["tool_parallelism"] == v4["tool_parallelism"]
    assert manifest["fault_injection"] == v4["fault_injection"]
    assert manifest["formal_runs"] == v4["formal_runs"]
    assert manifest["formal_execution_contract"]["manifest_sha256_semantics"] == "sha256(frozen manifest raw bytes)"
    assert "evals/runtime_v1/frozen_manifest.py" in manifest["authority_file_sha256"]
    assert any(
        item["experiment_id"] == "runtime_v1_evaluation_v4"
        and item["aggregation_status"] == "formal_invalidated"
        for item in manifest["historical_run_disposition"]
    )
    assert invalidation["execution_id"] == "1e88c3fcf70b41a18b32b6d6c087b0cb"
    assert invalidation["aggregation_status"] == "formal_invalidated"


def test_v5_preflight_and_formal_plan_share_manifest_file_bytes_identity(tmp_path: Path) -> None:
    from evals.runtime_v1.formal import FormalExecutionPlan
    from evals.runtime_v1.formal_preflight_v5 import preflight_formal_suite
    from evals.runtime_v1.freeze_v5 import write_v5_freeze_manifest

    manifest_path = write_v5_freeze_manifest(
        tmp_path / "runtime_v1_evaluation_v5.json",
        v4_manifest_path=Path("evals/runtime_v1/frozen/runtime_v1_evaluation_v4.json"),
        frozen_at="2026-09-09T00:00:00Z",
    )
    preflight = preflight_formal_suite(
        manifest_path,
        git_runner=_git_clean,
        docker_image_digest=lambda _image: "sha256:d764629ce0ddd8c71fd371e9901efb324a95789d2315a47db7e4d27e78f1b0e9",
    )
    plan = FormalExecutionPlan.from_frozen_manifest(
        manifest_path,
        eval_suite_commit=preflight.eval_suite_commit,
    )
    expected = hashlib.sha256(manifest_path.read_bytes()).hexdigest()

    assert preflight.manifest_sha256 == expected
    assert plan.manifest_sha256 == expected
