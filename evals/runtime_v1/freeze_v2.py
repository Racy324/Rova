from __future__ import annotations

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from .context_cases import contract_for, prompt_sha256, validator_sha256
from .fixtures import dry_run_context_fixtures, fresh_workspace
from .tool_parallel import frozen_tool_scenarios


SUITE_VERSION = "runtime_v1_evaluation_v2"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _context_cases() -> list[dict[str, object]]:
    cases: list[dict[str, object]] = []
    repository_root = Path(__file__).resolve().parents[2]
    with TemporaryDirectory(prefix="rova-runtime-v1-freeze-v2-") as temporary:
        root = Path(temporary)
        for fixture in dry_run_context_fixtures():
            with fresh_workspace(fixture, root) as workspace:
                contract = contract_for(fixture.case_id)
                cases.append({
                    "case_id": fixture.case_id,
                    "fixture_source": fixture.source.relative_to(repository_root).as_posix(),
                    "fixture_sha256": fixture.sha256,
                    "prompt_sha256": prompt_sha256(fixture.case_id, workspace),
                    "validator_sha256": validator_sha256(fixture.case_id),
                    "allowed_change_paths": list(contract.allowed_change_paths),
                })
    return cases


def _authority_hashes(repository_root: Path) -> dict[str, str]:
    paths = (
        "evals/runtime_v1/context_ab.py",
        "evals/runtime_v1/context_cases.py",
        "evals/runtime_v1/fixtures.py",
        "evals/runtime_v1/runtime_factory.py",
        "evals/runtime_v1/tool_parallel.py",
        "evals/runtime_v1/fault_benchmark.py",
        "evals/runtime_v1/runner.py",
        "rova/app/runtime.py",
        "rova/ai/providers/openai_compatible.py",
    )
    return {relative: _sha256(repository_root / relative) for relative in paths}


def write_v2_freeze_manifest(
    path: Path,
    *,
    runtime_commit: str,
    frozen_at: str,
    provider: str,
    model: str,
    sandbox_image: str,
    sandbox_image_digest: str,
    provider_timeout_seconds: float = 60.0,
    provider_max_retries: int = 2,
    source_root: Path | None = None,
) -> Path:
    """Write a new, complete v2 suite contract without recording an eval-suite commit."""
    if len(runtime_commit) != 40:
        raise ValueError("runtime_commit must be a full 40-character SHA")
    repository_root = Path(source_root or Path(__file__).resolve().parents[2]).resolve()
    authorities = _authority_hashes(repository_root)
    manifest = {
        "schema_version": 2,
        "suite": "runtime-v1-evaluation",
        "suite_version": SUITE_VERSION,
        "frozen_at": frozen_at,
        "runtime_commit": runtime_commit,
        "context_cases": _context_cases(),
        "context_profiles": {
            "base": {"compaction": False, "tool_result_externalization": False, "overflow_recovery": False},
            "full": {
                "compaction": True,
                "tool_result_externalization": True,
                "overflow_recovery": True,
                "compaction_policy": {"context_window": 64_000, "reserve_tokens": 12_000, "keep_recent_tokens": 20_000},
            },
        },
        "shared_runtime_settings": {
            "max_turns": 6,
            "provider_max_retries": provider_max_retries,
            "permission_mode": "full",
            "approval_handler": "AlwaysApprove",
            "experience_review_enabled": False,
            "execution_environment": "sandbox",
            "terminal_backend": "docker",
            "workspace_policy": "fresh_eval_owned_sandbox_no_apply",
            "cleanup_policy": "discard_sandbox_and_delete_workspace_after_every_run",
            "keep_failed_workspace": False,
        },
        "model": {
            "provider": provider,
            "name": model,
            "context_window": 64_000,
            "temperature": None,
            "max_tokens": None,
            "provider_timeout_seconds": provider_timeout_seconds,
        },
        "sandbox": {"image": sandbox_image, "image_digest": sandbox_image_digest},
        "tool_parallelism": {
            "scenarios": [item.to_dict() for item in frozen_tool_scenarios()],
            "warmup_samples_per_scenario": 3,
            "measured_samples_per_scenario": 30,
            "provider_requests": 0,
            "authority_file_sha256": {"evals/runtime_v1/tool_parallel.py": authorities["evals/runtime_v1/tool_parallel.py"]},
        },
        "fault_injection": {
            "case_ids": [f"FI{index:02d}" for index in range(1, 13)],
            "repeats": 3,
            "provider_requests": 0,
            "authority_file_sha256": {
                "evals/runtime_v1/fault_benchmark.py": authorities["evals/runtime_v1/fault_benchmark.py"],
                "evals/runtime_v1/runner.py": authorities["evals/runtime_v1/runner.py"],
            },
        },
        "authority_file_sha256": authorities,
        "metric_definitions": {
            "context_ab": ["validator_success", "total_input_tokens", "average_input_tokens", "max_input_tokens", "output_tokens", "duration_ms", "provider_request_count", "tool_call_count", "externalization_count", "compaction_count", "overflow_recovery_count", "harness_failure", "termination_reason"],
            "tool_parallelism": ["sequential_p50_ms", "parallel_p50_ms", "sequential_p95_ms", "parallel_p95_ms", "speedup", "latency_reduction", "ordering_violations", "failure_isolation_violations", "sequential_fallback_violations"],
            "fault_injection": ["recoverable_fault_recovery_rate", "failure_policy_correctness", "duplicate_side_effect_count", "partial_commit_violations", "transparent_tool_retry_violations"],
        },
        "formal_runs": {"context_ab": 12, "tool_parallelism_batches": 360, "fault_injection": 36},
        "historical_run_disposition": [
            {"experiment_id": "dry-run-live-20260908*", "aggregation_status": "invalid", "reason": "Host/Sandbox validator path error."},
            {"experiment_id": "pre-freeze-calibration-20260908", "aggregation_status": "calibration_only", "reason": "Pre-freeze calibration only."},
        ],
        "supersedes": {
            "suite_version": "runtime_v1_context_64k_v1",
            "status": "pre_formal_superseded",
            "reason": "Context freeze was reproducible, but Tool/Fault authority contracts were incomplete; no Formal Run used v1.",
        },
        "formal_change_lock": True,
        "formal_change_lock_rule": "Formal results cannot change frozen fixtures, prompts, validators, profiles, metrics, repeats, Tool scenarios, or Fault scripts. An invalidating defect requires a new suite version and manifest.",
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
    return destination


def write_v1_supersession_record(path: Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "suite_version": "runtime_v1_context_64k_v1",
        "status": "pre_formal_superseded",
        "reason": "Context freeze was reproducible, but Tool/Fault authority contracts were incomplete; no Formal Run used v1.",
        "superseded_by": SUITE_VERSION,
    }
    with destination.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
    return destination
