from __future__ import annotations

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Iterable

from .context_cases import contract_for, prompt_sha256, validator_sha256
from .fixtures import dry_run_context_fixtures, fresh_workspace


FREEZE_SCHEMA_VERSION = 1
SUITE_VERSION = "runtime_v1_context_64k_v1"
FORMAL_CONTEXT_REPEATS = 3
FORMAL_TOOL_WARMUPS = 3
FORMAL_TOOL_SAMPLES = 30
FORMAL_FAULT_REPEATS = 3


def _source_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _context_cases() -> list[dict[str, object]]:
    cases: list[dict[str, object]] = []
    repository_root = Path(__file__).resolve().parents[2]
    with TemporaryDirectory(prefix="rova-runtime-v1-freeze-") as temporary:
        root = Path(temporary)
        for fixture in dry_run_context_fixtures():
            with fresh_workspace(fixture, root) as workspace:
                contract = contract_for(fixture.case_id)
                cases.append(
                    {
                        "case_id": fixture.case_id,
                        "fixture_source": fixture.source.relative_to(repository_root).as_posix(),
                        "fixture_sha256": fixture.sha256,
                        "prompt_sha256": prompt_sha256(fixture.case_id, workspace),
                        "validator_sha256": validator_sha256(fixture.case_id),
                        "allowed_change_paths": list(contract.allowed_change_paths),
                    }
                )
    return cases


def _historical_run_disposition() -> list[dict[str, str]]:
    invalid_reason = (
        "Context correctness validator used the wrong Host/Sandbox path: it read the Host "
        "fixture copy rather than the Sandbox execution filesystem; results are not comparable."
    )
    return [
        {
            "experiment_id": experiment_id,
            "aggregation_status": "invalid",
            "reason": invalid_reason,
        }
        for experiment_id in (
            "dry-run-live-20260908",
            "dry-run-live-20260908-r2",
            "dry-run-live-20260908-r3",
            "dry-run-live-20260908-r4",
        )
    ] + [
        {
            "experiment_id": "pre-freeze-calibration-20260908",
            "aggregation_status": "calibration_only",
            "reason": "Used only to verify freeze criteria; it is not a formal benchmark repeat.",
        }
    ]


def _implementation_hashes(source_root: Path, extra_paths: Iterable[Path]) -> dict[str, str]:
    return {
        path.relative_to(source_root).as_posix(): _source_sha256(path)
        for path in sorted(extra_paths)
    }


def write_freeze_manifest(
    path: Path,
    *,
    git_commit_sha: str,
    frozen_at: str,
    provider: str,
    model: str,
    sandbox_image: str,
    sandbox_image_digest: str,
    provider_timeout_seconds: float = 60.0,
    provider_max_retries: int = 2,
    source_root: Path | None = None,
) -> Path:
    """Publish a new, write-once Runtime V1 formal-suite manifest."""
    if len(git_commit_sha) != 40:
        raise ValueError("git_commit_sha must be a full 40-character SHA")
    if not provider or not model or not sandbox_image or not sandbox_image_digest:
        raise ValueError("freeze manifest requires provider, model, sandbox image, and digest")
    root = Path(source_root or Path(__file__).resolve().parents[2]).resolve()
    implementation_files = (
        root / "evals" / "runtime_v1" / "context_cases.py",
        root / "evals" / "runtime_v1" / "context_ab.py",
        root / "evals" / "runtime_v1" / "fixtures.py",
        root / "evals" / "runtime_v1" / "runtime_factory.py",
        root / "evals" / "runtime_v1" / "tool_parallel.py",
        root / "evals" / "runtime_v1" / "fault_benchmark.py",
        root / "rova" / "app" / "runtime.py",
        root / "rova" / "ai" / "providers" / "openai_compatible.py",
    )
    manifest = {
        "schema_version": FREEZE_SCHEMA_VERSION,
        "suite": "runtime-v1-evaluation",
        "suite_version": SUITE_VERSION,
        "frozen_at": frozen_at,
        "git_commit_sha": git_commit_sha,
        "implementation_file_sha256": _implementation_hashes(root, implementation_files),
        "context_cases": _context_cases(),
        "context_profiles": {
            "base": {
                "compaction": False,
                "tool_result_externalization": False,
                "overflow_recovery": False,
            },
            "full": {
                "compaction": True,
                "tool_result_externalization": True,
                "overflow_recovery": True,
                "compaction_policy": {
                    "context_window": 64_000,
                    "reserve_tokens": 12_000,
                    "keep_recent_tokens": 20_000,
                },
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
        "formal_runs": {
            "context_ab": 2 * 2 * FORMAL_CONTEXT_REPEATS,
            "tool_parallelism_batches": 12 * FORMAL_TOOL_SAMPLES,
            "fault_injection": 12 * FORMAL_FAULT_REPEATS,
        },
        "tool_parallelism": {
            "scenario_count": 12,
            "warmup_samples_per_scenario": FORMAL_TOOL_WARMUPS,
            "measured_samples_per_scenario": FORMAL_TOOL_SAMPLES,
            "provider_requests": 0,
        },
        "fault_injection": {
            "case_ids": [f"FI{index:02d}" for index in range(1, 13)],
            "repeats": FORMAL_FAULT_REPEATS,
            "provider_requests": 0,
        },
        "metric_definitions": {
            "context_ab": [
                "success", "actual_input_tokens", "actual_output_tokens", "duration_ms",
                "overflow_recovery_count", "compaction_count", "externalization_count",
                "tool_call_count", "termination_reason",
            ],
            "tool_parallelism": [
                "p50_duration_ms", "p95_duration_ms", "speedup", "source_order_correct",
                "failure_isolation", "mutation_fallback",
            ],
            "fault_injection": [
                "recoverable_fault_recovery_rate", "failure_policy_correctness",
                "duplicate_side_effect_count", "partial_commit_violations",
                "transparent_tool_retry_violations",
            ],
        },
        "historical_run_disposition": _historical_run_disposition(),
        "formal_change_lock": True,
        "formal_change_lock_rule": (
            "After this manifest is frozen, formal results cannot be used to modify fixtures, "
            "prompts, validators, profiles, metrics, or repeat counts. An invalidating Runtime "
            "bug requires a new suite version and a new freeze manifest before any formal rerun."
        ),
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
    return destination
