from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .freeze_v2 import _authority_hashes, _context_cases


SUITE_VERSION = "runtime_v1_evaluation_v3"

FAULT_OBSERVATION_SCHEMA = [
    "provider_attempt_count",
    "recovered",
    "terminated",
    "termination_reason",
    "compaction_count",
    "tool_execution_count",
    "side_effect_execution_count",
    "partial_tool_execution_count",
    "partial_commit_violations",
    "transparent_tool_retry_violations",
    "unexpected_retry_violations",
]

V2_SUPERSESSION_REASON = (
    "Context and Tool contracts were complete, but Fault formal observations did not persist "
    "actual runtime facts; no Formal Run used v2."
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_v2_manifest(path: Path) -> dict[str, object]:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if manifest.get("suite_version") != "runtime_v1_evaluation_v2":
        raise ValueError("v3 must be frozen from runtime_v1_evaluation_v2")
    return manifest


def write_v3_freeze_manifest(
    path: Path,
    *,
    v2_manifest_path: Path,
    frozen_at: str,
    source_root: Path | None = None,
) -> Path:
    """Freeze v3 by preserving v2 Context/Tool contracts and adding fault observations."""
    v2 = _load_v2_manifest(Path(v2_manifest_path))
    repository_root = Path(source_root or Path(__file__).resolve().parents[2]).resolve()
    authorities = _authority_hashes(repository_root)

    # V3 does not retune Context or Tool. Refuse to create a manifest if either
    # inherited contract no longer matches its frozen execution authority.
    for relative in (
        "evals/runtime_v1/context_ab.py",
        "evals/runtime_v1/context_cases.py",
        "evals/runtime_v1/fixtures.py",
        "evals/runtime_v1/runtime_factory.py",
        "evals/runtime_v1/tool_parallel.py",
    ):
        if v2["authority_file_sha256"][relative] != authorities[relative]:
            raise ValueError(f"v2 Context/Tool authority changed: {relative}")
    if _context_cases() != v2["context_cases"]:
        raise ValueError("v2 Context fixture contract changed")

    fault = dict(v2["fault_injection"])
    fault["authority_file_sha256"] = {
        "evals/runtime_v1/fault_benchmark.py": authorities["evals/runtime_v1/fault_benchmark.py"],
        "evals/runtime_v1/runner.py": authorities["evals/runtime_v1/runner.py"],
    }
    fault["observation_schema"] = list(FAULT_OBSERVATION_SCHEMA)
    fault["observation_authority"] = {
        "expected_policy": "ExpectedFaultPolicy",
        "actual_observation": "FaultRunObservation",
        "evaluation": "FaultRunResult",
    }

    manifest = {
        "schema_version": 3,
        "suite": v2["suite"],
        "suite_version": SUITE_VERSION,
        "frozen_at": frozen_at,
        "runtime_commit": v2["runtime_commit"],
        "context_cases": v2["context_cases"],
        "context_profiles": v2["context_profiles"],
        "shared_runtime_settings": v2["shared_runtime_settings"],
        "model": v2["model"],
        "sandbox": v2["sandbox"],
        "tool_parallelism": v2["tool_parallelism"],
        "fault_injection": fault,
        "authority_file_sha256": authorities,
        "metric_definitions": {
            **v2["metric_definitions"],
            "fault_injection": [
                "recoverable_fault_recovery_rate",
                "failure_policy_correctness",
                "provider_attempt_count",
                "termination_reason",
                "compaction_count",
                "tool_execution_count",
                "duplicate_side_effect_count",
                "partial_commit_violations",
                "transparent_tool_retry_violations",
                "unexpected_retry_violations",
            ],
        },
        "formal_runs": v2["formal_runs"],
        "historical_run_disposition": [
            *v2["historical_run_disposition"],
            {
                "experiment_id": "runtime_v1_evaluation_v2",
                "aggregation_status": "pre_formal_superseded",
                "reason": V2_SUPERSESSION_REASON,
            },
        ],
        "supersedes": {
            "suite_version": "runtime_v1_evaluation_v2",
            "status": "pre_formal_superseded",
            "reason": V2_SUPERSESSION_REASON,
        },
        "formal_change_lock": True,
        "formal_change_lock_rule": v2["formal_change_lock_rule"],
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
    return destination


def write_v2_supersession_record(path: Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(
            {
                "suite_version": "runtime_v1_evaluation_v2",
                "status": "pre_formal_superseded",
                "reason": V2_SUPERSESSION_REASON,
                "superseded_by": SUITE_VERSION,
            },
            handle,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
        handle.write("\n")
    return destination
