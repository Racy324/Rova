"""Freeze Runtime V1 Evaluation v5 after repairing Formal metadata authority."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .freeze_v2 import _context_cases
from .freeze_v4 import FORMAL_AUTHORITY_FILES as V4_FORMAL_AUTHORITY_FILES
from .frozen_manifest import load_frozen_manifest


SUITE_VERSION = "runtime_v1_evaluation_v5"
V4_INVALIDATION_REASON = (
    "The only v4 Formal execution (1e88c3fcf70b41a18b32b6d6c087b0cb) was fail-closed invalidated: "
    "metadata used a canonical JSON manifest hash rather than the frozen file-byte hash, "
    "and Context duration instrumentation recorded evaluator time rather than Runtime wall-clock."
)

FORMAL_AUTHORITY_FILES = tuple(
    relative
    for relative in V4_FORMAL_AUTHORITY_FILES
    if relative not in {"evals/runtime_v1/freeze_v4.py", "evals/runtime_v1/formal_preflight_v4.py"}
) + (
    "evals/runtime_v1/frozen_manifest.py",
    "evals/runtime_v1/freeze_v5.py",
    "evals/runtime_v1/formal_preflight_v5.py",
)

FORMAL_EXECUTION_CONTRACT = {
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
    "manifest_sha256_semantics": "sha256(frozen manifest raw bytes)",
}


def authority_hashes(repository_root: Path) -> dict[str, str]:
    return {
        relative: hashlib.sha256((repository_root / relative).read_bytes()).hexdigest()
        for relative in FORMAL_AUTHORITY_FILES
    }


def _load_v4_manifest(path: Path) -> dict[str, object]:
    frozen = load_frozen_manifest(path)
    if frozen.document.get("suite_version") != "runtime_v1_evaluation_v4":
        raise ValueError("v5 must be frozen from runtime_v1_evaluation_v4")
    return frozen.document


def write_v5_freeze_manifest(
    path: Path,
    *,
    v4_manifest_path: Path,
    frozen_at: str,
    source_root: Path | None = None,
) -> Path:
    """Write v5 without changing any frozen benchmark task/profile/metric parameter."""
    v4 = _load_v4_manifest(v4_manifest_path)
    root = Path(source_root or Path(__file__).resolve().parents[2]).resolve()
    if _context_cases() != v4["context_cases"]:
        raise ValueError("v4 Context fixture, prompt, validator, or boundary contract changed")
    manifest = {
        "schema_version": 5,
        "suite": v4["suite"],
        "suite_version": SUITE_VERSION,
        "frozen_at": frozen_at,
        "runtime_commit": v4["runtime_commit"],
        "context_cases": v4["context_cases"],
        "context_profiles": v4["context_profiles"],
        "shared_runtime_settings": v4["shared_runtime_settings"],
        "model": v4["model"],
        "sandbox": v4["sandbox"],
        "tool_parallelism": v4["tool_parallelism"],
        "fault_injection": v4["fault_injection"],
        "authority_file_sha256": authority_hashes(root),
        "metric_definitions": v4["metric_definitions"],
        "formal_runs": v4["formal_runs"],
        "formal_execution_contract": FORMAL_EXECUTION_CONTRACT,
        "metadata_contract": v4["metadata_contract"],
        "historical_run_disposition": [
            *v4["historical_run_disposition"],
            {
                "experiment_id": "runtime_v1_evaluation_v4",
                "execution_id": "1e88c3fcf70b41a18b32b6d6c087b0cb",
                "aggregation_status": "formal_invalidated",
                "reason": V4_INVALIDATION_REASON,
            },
        ],
        "supersedes": {
            "suite_version": "runtime_v1_evaluation_v4",
            "status": "formal_invalidated",
            "reason": V4_INVALIDATION_REASON,
        },
        "formal_change_lock": True,
        "formal_change_lock_rule": v4["formal_change_lock_rule"],
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
    # Read through the shared byte-level parser before returning the freeze.
    load_frozen_manifest(destination)
    return destination


def write_v4_invalidation_record(path: Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "suite_version": "runtime_v1_evaluation_v4",
        "execution_id": "1e88c3fcf70b41a18b32b6d6c087b0cb",
        "aggregation_status": "formal_invalidated",
        "reason": V4_INVALIDATION_REASON,
        "must_not_enter_formal_aggregation": True,
    }
    with destination.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
    return destination
