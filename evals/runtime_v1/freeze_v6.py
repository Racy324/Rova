"""Freeze Runtime V1 Evaluation v6 with Context runtime-failure containment."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .freeze_v2 import _context_cases
from .freeze_v5 import FORMAL_AUTHORITY_FILES as V5_FORMAL_AUTHORITY_FILES
from .frozen_manifest import load_frozen_manifest


SUITE_VERSION = "runtime_v1_evaluation_v6"
V5_INTERRUPTED_REASON = (
    "The v5 Formal execution bacd33b52df14f0aaccf18a2d11f35e5 was interrupted and remains diagnostic only: "
    "a Runtime Context failure escaped the Formal Context executor instead of becoming a failed Context record."
)

FORMAL_AUTHORITY_FILES = tuple(
    relative
    for relative in V5_FORMAL_AUTHORITY_FILES
    if relative not in {"evals/runtime_v1/freeze_v5.py", "evals/runtime_v1/formal_preflight_v5.py"}
) + (
    "evals/runtime_v1/freeze_v6.py",
    "evals/runtime_v1/formal_preflight_v6.py",
)

FORMAL_EXECUTION_CONTRACT = {
    "formal_record_schema_version": 2,
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
    "context_runtime_failure_boundary": {
        "runtime_prompt_exception": "persist_failed_context_record_and_continue",
        "eval_infrastructure_phases": "fail_closed",
    },
}


def authority_hashes(repository_root: Path) -> dict[str, str]:
    return {
        relative: hashlib.sha256((repository_root / relative).read_bytes()).hexdigest()
        for relative in FORMAL_AUTHORITY_FILES
    }


def _load_v5_manifest(path: Path) -> dict[str, object]:
    frozen = load_frozen_manifest(path)
    if frozen.document.get("suite_version") != "runtime_v1_evaluation_v5":
        raise ValueError("v6 must be frozen from runtime_v1_evaluation_v5")
    return frozen.document


def write_v6_freeze_manifest(
    path: Path,
    *,
    v5_manifest_path: Path,
    frozen_at: str,
    source_root: Path | None = None,
) -> Path:
    """Write v6 without changing any v5 benchmark parameter or metric."""
    v5 = _load_v5_manifest(v5_manifest_path)
    root = Path(source_root or Path(__file__).resolve().parents[2]).resolve()
    if _context_cases() != v5["context_cases"]:
        raise ValueError("v5 Context fixture, prompt, validator, or boundary contract changed")
    manifest = {
        "schema_version": 6,
        "suite": v5["suite"],
        "suite_version": SUITE_VERSION,
        "frozen_at": frozen_at,
        "runtime_commit": v5["runtime_commit"],
        "context_cases": v5["context_cases"],
        "context_profiles": v5["context_profiles"],
        "shared_runtime_settings": v5["shared_runtime_settings"],
        "model": v5["model"],
        "sandbox": v5["sandbox"],
        "tool_parallelism": v5["tool_parallelism"],
        "fault_injection": v5["fault_injection"],
        "authority_file_sha256": authority_hashes(root),
        "metric_definitions": v5["metric_definitions"],
        "formal_runs": v5["formal_runs"],
        "formal_execution_contract": FORMAL_EXECUTION_CONTRACT,
        "metadata_contract": v5["metadata_contract"],
        "historical_run_disposition": [
            *v5["historical_run_disposition"],
            {
                "experiment_id": "runtime_v1_evaluation_v5",
                "execution_id": "bacd33b52df14f0aaccf18a2d11f35e5",
                "aggregation_status": "formal_interrupted_diagnostic",
                "reason": V5_INTERRUPTED_REASON,
            },
        ],
        "supersedes": {
            "suite_version": "runtime_v1_evaluation_v5",
            "status": "formal_interrupted_diagnostic",
            "reason": V5_INTERRUPTED_REASON,
        },
        "formal_change_lock": True,
        "formal_change_lock_rule": v5["formal_change_lock_rule"],
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
    load_frozen_manifest(destination)
    return destination
