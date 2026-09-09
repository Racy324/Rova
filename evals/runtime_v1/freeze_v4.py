"""Final Runtime V1 evaluation freeze with Formal execution authority."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .freeze_v2 import _context_cases


SUITE_VERSION = "runtime_v1_evaluation_v4"
V3_SUPERSESSION_REASON = (
    "Formal benchmark contract was complete, but the new Formal execution, persistence, "
    "completeness, aggregation, and simulation authority had not yet been frozen; "
    "no Formal Run used v3."
)

# These are the concrete files reached by the Formal runner's Context, Tool, or
# Fault paths, plus the code that validates the frozen contract.  This list is
# intentionally suite-specific rather than a generic dependency tracker.
FORMAL_AUTHORITY_FILES = (
    "evals/runtime_v1/formal.py",
    "evals/runtime_v1/context_ab.py",
    "evals/runtime_v1/context_cases.py",
    "evals/runtime_v1/fixtures.py",
    "evals/runtime_v1/runtime_factory.py",
    "evals/runtime_v1/tool_parallel.py",
    "evals/runtime_v1/fault_benchmark.py",
    "evals/runtime_v1/runner.py",
    "evals/runtime_v1/freeze_v4.py",
    "evals/runtime_v1/formal_preflight_v4.py",
    "rova/eval/models.py",
    "rova/eval/runner.py",
    "rova/trace/models.py",
    "rova/trace/store.py",
    "rova/ai/context.py",
    "rova/ai/events.py",
    "rova/ai/messages.py",
    "rova/ai/models.py",
    "rova/ai/stream.py",
    "rova/ai/providers/openai_compatible.py",
    "rova/agent_core/agent.py",
    "rova/agent_core/retry.py",
    "rova/agent_core/tools.py",
    "rova/agent_core/tool_output.py",
    "rova/agent_session/agent_session.py",
    "rova/agent_session/compaction.py",
    "rova/app/runtime.py",
    "rova/app/workspace/approval.py",
    "rova/app/workspace/environment.py",
    "rova/app/workspace/sandbox.py",
    "rova/app/workspace/sandbox_control.py",
    "rova/app/workspace/terminal.py",
    "rova/app/workspace/workspace.py",
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
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def authority_hashes(repository_root: Path) -> dict[str, str]:
    """Hash every concrete authority that changes Formal execution semantics."""
    return {relative: _sha256(repository_root / relative) for relative in FORMAL_AUTHORITY_FILES}


def _load_v3_manifest(path: Path) -> dict[str, object]:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if manifest.get("suite_version") != "runtime_v1_evaluation_v3":
        raise ValueError("v4 must be frozen from runtime_v1_evaluation_v3")
    return manifest


def _formal_counts(v3: dict[str, object]) -> dict[str, int]:
    tools = v3["tool_parallelism"]
    faults = v3["fault_injection"]
    return {
        "context_ab": int(v3["formal_runs"]["context_ab"]),
        "tool_parallelism_warmups": len(tools["scenarios"]) * int(tools["warmup_samples_per_scenario"]),
        "tool_parallelism_measurements": len(tools["scenarios"]) * int(tools["measured_samples_per_scenario"]),
        "fault_injection": len(faults["case_ids"]) * int(faults["repeats"]),
    }


def _fault_contract(v3: dict[str, object]) -> dict[str, object]:
    """Keep FI scripts/repeats untouched while freezing their actual observation shape."""
    fault = dict(v3["fault_injection"])
    observation_schema = list(fault["observation_schema"])
    if "duplicate_side_effect_count" not in observation_schema:
        observation_schema.append("duplicate_side_effect_count")
    fault["observation_schema"] = observation_schema
    return fault


def write_v4_freeze_manifest(
    path: Path,
    *,
    v3_manifest_path: Path,
    frozen_at: str,
    source_root: Path | None = None,
) -> Path:
    """Freeze v4 without retuning any benchmark parameter inherited from v3."""
    v3 = _load_v3_manifest(v3_manifest_path)
    root = Path(source_root or Path(__file__).resolve().parents[2]).resolve()
    if _context_cases() != v3["context_cases"]:
        raise ValueError("v3 Context fixture, prompt, validator, or boundary contract changed")

    manifest = {
        "schema_version": 4,
        "suite": v3["suite"],
        "suite_version": SUITE_VERSION,
        "frozen_at": frozen_at,
        "runtime_commit": v3["runtime_commit"],
        "context_cases": v3["context_cases"],
        "context_profiles": v3["context_profiles"],
        "shared_runtime_settings": v3["shared_runtime_settings"],
        "model": v3["model"],
        "sandbox": v3["sandbox"],
        "tool_parallelism": v3["tool_parallelism"],
        "fault_injection": _fault_contract(v3),
        "authority_file_sha256": authority_hashes(root),
        "metric_definitions": v3["metric_definitions"],
        "formal_runs": _formal_counts(v3),
        "formal_execution_contract": FORMAL_EXECUTION_CONTRACT,
        "metadata_contract": {
            "required": [
                "suite_version",
                "runtime_commit",
                "eval_suite_commit",
                "manifest_sha256",
                "authority_hashes",
                "execution_id",
                "execution_kind",
                "resolved_provider_fingerprint",
                "timestamps",
            ],
            "provider_fingerprint": "sanitized; credentials are prohibited",
        },
        "historical_run_disposition": [
            *v3["historical_run_disposition"],
            {
                "experiment_id": "runtime_v1_context_64k_v1",
                "aggregation_status": "pre_formal_superseded",
                "reason": "Tool and Fault freeze contracts were incomplete; no Formal Run used v1.",
            },
            {
                "experiment_id": "runtime_v1_evaluation_v3",
                "aggregation_status": "pre_formal_superseded",
                "reason": V3_SUPERSESSION_REASON,
            },
            {
                "experiment_id": "simulation-*",
                "aggregation_status": "simulation_only",
                "reason": "Simulation executions never enter Formal aggregation.",
            },
        ],
        "supersedes": {
            "suite_version": "runtime_v1_evaluation_v3",
            "status": "pre_formal_superseded",
            "reason": V3_SUPERSESSION_REASON,
        },
        "formal_change_lock": True,
        "formal_change_lock_rule": v3["formal_change_lock_rule"],
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
    return destination


def write_v3_supersession_record(path: Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(
            {
                "suite_version": "runtime_v1_evaluation_v3",
                "status": "pre_formal_superseded",
                "reason": V3_SUPERSESSION_REASON,
                "superseded_by": SUITE_VERSION,
            },
            handle,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
        handle.write("\n")
    return destination
