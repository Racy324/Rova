"""Manifest-driven Runtime V1 formal execution infrastructure.

This module deliberately owns evaluation orchestration only.  It never changes
Rova Runtime behavior and a simulation is written below the separate
``simulations`` root by its caller.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import shutil
import statistics
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

from rova.agent_session.compaction import CompactionPolicy
from rova.ai.models import Model
from rova.eval import EvalCase, EvalExecution, EvalRunner
from rova.trace import JsonlTraceStore

from .context_ab import _ContextValidator, _DRY_MODEL, _run_case
from .fixtures import dry_run_context_fixtures
from .fault_benchmark import run_fault_case
from .frozen_manifest import load_frozen_manifest, manifest_sha256
from .runtime_factory import ContextManagementProfile
from .tool_parallel import FrozenToolScenario, run_tool_parallel_scenarios


class FormalExecutionKind(Enum):
    FORMAL = "formal"
    SIMULATION = "simulation"


class FormalExecutionState(Enum):
    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    INTERRUPTED = "interrupted"
    INVALID = "invalid"


class FormalStoreError(RuntimeError):
    pass


class FormalCompletenessError(RuntimeError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _append_jsonl(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    raw = path.read_text(encoding="utf-8")
    if raw and not raw.endswith("\n"):
        raise FormalStoreError(f"truncated JSONL: {path}")
    return [json.loads(line) for line in raw.splitlines() if line]


def _sanitized_provider_fingerprint(model: Model) -> dict[str, object]:
    """Record reproducibility-relevant model settings without credentials."""
    value: dict[str, object] = {
        "provider": model.provider,
        "model": model.model,
        "context_window": model.context_window,
        "temperature": model.temperature,
        "max_tokens": model.max_tokens,
        "provider_timeout_seconds": model.provider_timeout,
    }
    if model.base_url:
        value["endpoint_sha256"] = hashlib.sha256(model.base_url.encode("utf-8")).hexdigest()
    return value


def _contains_sensitive_key(value: object) -> bool:
    sensitive = {"api", "api_key", "authorization", "cookie", "password", "secret", "token"}
    if isinstance(value, dict):
        return any(str(key).lower() in sensitive or _contains_sensitive_key(item) for key, item in value.items())
    if isinstance(value, list):
        return any(_contains_sensitive_key(item) for item in value)
    return False


@dataclass(frozen=True)
class FormalExecutionPlan:
    manifest: dict[str, object]
    manifest_bytes: bytes
    manifest_sha256: str
    runtime_commit: str
    eval_suite_commit: str
    execution_kind: FormalExecutionKind
    execution_id: str
    context_repeats: int
    tool_warmups: int
    tool_measurements: int
    fault_repeats: int
    resolved_provider_fingerprint: dict[str, object]
    sandboxed_context: bool

    @classmethod
    def from_frozen_manifest(
        cls,
        path: Path,
        *,
        eval_suite_commit: str,
        execution_id: str | None = None,
    ) -> "FormalExecutionPlan":
        frozen = load_frozen_manifest(path)
        return cls.from_manifest(
            frozen.document,
            manifest_bytes=frozen.raw_bytes,
            eval_suite_commit=eval_suite_commit,
            execution_id=execution_id,
        )

    @classmethod
    def from_manifest(
        cls,
        manifest: dict[str, object],
        *,
        manifest_bytes: bytes,
        eval_suite_commit: str,
        execution_id: str | None = None,
    ) -> "FormalExecutionPlan":
        if manifest.get("suite_version") is None or len(str(manifest.get("runtime_commit", ""))) != 40:
            raise ValueError("invalid frozen formal manifest")
        cases = manifest["context_cases"]
        profiles = manifest["context_profiles"]
        if not isinstance(cases, list) or not isinstance(profiles, dict) or set(profiles) != {"base", "full"}:
            raise ValueError("manifest Context contract is incomplete")
        context_total = int(manifest["formal_runs"]["context_ab"])
        divisor = len(cases) * len(profiles)
        if divisor == 0 or context_total % divisor:
            raise ValueError("Context formal run count is not divisible by its frozen matrix")
        tool = manifest["tool_parallelism"]
        fault = manifest["fault_injection"]
        provider = manifest["model"]
        fingerprint = {
            "provider": provider["provider"], "model": provider["name"],
            "context_window": provider["context_window"], "temperature": provider["temperature"],
            "max_tokens": provider["max_tokens"], "provider_timeout_seconds": provider["provider_timeout_seconds"],
        }
        return cls(
            manifest=dict(manifest), manifest_bytes=manifest_bytes,
            manifest_sha256=manifest_sha256(manifest_bytes), runtime_commit=str(manifest["runtime_commit"]),
            eval_suite_commit=eval_suite_commit, execution_kind=FormalExecutionKind.FORMAL,
            execution_id=execution_id or uuid4().hex, context_repeats=context_total // divisor,
            tool_warmups=int(tool["warmup_samples_per_scenario"]), tool_measurements=int(tool["measured_samples_per_scenario"]),
            fault_repeats=int(fault["repeats"]), resolved_provider_fingerprint=fingerprint, sandboxed_context=False,
        )

    def for_simulation(self, *, sandboxed_context: bool = False) -> "FormalExecutionPlan":
        return replace(self, execution_kind=FormalExecutionKind.SIMULATION, execution_id=uuid4().hex,
                       context_repeats=1, tool_warmups=1, tool_measurements=2, fault_repeats=1,
                       resolved_provider_fingerprint={"provider": "deterministic_local", "model": "runtime-v1-development"},
                       sandboxed_context=sandboxed_context)

    @property
    def suite_version(self) -> str:
        return str(self.manifest["suite_version"])

    @property
    def authority_hashes(self) -> dict[str, str]:
        return dict(self.manifest["authority_file_sha256"])


@dataclass(frozen=True)
class FormalRecord:
    execution_id: str
    execution_kind: str
    suite_version: str
    runtime_commit: str
    eval_suite_commit: str
    manifest_sha256: str
    authority_hashes: dict[str, str]
    experiment: str
    item_id: str
    profile: str
    repeat_index: int
    sample_index: int
    phase: str
    started_at: str
    ended_at: str
    duration_ms: float
    trace_run_id: str | None
    payload: dict[str, object]

    @classmethod
    def from_payload(cls, plan: FormalExecutionPlan, *, experiment: str, item_id: str, profile: str,
                     repeat_index: int, sample_index: int, phase: str, payload: dict[str, object],
                     trace_run_id: str | None, started_at: str | None = None, duration_ms: float = 0.0) -> "FormalRecord":
        start = started_at or _now()
        return cls(plan.execution_id, plan.execution_kind.value, plan.suite_version, plan.runtime_commit,
                   plan.eval_suite_commit, plan.manifest_sha256, plan.authority_hashes, experiment, item_id,
                   profile, repeat_index, sample_index, phase, start, _now(), duration_ms, trace_run_id, payload)

    @property
    def logical_identity(self) -> tuple[str, str, str, int, int, str]:
        return self.experiment, self.item_id, self.profile, self.repeat_index, self.sample_index, self.phase

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "FormalRecord":
        return cls(**value)  # type: ignore[arg-type]


class FormalExecutionStore:
    def __init__(self, root: Path, plan: FormalExecutionPlan) -> None:
        self.root, self.plan = Path(root), plan
        self.raw_root = self.root / "raw"
        self.trace_store = JsonlTraceStore(self.root / "traces" / "runs.jsonl")

    @classmethod
    def create(cls, root: Path, plan: FormalExecutionPlan) -> "FormalExecutionStore":
        store = cls(root, plan)
        if store.root.exists():
            raise FormalStoreError(f"formal execution root already exists: {store.root}")
        if _contains_sensitive_key(plan.resolved_provider_fingerprint):
            raise FormalStoreError("provider fingerprint must not contain credentials")
        store.root.mkdir(parents=True)
        _write_bytes(store.root / "frozen-manifest.json", plan.manifest_bytes)
        _write_json(store.root / "formal-metadata.json", {
            "suite_version": plan.suite_version, "runtime_commit": plan.runtime_commit,
            "eval_suite_commit": plan.eval_suite_commit, "manifest_sha256": plan.manifest_sha256,
            "execution_id": plan.execution_id, "execution_kind": plan.execution_kind.value,
            "resolved_provider_fingerprint": plan.resolved_provider_fingerprint,
            "sandboxed_context": plan.sandboxed_context,
            "formal_execution_started": False,
        })
        store._write_state(FormalExecutionState.CREATED)
        return store

    def _write_state(self, state: FormalExecutionState, *, reason: str | None = None) -> None:
        _write_json(self.root / "execution.json", {"execution_id": self.plan.execution_id,
                    "execution_kind": self.plan.execution_kind.value, "state": state.value, "updated_at": _now(), "reason": reason})

    def mark_running(self) -> None:
        self._write_state(FormalExecutionState.RUNNING)
        metadata = self.execution_metadata()
        metadata["formal_execution_started"] = self.plan.execution_kind is FormalExecutionKind.FORMAL
        _write_json(self.root / "formal-metadata.json", metadata)

    def mark_completed(self) -> None:
        self._write_state(FormalExecutionState.COMPLETED)

    def mark_interrupted(self, reason: str) -> None:
        self._write_state(FormalExecutionState.INTERRUPTED, reason=reason)

    def execution_metadata(self) -> dict[str, object]:
        return json.loads((self.root / "formal-metadata.json").read_text(encoding="utf-8"))

    def state(self) -> FormalExecutionState:
        return FormalExecutionState(json.loads((self.root / "execution.json").read_text(encoding="utf-8"))["state"])

    def _record_path(self, experiment: str) -> Path:
        return self.raw_root / {"context_ab": "context-results.jsonl", "tool_parallelism": "tool-samples.jsonl", "fault_injection": "fault-results.jsonl"}[experiment]

    def records(self, experiment: str | None = None) -> list[FormalRecord]:
        paths = [self._record_path(experiment)] if experiment else [self._record_path(item) for item in ("context_ab", "tool_parallelism", "fault_injection")]
        return [FormalRecord.from_dict(item) for path in paths for item in _read_jsonl(path)]

    def append(self, record: FormalRecord) -> None:
        if record.execution_id != self.plan.execution_id:
            raise FormalStoreError("record belongs to another execution")
        if record.logical_identity in {item.logical_identity for item in self.records()}:
            raise FormalStoreError("duplicate logical identity")
        _append_jsonl(self._record_path(record.experiment), record.to_dict())
        _append_jsonl(self.raw_root / "run-index.jsonl", {
            "execution_id": record.execution_id,
            "execution_kind": record.execution_kind,
            "experiment": record.experiment,
            "logical_identity": list(record.logical_identity),
            "trace_run_id": record.trace_run_id,
        })

    def append_trace(self, trace) -> None:
        self.trace_store.append(trace)


class CompletenessValidator:
    def validate(self, store: FormalExecutionStore, *, allow_running: bool = False) -> dict[str, int]:
        state = store.state()
        if state is not FormalExecutionState.COMPLETED and not (allow_running and state is FormalExecutionState.RUNNING):
            raise FormalCompletenessError(f"execution is not complete: {state.value}")
        if manifest_sha256((store.root / "frozen-manifest.json").read_bytes()) != store.plan.manifest_sha256:
            raise FormalCompletenessError("persisted frozen manifest does not match the execution plan")
        metadata = store.execution_metadata()
        for key, expected in {
            "execution_id": store.plan.execution_id,
            "execution_kind": store.plan.execution_kind.value,
            "suite_version": store.plan.suite_version,
            "runtime_commit": store.plan.runtime_commit,
            "eval_suite_commit": store.plan.eval_suite_commit,
            "manifest_sha256": store.plan.manifest_sha256,
        }.items():
            if metadata.get(key) != expected:
                raise FormalCompletenessError("execution metadata does not match the frozen plan")
        records = store.records()
        identities = [item.logical_identity for item in records]
        if len(identities) != len(set(identities)):
            raise FormalCompletenessError("duplicate logical identity")
        for item in records:
            if (item.execution_id != store.plan.execution_id or item.execution_kind != store.plan.execution_kind.value
                    or item.suite_version != store.plan.suite_version
                    or item.runtime_commit != store.plan.runtime_commit or item.eval_suite_commit != store.plan.eval_suite_commit
                    or item.manifest_sha256 != store.plan.manifest_sha256 or item.authority_hashes != store.plan.authority_hashes):
                raise FormalCompletenessError("record metadata does not match frozen execution")
        context = store.records("context_ab")
        tool = store.records("tool_parallelism")
        fault = store.records("fault_injection")
        expected_context = len(store.plan.manifest["context_cases"]) * len(store.plan.manifest["context_profiles"]) * store.plan.context_repeats
        expected_warmups = len(store.plan.manifest["tool_parallelism"]["scenarios"]) * store.plan.tool_warmups
        expected_measurements = len(store.plan.manifest["tool_parallelism"]["scenarios"]) * store.plan.tool_measurements
        expected_fault = len(store.plan.manifest["fault_injection"]["case_ids"]) * store.plan.fault_repeats
        counts = {"context": len(context), "tool_measurements": sum(item.phase == "measurement" for item in tool),
                  "tool_warmups": sum(item.phase == "warmup" for item in tool), "fault": len(fault)}
        if counts != {"context": expected_context, "tool_measurements": expected_measurements, "tool_warmups": expected_warmups, "fault": expected_fault}:
            raise FormalCompletenessError(f"incomplete formal execution: {counts}")
        self._validate_expected_identities(store, context, tool, fault)
        self._validate_payloads(store, context, tool, fault)
        trace_ids = {item.run_id for item in store.trace_store.load_all()}
        if any(item.trace_run_id is not None and item.trace_run_id not in trace_ids for item in records):
            raise FormalCompletenessError("missing trace reference")
        for item in context:
            run_ids = item.payload.get("trace_run_ids")
            if not isinstance(run_ids, list) or not run_ids or any(run_id not in trace_ids for run_id in run_ids):
                raise FormalCompletenessError("Context result has incomplete trace references")
        index = _read_jsonl(store.raw_root / "run-index.jsonl")
        if {tuple(item.get("logical_identity", ())) for item in index} != set(identities):
            raise FormalCompletenessError("run index does not cover exactly the persisted raw records")
        return counts

    @staticmethod
    def _validate_expected_identities(store: FormalExecutionStore, context: list[FormalRecord], tool: list[FormalRecord], fault: list[FormalRecord]) -> None:
        expected_context = {
            ("context_ab", str(case["case_id"]), profile, repeat, 1, "run")
            for case in store.plan.manifest["context_cases"]
            for profile in store.plan.manifest["context_profiles"]
            for repeat in range(1, store.plan.context_repeats + 1)
        }
        expected_tool = {
            ("tool_parallelism", str(scenario["scenario_id"]), str(scenario["expected_execution_mode"]), 1, sample, phase)
            for scenario in store.plan.manifest["tool_parallelism"]["scenarios"]
            for phase, total in (("warmup", store.plan.tool_warmups), ("measurement", store.plan.tool_measurements))
            for sample in range(1, total + 1)
        }
        expected_fault = {
            ("fault_injection", str(case_id), "fault-script", repeat, 1, "run")
            for case_id in store.plan.manifest["fault_injection"]["case_ids"]
            for repeat in range(1, store.plan.fault_repeats + 1)
        }
        if {item.logical_identity for item in context} != expected_context:
            raise FormalCompletenessError("Context logical identities differ from the manifest")
        if {item.logical_identity for item in tool} != expected_tool:
            raise FormalCompletenessError("Tool logical identities differ from the manifest")
        if {item.logical_identity for item in fault} != expected_fault:
            raise FormalCompletenessError("Fault logical identities differ from the manifest")

    @staticmethod
    def _validate_payloads(store: FormalExecutionStore, context: list[FormalRecord], tool: list[FormalRecord], fault: list[FormalRecord]) -> None:
        context_fields = {"validator_success", "input_tokens", "average_input_tokens", "max_input_tokens", "output_tokens", "provider_request_count", "tool_call_count", "compaction_count", "externalization_count", "overflow_recovery_count", "termination_reason", "harness_failure", "fixture_sha256", "prompt_sha256", "validator_sha256", "trace_run_ids"}
        tool_fields = {"scenario_id", "duration_ms", "source_order_correct", "failure_isolated", "mutation_fallback"}
        fault_fields = {"expected_policy", "observation", "passed", "violations"}
        if any(not context_fields.issubset(item.payload) for item in context):
            raise FormalCompletenessError("Context result payload is incomplete")
        if any(not tool_fields.issubset(item.payload) for item in tool):
            raise FormalCompletenessError("Tool result payload is incomplete")
        if any(not fault_fields.issubset(item.payload) for item in fault):
            raise FormalCompletenessError("Fault result payload is incomplete")
        cases = {str(item["case_id"]): item for item in store.plan.manifest["context_cases"]}
        for item in context:
            frozen = cases[item.item_id]
            if (
                item.payload["fixture_sha256"] != frozen["fixture_sha256"]
                or item.payload["prompt_sha256"] != frozen["prompt_sha256"]
                or item.payload["validator_sha256"] != frozen["validator_sha256"]
                or item.payload.get("provider_fingerprint") != store.plan.resolved_provider_fingerprint
            ):
                raise FormalCompletenessError("Context result does not match frozen authority")
        scenarios = {str(item["scenario_id"]): item for item in store.plan.manifest["tool_parallelism"]["scenarios"]}
        for item in tool:
            frozen = scenarios[item.item_id]
            for key in ("batch_size", "tool_classes", "per_tool_latency_ms", "requested_runtime_mode", "expected_execution_mode", "failing_call_index"):
                if item.payload.get(key) != frozen.get(key):
                    raise FormalCompletenessError("Tool sample does not match frozen scenario")
        for item in fault:
            expected = item.payload["expected_policy"]
            observation = item.payload["observation"]
            if expected.get("case_id") != item.item_id:
                raise FormalCompletenessError("Fault expected policy does not match its case")
            if not set(store.plan.manifest["fault_injection"]["observation_schema"]).issubset(observation):
                raise FormalCompletenessError("Fault observation does not match frozen schema")


class FormalAggregator:
    def aggregate(self, store: FormalExecutionStore) -> dict[str, object]:
        counts = CompletenessValidator().validate(store)
        context = store.records("context_ab")
        tool = [item for item in store.records("tool_parallelism") if item.phase == "measurement"]
        fault = store.records("fault_injection")
        context_summary = self._context(context)
        tool_summary = self._tool(tool)
        fault_summary = self._fault(fault)
        return {"execution_id": store.plan.execution_id, "execution_kind": store.plan.execution_kind.value,
                "suite_version": store.plan.suite_version, "counts": counts, "context": context_summary,
                "tool": tool_summary, "fault": fault_summary}

    @staticmethod
    def _context(records: list[FormalRecord]) -> dict[str, object]:
        groups: dict[str, list[FormalRecord]] = {}
        for item in records: groups.setdefault(f"{item.item_id}/{item.profile}", []).append(item)
        output: dict[str, object] = {}
        for key, items in groups.items():
            payloads = [item.payload for item in items]
            inputs = [value["input_tokens"] for value in payloads if value.get("input_tokens") is not None]
            output[key] = {"runs": len(items), "success_rate": sum(bool(value["validator_success"]) for value in payloads) / len(items),
                           "mean_input_tokens": statistics.mean(inputs) if inputs else None,
                           "median_input_tokens": statistics.median(inputs) if inputs else None,
                           "mean_average_input_tokens": statistics.mean(value["average_input_tokens"] for value in payloads if value.get("average_input_tokens") is not None) if inputs else None,
                           "max_single_request_input_tokens": max((value["max_input_tokens"] for value in payloads if value.get("max_input_tokens") is not None), default=None),
                           "mean_duration_ms": statistics.mean(float(value["duration_ms"]) for value in payloads),
                           "provider_requests": sum(int(value["provider_request_count"]) for value in payloads),
                           "compactions": sum(int(value["compaction_count"]) for value in payloads),
                           "externalizations": sum(int(value["externalization_count"]) for value in payloads),
                           "overflow_recoveries": sum(int(value["overflow_recovery_count"]) for value in payloads)}
        return output

    @staticmethod
    def _tool(records: list[FormalRecord]) -> dict[str, object]:
        by_scenario: dict[str, list[FormalRecord]] = {}
        for item in records: by_scenario.setdefault(item.item_id, []).append(item)
        result: dict[str, object] = {"scenarios": {}, "ordering_violations": 0, "failure_isolation_violations": 0, "mutation_fallback_violations": 0}
        for key, items in by_scenario.items():
            latencies = [float(item.payload["duration_ms"]) for item in items]
            result["scenarios"][key] = {"p50_ms": statistics.median(latencies), "p95_ms": _p95(latencies), "samples": len(items)}
            result["ordering_violations"] += sum(not bool(item.payload["source_order_correct"]) for item in items)
            result["failure_isolation_violations"] += sum(not bool(item.payload["failure_isolated"]) for item in items)
            result["mutation_fallback_violations"] += sum(not bool(item.payload["mutation_fallback"]) for item in items)
        result["pairs"] = _tool_pairs(result["scenarios"])
        return result

    @staticmethod
    def _fault(records: list[FormalRecord]) -> dict[str, object]:
        payloads = [item.payload for item in records]
        violations = [value["violations"] for value in payloads]
        recoverable = [value for value in payloads if value["expected_policy"]["case_id"] in {"FI01", "FI03", "FI04", "FI06", "FI07"}]
        return {"runs": len(records), "recoverable_fault_recovery_rate": sum(bool(value["observation"]["recovered"]) for value in recoverable) / len(recoverable),
                "failure_policy_correctness": sum(not value for value in violations) / len(violations),
                "duplicate_side_effect_count": sum(int(value["observation"]["duplicate_side_effect_count"]) for value in payloads),
                "partial_commit_violations": sum(int(value["observation"]["partial_commit_violations"]) for value in payloads),
                "transparent_tool_retry_violations": sum(int(value["observation"]["transparent_tool_retry_violations"]) for value in payloads),
                "unexpected_retry_violations": sum(int(value["observation"]["unexpected_retry_violations"]) for value in payloads)}

    def write(self, store: FormalExecutionStore, summary: dict[str, object]) -> None:
        _write_json(store.root / "summary.json", summary)
        lines = ["# Runtime V1 Formal Execution", "", f"- Execution: `{summary['execution_id']}`", f"- Kind: `{summary['execution_kind']}`", "", "```json", json.dumps(summary, ensure_ascii=False, indent=2), "```", ""]
        (store.root / "report.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")


class FormalRunner:
    def __init__(self, plan: FormalExecutionPlan, root: Path, *, model: Model | None = None, stream_fn: Callable | None = None,
                 keep_failed_workspace: bool = False) -> None:
        execution_parent = "simulations" if plan.execution_kind is FormalExecutionKind.SIMULATION else "formal"
        self.plan, self.root, self.model, self.stream_fn = (
            plan,
            Path(root) / execution_parent / plan.execution_id,
            model,
            stream_fn,
        )
        self.keep_failed_workspace = keep_failed_workspace
        self._context_work_root: Path | None = None

    async def run(self) -> dict[str, object]:
        if self.plan.execution_kind is FormalExecutionKind.FORMAL and (self.model is None or self.stream_fn is None):
            raise ValueError("formal execution requires an explicitly resolved provider model and stream function")
        if self.plan.execution_kind is FormalExecutionKind.FORMAL:
            assert self.model is not None
            actual = _sanitized_provider_fingerprint(self.model)
            expected = self.plan.resolved_provider_fingerprint
            for key in ("provider", "model", "context_window", "temperature", "max_tokens", "provider_timeout_seconds"):
                if actual.get(key) != expected.get(key):
                    raise FormalCompletenessError("resolved provider/model configuration differs from frozen manifest")
        store = FormalExecutionStore.create(self.root, self.plan)
        store.mark_running()
        try:
            await self._run_tool(store)
            await self._run_fault(store)
            await self._run_context(store)
            CompletenessValidator().validate(store, allow_running=True)
            store.mark_completed()
            summary = FormalAggregator().aggregate(store)
            FormalAggregator().write(store, summary)
            return summary
        except BaseException as error:
            store.mark_interrupted(type(error).__name__)
            raise

    async def run_context_sandbox_smoke(self) -> dict[str, object]:
        """Run the formal Context executor's Docker branch without Tool/Fault suites.

        This is a coverage smoke, not a complete Formal execution and therefore
        deliberately does not call the full-suite CompletenessValidator or
        FormalAggregator.
        """
        if self.plan.execution_kind is not FormalExecutionKind.SIMULATION or not self.plan.sandboxed_context:
            raise ValueError("sandboxed Context smoke requires a sandboxed simulation plan")
        store = FormalExecutionStore.create(self.root, self.plan)
        store.mark_running()
        try:
            sandbox = self.plan.manifest["sandbox"]
            resolved_digest = _resolve_docker_image_digest(str(sandbox["image"]))
            if resolved_digest != sandbox["image_digest"]:
                raise FormalCompletenessError("resolved Docker image digest differs from the frozen manifest")
            metadata = store.execution_metadata()
            metadata["resolved_sandbox_image_digest"] = resolved_digest
            metadata["context_smoke_scope"] = {
                "case_ids": ["CM01_large_tool_output_repair"], "profiles": ["base", "full"],
            }
            _write_json(store.root / "formal-metadata.json", metadata)
            await self._run_context(
                store,
                case_ids={"CM01_large_tool_output_repair"},
                profiles={"base", "full"},
            )
            records = store.records("context_ab")
            if len(records) != 2 or any(not record.payload.get("sandbox_created") or not record.payload.get("sandbox_discarded") or not record.payload.get("host_workspace_unchanged") or not record.payload.get("sandbox_shell_verified") for record in records):
                raise FormalCompletenessError("sandboxed Context smoke did not produce the expected isolated records")
            store.mark_completed()
            summary = {
                "execution_id": self.plan.execution_id,
                "execution_kind": self.plan.execution_kind.value,
                "scope": "context_sandbox_smoke",
                "counts": {"context": len(records)},
                "sandbox_image": sandbox["image"],
                "sandbox_image_digest": resolved_digest,
            }
            _write_json(store.root / "summary.json", summary)
            (store.root / "report.md").write_text(
                "# Runtime V1 Sandboxed Context Simulation\n\n"
                f"- Execution: `{self.plan.execution_id}`\n"
                f"- Context records: {len(records)}\n"
                f"- Image digest: `{resolved_digest}`\n",
                encoding="utf-8",
                newline="\n",
            )
            return summary
        except BaseException as error:
            store.mark_interrupted(type(error).__name__)
            raise

    async def _run_tool(self, store: FormalExecutionStore) -> None:
        for raw in self.plan.manifest["tool_parallelism"]["scenarios"]:
            scenario = FrozenToolScenario.from_mapping(raw)
            for phase, count in (("warmup", self.plan.tool_warmups), ("measurement", self.plan.tool_measurements)):
                for sample in range(1, count + 1):
                    started = _now(); observation = (await run_tool_parallel_scenarios([scenario]))[0]
                    store.append(FormalRecord.from_payload(self.plan, experiment="tool_parallelism", item_id=scenario.scenario_id,
                        profile=observation.mode, repeat_index=1, sample_index=sample, phase=phase,
                        payload={**scenario.to_dict(), **asdict(observation)}, trace_run_id=None, started_at=started, duration_ms=observation.duration_ms))

    async def _run_fault(self, store: FormalExecutionStore) -> None:
        trace_root = store.root / "fault-source-traces"
        for case_id in self.plan.manifest["fault_injection"]["case_ids"]:
            for repeat in range(1, self.plan.fault_repeats + 1):
                started = _now(); result = await run_fault_case(case_id, repeat_index=repeat, trace_root=trace_root)
                source = JsonlTraceStore(trace_root / f"{case_id}-{repeat}.jsonl").load_all()[-1]
                store.append_trace(source)
                store.append(FormalRecord.from_payload(self.plan, experiment="fault_injection", item_id=case_id, profile="fault-script",
                    repeat_index=repeat, sample_index=1, phase="run", payload={"expected_policy": asdict(result.expected_policy),
                    "observation": asdict(result.observation), "passed": result.passed, "violations": list(result.violations)},
                    trace_run_id=source.run_id, started_at=started, duration_ms=source.duration_ms or 0.0))
        shutil.rmtree(trace_root, ignore_errors=True)

    async def _run_context(
        self,
        store: FormalExecutionStore,
        *,
        case_ids: set[str] | None = None,
        profiles: set[str] | None = None,
    ) -> None:
        fixture_map = {item.case_id: item for item in dry_run_context_fixtures()}
        settings = self.plan.manifest["shared_runtime_settings"]
        sandbox = self.plan.manifest["sandbox"]
        # Eval workspaces are intentionally outside the durable result tree:
        # formal metadata never exposes their absolute paths, and Windows path
        # limits remain safe even for a long result/execution identifier.
        self._context_work_root = Path(tempfile.mkdtemp(prefix="rova-eval-v1-"))
        try:
            for case_number, raw_case in enumerate(self.plan.manifest["context_cases"], start=1):
                fixture = fixture_map[str(raw_case["case_id"])]
                if case_ids is not None and fixture.case_id not in case_ids:
                    continue
                if fixture.sha256 != raw_case["fixture_sha256"]:
                    raise FormalCompletenessError("fixture hash differs from frozen manifest")
                for profile_name, raw_profile in self.plan.manifest["context_profiles"].items():
                    if profiles is not None and profile_name not in profiles:
                        continue
                    profile = self._profile(profile_name, raw_profile)
                    for repeat in range(1, self.plan.context_repeats + 1):
                        root = self._context_work_root / f"{case_number}-{profile_name[:1]}-{repeat}"
                        started = _now()
                        execution, requests, observation = await _run_case(
                            fixture,
                            profile,
                            root=root,
                            model=self.model or _DRY_MODEL,
                            stream_fn=self.stream_fn,
                            use_development_provider=self.plan.execution_kind is FormalExecutionKind.SIMULATION,
                        isolated_sandbox=(
                            self.plan.execution_kind is FormalExecutionKind.FORMAL
                            or self.plan.sandboxed_context
                        ),
                            sandbox_image=str(sandbox["image"]),
                            keep_failed_workspace=self.keep_failed_workspace,
                            max_turns=int(settings["max_turns"]),
                            provider_max_retries=int(settings["provider_max_retries"]),
                            exercise_sandbox_shell=self.plan.sandboxed_context,
                        )

                        class Executor:
                            async def execute(self, _case):
                                return execution

                        result = (
                            await EvalRunner(
                                Executor(),
                                lambda _case: [_ContextValidator()],
                                suite_id=self.plan.execution_id,
                            ).run([EvalCase(fixture.case_id, fixture.case_id, "formal")])
                        ).case_results[0]
                        traces = list(execution.artifacts["run_traces"])
                        for trace in traces:
                            store.append_trace(trace)
                        usage = [trace.actual_usage for trace in traces if trace.actual_usage is not None]
                        runtime_duration_ms = sum(trace.duration_ms or 0.0 for trace in traces)
                        input_tokens = [item.input_tokens for item in usage]
                        output_tokens = [item.output_tokens for item in usage]
                        payload = {
                            "validator_success": result.task_success,
                            "duration_ms": runtime_duration_ms,
                            "input_tokens": sum(input_tokens) if input_tokens else None,
                            "average_input_tokens": (sum(input_tokens) / requests) if input_tokens and requests else None,
                            "max_input_tokens": max(input_tokens) if input_tokens else None,
                            "output_tokens": sum(output_tokens) if output_tokens else None,
                            "total_tokens": sum(item.total_tokens for item in usage) if usage else None,
                            "provider_request_count": requests,
                            "tool_call_count": sum(len(call.tool_calls) for trace in traces for call in trace.steps),
                            "compaction_count": observation.compaction_count,
                            "externalization_count": observation.externalization_count,
                            "overflow_recovery_count": observation.overflow_recovery_count,
                            "termination_reason": observation.termination_reason,
                            "harness_failure": result.to_dict()["failure_reason"] == "runtime_harness_error",
                            "eval_result": result.to_dict(),
                            "fixture_sha256": fixture.sha256,
                            "prompt_sha256": raw_case["prompt_sha256"],
                            "validator_sha256": raw_case["validator_sha256"],
                            "allowed_change_paths": raw_case["allowed_change_paths"],
                            "provider_fingerprint": self.plan.resolved_provider_fingerprint,
                            "max_turns": settings["max_turns"],
                            "trace_run_ids": [trace.run_id for trace in traces],
                            "retained_workspace": bool(self.keep_failed_workspace and result.task_success is False),
                            "sandbox_created": bool(execution.artifacts["sandbox_created"]),
                            "sandbox_discarded": bool(execution.artifacts["sandbox_discarded"]),
                            "host_workspace_unchanged": bool(execution.artifacts["host_workspace_unchanged"]),
                            "sandbox_shell_verified": bool(execution.artifacts["sandbox_shell_verified"]),
                            "sandbox_image": sandbox["image"] if (self.plan.execution_kind is FormalExecutionKind.FORMAL or self.plan.sandboxed_context) else None,
                            "sandbox_image_digest": sandbox["image_digest"] if (self.plan.execution_kind is FormalExecutionKind.FORMAL or self.plan.sandboxed_context) else None,
                        }
                        store.append(FormalRecord.from_payload(
                            self.plan,
                            experiment="context_ab",
                            item_id=fixture.case_id,
                            profile=profile_name,
                            repeat_index=repeat,
                            sample_index=1,
                            phase="run",
                            payload=payload,
                            trace_run_id=execution.run_trace.run_id,
                            started_at=started, duration_ms=runtime_duration_ms))
                        if not payload["retained_workspace"]:
                            shutil.rmtree(root, ignore_errors=True)
        finally:
            if not self.keep_failed_workspace:
                shutil.rmtree(self._context_work_root, ignore_errors=True)

    @staticmethod
    def _profile(name: str, raw: dict[str, object]) -> ContextManagementProfile:
        if name == "base":
            profile = ContextManagementProfile.base()
        elif name == "full":
            policy = raw["compaction_policy"]
            profile = ContextManagementProfile.full(
                CompactionPolicy(
                    reserve_tokens=int(policy["reserve_tokens"]),
                    keep_recent_tokens=int(policy["keep_recent_tokens"]),
                )
            )
        else:
            raise FormalCompletenessError("unknown frozen Context profile")
        if (profile.compaction_policy is not None) != bool(raw["compaction"]) or profile.enable_tool_result_externalization != bool(raw["tool_result_externalization"]) or profile.enable_context_overflow_recovery != bool(raw["overflow_recovery"]):
            raise FormalCompletenessError("Context runtime profile differs from frozen manifest")
        return profile


def _p95(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[max(0, min(len(ordered) - 1, math.ceil(len(ordered) * 0.95) - 1))]


def _tool_pairs(scenarios: dict[str, object]) -> dict[str, object]:
    """Derive sequential/parallel comparisons from persisted measurements."""
    pairs: dict[str, dict[str, object]] = {}
    for scenario_id, metrics in scenarios.items():
        if not isinstance(metrics, dict):
            continue
        if scenario_id.endswith("_sequential"):
            base, mode = scenario_id[:-len("_sequential")], "sequential"
        elif scenario_id.endswith("_parallel"):
            base, mode = scenario_id[:-len("_parallel")], "parallel"
        else:
            continue
        pairs.setdefault(base, {})[mode] = metrics
    output: dict[str, object] = {}
    for base, values in pairs.items():
        sequential, parallel = values.get("sequential"), values.get("parallel")
        if not isinstance(sequential, dict) or not isinstance(parallel, dict):
            continue
        sequential_p50 = float(sequential["p50_ms"])
        parallel_p50 = float(parallel["p50_ms"])
        output[base] = {
            "sequential_p50_ms": sequential_p50,
            "parallel_p50_ms": parallel_p50,
            "sequential_p95_ms": float(sequential["p95_ms"]),
            "parallel_p95_ms": float(parallel["p95_ms"]),
            "speedup": sequential_p50 / parallel_p50 if parallel_p50 else None,
            "latency_reduction": (sequential_p50 - parallel_p50) / sequential_p50 if sequential_p50 else None,
        }
    return output


def _resolve_docker_image_digest(image: str) -> str:
    """Resolve the image content digest used by the sandboxed smoke."""
    completed = subprocess.run(
        ["docker", "image", "inspect", "--format", "{{json .RepoDigests}} {{.Id}}", image],
        check=True,
        capture_output=True,
        text=True,
    )
    output = completed.stdout.strip()
    raw_digests, image_id = output.rsplit(" ", 1)
    digests = json.loads(raw_digests)
    if digests:
        return str(digests[0]).split("@", 1)[-1]
    return image_id
