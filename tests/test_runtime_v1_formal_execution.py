from __future__ import annotations

import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path

import pytest


def _successful_context_execution(case_id: str, profile: str, index: int):
    from evals.runtime_v1.context_ab import ContextDryRunObservation
    from rova.ai.messages import AssistantMessage, TextBlock
    from rova.eval import EvalExecution
    from rova.trace.models import RunStatus, RunTrace, TerminationReason

    started_at = datetime(2026, 9, 9, tzinfo=timezone.utc)
    trace = RunTrace(
        run_id=f"context-runtime-{index}",
        started_at=started_at,
        ended_at=started_at,
        duration_ms=1.0,
        status=RunStatus.COMPLETED,
        termination_reason=TerminationReason.FINAL_RESPONSE,
    )
    execution = EvalExecution(
        case_id,
        trace,
        AssistantMessage([TextBlock("completed")]),
        artifacts={
            "validator_passed": True,
            "run_traces": (trace,),
            "sandbox_created": False,
            "host_workspace_unchanged": True,
            "sandbox_discarded": True,
            "sandbox_shell_verified": True,
        },
    )
    observation = ContextDryRunObservation(
        case_id, profile, 1, 0, 0, 0, "final_response", None, None, 1.0,
    )
    return execution, 1, observation


@pytest.mark.asyncio
async def test_formal_context_runtime_failure_is_persisted_and_later_runs_continue(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from evals.runtime_v1 import formal
    from evals.runtime_v1.context_ab import ContextRuntimeExecutionError
    from rova.agent_session.agent_session import CompactionInputTooLarge

    manifest_path = Path("evals/runtime_v1/frozen/runtime_v1_evaluation_v6.json")
    plan = formal.FormalExecutionPlan.from_frozen_manifest(
        manifest_path, eval_suite_commit="e" * 40,
    ).for_simulation()
    calls = 0

    async def fake_run_case(fixture, profile, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ContextRuntimeExecutionError(
                CompactionInputTooLarge("summary input does not fit"),
                provider_request_count=0,
                traces=(),
                duration_ms=2.0,
            )
        return _successful_context_execution(fixture.case_id, profile.name, calls)

    monkeypatch.setattr(formal, "_run_case", fake_run_case)
    summary = await formal.FormalRunner(plan, tmp_path).run()
    store = formal.FormalExecutionStore(tmp_path / "simulations" / plan.execution_id, plan)
    failed, *remaining = store.records("context_ab")

    assert calls == 4
    assert store.state() is formal.FormalExecutionState.COMPLETED
    assert len(remaining) == 3
    assert failed.payload["validator_success"] is False
    assert failed.payload["validator_status"] == "not_run"
    assert failed.payload["runtime_failure_type"] == "CompactionInputTooLarge"
    assert failed.payload["runtime_failure_message"] == "runtime execution raised CompactionInputTooLarge"
    assert failed.payload["trace_run_ids"] == []
    assert failed.payload["provider_request_count"] == 0
    assert summary["context"]["CM01_large_tool_output_repair/base"]["task_failures"] == 1
    assert summary["context"]["CM01_large_tool_output_repair/base"]["runtime_execution_failures"] == 1


@pytest.mark.asyncio
async def test_formal_context_store_failure_remains_execution_level_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from evals.runtime_v1.formal import FormalExecutionPlan, FormalExecutionState, FormalExecutionStore, FormalRunner, FormalStoreError

    manifest_path = Path("evals/runtime_v1/frozen/runtime_v1_evaluation_v6.json")
    plan = FormalExecutionPlan.from_frozen_manifest(manifest_path, eval_suite_commit="e" * 40).for_simulation()
    original_append = FormalExecutionStore.append

    def fail_context_persistence(self, record):
        if record.experiment == "context_ab":
            raise FormalStoreError("injected context store failure")
        return original_append(self, record)

    monkeypatch.setattr(FormalExecutionStore, "append", fail_context_persistence)

    with pytest.raises(FormalStoreError, match="injected context store failure"):
        await FormalRunner(plan, tmp_path).run()

    store = FormalExecutionStore(tmp_path / "simulations" / plan.execution_id, plan)
    assert store.state() is FormalExecutionState.INTERRUPTED


def test_formal_plan_is_manifest_driven_and_simulation_only_changes_scale() -> None:
    from evals.runtime_v1.formal import FormalExecutionKind, FormalExecutionPlan

    manifest_path = Path("evals/runtime_v1/frozen/runtime_v1_evaluation_v6.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    formal = FormalExecutionPlan.from_frozen_manifest(manifest_path, eval_suite_commit="e" * 40)
    simulation = formal.for_simulation()
    sandboxed_simulation = formal.for_simulation(sandboxed_context=True)

    assert formal.execution_kind is FormalExecutionKind.FORMAL
    assert formal.context_repeats == manifest["formal_runs"]["context_ab"] // 4
    assert formal.tool_measurements == manifest["tool_parallelism"]["measured_samples_per_scenario"]
    assert formal.fault_repeats == manifest["fault_injection"]["repeats"]
    assert simulation.execution_kind is FormalExecutionKind.SIMULATION
    assert simulation.context_repeats == 1
    assert simulation.tool_warmups == 1
    assert simulation.tool_measurements == 2
    assert simulation.fault_repeats == 1
    assert simulation.manifest_sha256 == formal.manifest_sha256
    assert simulation.sandboxed_context is False
    assert sandboxed_simulation.sandboxed_context is True
    assert formal.manifest_sha256 == hashlib.sha256(manifest_path.read_bytes()).hexdigest()


def test_formal_store_persists_metadata_and_rejects_duplicate_logical_identity(tmp_path: Path) -> None:
    from evals.runtime_v1.formal import FormalExecutionKind, FormalExecutionPlan, FormalExecutionStore, FormalRecord, FormalStoreError

    manifest_path = Path("evals/runtime_v1/frozen/runtime_v1_evaluation_v6.json")
    plan = FormalExecutionPlan.from_frozen_manifest(manifest_path, eval_suite_commit="e" * 40).for_simulation()
    store = FormalExecutionStore.create(tmp_path / "simulation", plan)
    record = FormalRecord.from_payload(
        plan, experiment="tool_parallelism", item_id="TP_uniform_1_parallel", profile="parallel",
        repeat_index=1, sample_index=1, phase="measurement", payload={"duration_ms": 1.0}, trace_run_id=None,
    )
    store.append(record)

    assert store.execution_metadata()["execution_kind"] == "simulation"
    assert store.execution_metadata()["manifest_sha256"] == hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    assert (tmp_path / "simulation" / "frozen-manifest.json").read_bytes() == manifest_path.read_bytes()
    assert len(store.records("tool_parallelism")) == 1
    with pytest.raises(FormalStoreError, match="duplicate logical identity"):
        store.append(record)


def test_completeness_rejects_interrupted_or_malformed_formal_records(tmp_path: Path) -> None:
    from evals.runtime_v1.formal import (
        CompletenessValidator,
        FormalCompletenessError,
        FormalExecutionPlan,
        FormalExecutionStore,
        FormalRecord,
    )

    manifest_path = Path("evals/runtime_v1/frozen/runtime_v1_evaluation_v6.json")
    plan = FormalExecutionPlan.from_frozen_manifest(manifest_path, eval_suite_commit="e" * 40).for_simulation()
    store = FormalExecutionStore.create(tmp_path / "simulation", plan)
    store.mark_running()
    store.append(FormalRecord.from_payload(
        plan, experiment="context_ab", item_id="CM01_large_tool_output_repair", profile="base",
        repeat_index=1, sample_index=1, phase="run", payload={}, trace_run_id=None,
    ))

    with pytest.raises(FormalCompletenessError, match="incomplete"):
        CompletenessValidator().validate(store, allow_running=True)
    store.mark_interrupted("test")
    with pytest.raises(FormalCompletenessError, match="not complete"):
        CompletenessValidator().validate(store)


@pytest.mark.asyncio
async def test_formal_simulation_runs_through_store_completeness_and_aggregation(tmp_path: Path) -> None:
    from evals.runtime_v1.formal import FormalExecutionPlan, FormalExecutionStore, FormalRunner

    manifest_path = Path("evals/runtime_v1/frozen/runtime_v1_evaluation_v6.json")
    plan = FormalExecutionPlan.from_frozen_manifest(manifest_path, eval_suite_commit="e" * 40).for_simulation()
    output_root = tmp_path / "simulation"
    summary = await FormalRunner(plan, output_root).run()
    execution_root = output_root / "simulations" / plan.execution_id

    assert summary["execution_kind"] == "simulation"
    assert summary["counts"] == {"context": 4, "tool_measurements": 24, "tool_warmups": 12, "fault": 12}
    assert (execution_root / "summary.json").is_file()
    assert (execution_root / "report.md").is_file()
    assert (execution_root / "raw" / "context-results.jsonl").is_file()
    assert (execution_root / "raw" / "tool-samples.jsonl").is_file()
    assert (execution_root / "raw" / "fault-results.jsonl").is_file()
    assert (execution_root / "traces" / "runs.jsonl").is_file()
    assert not (execution_root / "cw").exists()
    assert summary["tool"]["pairs"]
    store = FormalExecutionStore(execution_root, plan)
    traces = {trace.run_id: trace for trace in store.trace_store.load_all()}
    for record in store.records("context_ab"):
        trace_duration_ms = sum(
            traces[trace_id].duration_ms or 0.0
            for trace_id in record.payload["trace_run_ids"]
        )
        assert record.duration_ms == pytest.approx(trace_duration_ms)
        assert record.payload["duration_ms"] == pytest.approx(trace_duration_ms)
