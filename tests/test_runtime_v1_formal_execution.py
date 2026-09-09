from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_formal_plan_is_manifest_driven_and_simulation_only_changes_scale() -> None:
    from evals.runtime_v1.formal import FormalExecutionKind, FormalExecutionPlan

    manifest = json.loads(Path("evals/runtime_v1/frozen/runtime_v1_evaluation_v3.json").read_text(encoding="utf-8"))
    formal = FormalExecutionPlan.from_manifest(manifest, eval_suite_commit="e" * 40)
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


def test_formal_store_persists_metadata_and_rejects_duplicate_logical_identity(tmp_path: Path) -> None:
    from evals.runtime_v1.formal import FormalExecutionKind, FormalExecutionPlan, FormalExecutionStore, FormalRecord, FormalStoreError

    manifest = json.loads(Path("evals/runtime_v1/frozen/runtime_v1_evaluation_v3.json").read_text(encoding="utf-8"))
    plan = FormalExecutionPlan.from_manifest(manifest, eval_suite_commit="e" * 40).for_simulation()
    store = FormalExecutionStore.create(tmp_path / "simulation", plan)
    record = FormalRecord.from_payload(
        plan, experiment="tool_parallelism", item_id="TP_uniform_1_parallel", profile="parallel",
        repeat_index=1, sample_index=1, phase="measurement", payload={"duration_ms": 1.0}, trace_run_id=None,
    )
    store.append(record)

    assert store.execution_metadata()["execution_kind"] == "simulation"
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

    manifest = json.loads(Path("evals/runtime_v1/frozen/runtime_v1_evaluation_v3.json").read_text(encoding="utf-8"))
    plan = FormalExecutionPlan.from_manifest(manifest, eval_suite_commit="e" * 40).for_simulation()
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
    from evals.runtime_v1.formal import FormalExecutionPlan, FormalRunner

    manifest = json.loads(Path("evals/runtime_v1/frozen/runtime_v1_evaluation_v3.json").read_text(encoding="utf-8"))
    plan = FormalExecutionPlan.from_manifest(manifest, eval_suite_commit="e" * 40).for_simulation()
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
