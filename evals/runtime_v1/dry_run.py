from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from .fault_benchmark import FaultDryRunReport, run_fault_dry_run
from .tool_parallel import ToolParallelDryRunReport, run_tool_parallel_dry_run


async def run_offline_dry_run(result_root: Path) -> dict[str, object]:
    """Persist the Provider-free Dry Run evidence alongside Context results."""
    root = Path(result_root)
    root.mkdir(parents=True, exist_ok=True)
    tool_report = await run_tool_parallel_dry_run()
    fault_report = await run_fault_dry_run()
    payload = {
        "tool_parallel": _tool_payload(tool_report),
        "fault_injection": _fault_payload(fault_report),
    }
    (root / "offline-dry-run-summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return payload


def _tool_payload(report: ToolParallelDryRunReport) -> dict[str, object]:
    return {
        "provider_request_count": report.provider_request_count,
        "failure_isolation_violations": report.failure_isolation_violations,
        "mutation_fallback_violations": report.mutation_fallback_violations,
        "observations": [asdict(item) for item in report.observations],
    }


def _fault_payload(report: FaultDryRunReport) -> dict[str, object]:
    return {
        "provider_request_count": report.provider_request_count,
        "recoverable_fault_recovery_rate": report.recoverable_fault_recovery_rate,
        "failure_policy_violations": report.failure_policy_violations,
        "duplicate_side_effect_count": report.duplicate_side_effect_count,
        "partial_commit_violations": report.partial_commit_violations,
        "transparent_tool_retry_violations": report.transparent_tool_retry_violations,
        "observations": [asdict(item) for item in report.observations],
    }
