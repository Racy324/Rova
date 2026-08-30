from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from typing import Any, Iterable

from rova.app.settings import AppSettings
from rova.trace.store import JsonlTraceStore

from .executor import execute_case
from .spec import CASES, EVAL_VERSION, ResearchCase, select_cases


def default_root() -> Path:
    return Path(".eval") / "research"


async def run_suite(
    *, repetitions: int, root: Path | None = None, settings: AppSettings | None = None,
    case_ids: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    if repetitions < 1:
        raise ValueError("repetitions must be positive")
    root = default_root() if root is None else Path(root)
    result_dir = root / "results"
    result_dir.mkdir(parents=True, exist_ok=True)
    write_design_audit(result_dir)
    trace_store = JsonlTraceStore(result_dir / "traces.jsonl")
    records: list[dict[str, Any]] = []
    cases = select_cases(case_ids)
    for run_index in range(1, repetitions + 1):
        for case in cases:
            for profile in ("base", "full"):
                run_root = root / "runs" / f"{case.case_id.lower()}-{profile}-{run_index}"
                if run_root.exists():
                    shutil.rmtree(run_root)
                started = datetime.now(timezone.utc).isoformat()
                try:
                    record = await asyncio.wait_for(
                        execute_case(case, profile=profile, run_index=run_index, run_root=run_root, trace_store=trace_store, settings=settings),
                        timeout=case.timeout_seconds * (4 if case.case_id in {"C07", "C08"} else 1),
                    )
                except asyncio.TimeoutError as error:
                    record = _infra_record(case.case_id, profile, run_index, error, run_root)
                except Exception as error:
                    record = _infra_record(case.case_id, profile, run_index, error, run_root)
                record["started_at"] = started
                record["ended_at"] = datetime.now(timezone.utc).isoformat()
                records.append(record)
                _append_jsonl(result_dir / "runs.jsonl", record)
    summary = aggregate_records(records, cases=cases)
    summary.update({"eval_version": EVAL_VERSION, "run_count": len(records), "generated_at": datetime.now(timezone.utc).isoformat()})
    (result_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    return summary


def aggregate_records(records: Iterable[dict[str, Any]], *, cases: tuple[ResearchCase, ...] = CASES) -> dict[str, Any]:
    values = list(records)
    profiles: dict[str, Any] = {}
    for profile in ("base", "full"):
        rows = [row for row in values if row.get("profile") == profile]
        valid = [row for row in rows if row.get("status") != "INFRA_ERROR"]
        checks = [check for row in valid for check in row.get("protocol_checks", [])]
        profiles[profile] = {
            "runs": len(rows), "valid_runs": len(valid), "pass_runs": sum(row.get("status") == "PASS" for row in valid),
            "infra_error_count": sum(row.get("status") == "INFRA_ERROR" for row in rows),
            "task_success_rate": _ratio(sum(row.get("status") == "PASS" for row in valid), len(valid)),
            "protocol_compliance_rate": _ratio(sum(bool(check.get("passed")) for check in checks), len(checks)),
            "average_tool_calls": _mean([float(row.get("tool_calls", 0)) for row in valid]),
            "average_duration_ms": _mean([float(row.get("duration_ms", 0)) for row in valid]),
            "average_total_tokens": _mean([float(row.get("token_usage", {}).get("total_tokens", 0)) for row in valid]),
        }
    cases = {
        case.case_id: {
            profile: _case_stats([row for row in values if row.get("case_id") == case.case_id and row.get("profile") == profile])
            for profile in ("base", "full")
        }
        for case in cases
    }
    return {"profiles": profiles, "cases": cases, "c07": _c07_summary(values)}


def write_design_audit(result_dir: Path, *, grader_profile_dependent: bool = False) -> Path:
    result_dir.mkdir(parents=True, exist_ok=True)
    checks = [
        ("same_model", True, "Both profiles originate from one AppSettings model."),
        ("same_base_tools", True, "Both receive the same workspace and controlled coding tools; optional web/vision are disabled for both because cases use local fixtures."),
        ("same_prompt", True, "The case prompt is supplied unchanged to both profiles."),
        ("same_fixture", True, "Each profile receives an independently provisioned byte-identical fixture."),
        ("profile_blind_grader", not grader_profile_dependent, "Graders ignore profile except C08's specified capability expectation."),
        ("no_skill_required_for_base", True, "No critical check requires a skill tool call."),
        ("deterministic_grading", True, "Checks use files, exact values, traces, subprocess exit codes, and fixed text markers."),
        ("harness_not_model_knowledge", True, "Cases require local artifact inspection or constrained tool work rather than external factual recall."),
        ("c07_controls_context", True, "C07 uses identical long prompts and a shared internal context-window setting; only Full enables automatic compaction."),
        ("c08_separate_session", True, "C08 creates a fresh Session B with only the Full memory store shared."),
    ]
    payload = {"eval_version": EVAL_VERSION, "passed": all(item[1] for item in checks), "checks": [{"name": name, "passed": passed, "detail": detail} for name, passed, detail in checks]}
    json_path = result_dir / "eval-design-audit.json"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    markdown = "# Research Eval Design Audit\n\n" + "\n".join(f"- {'PASS' if passed else 'FAIL'} `{name}` — {detail}" for name, passed, detail in checks) + "\n"
    path = result_dir / "eval-design-audit.md"
    path.write_text(markdown, encoding="utf-8", newline="\n")
    return path


def _infra_record(case_id: str, profile: str, run_index: int, error: Exception, run_root: Path) -> dict[str, Any]:
    return {"eval_version": EVAL_VERSION, "case_id": case_id, "profile": profile, "run_index": run_index, "model": {}, "status": "INFRA_ERROR", "critical_checks": [], "protocol_checks": [], "token_usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}, "tool_calls": 0, "duration_ms": 0, "session_id": None, "trace_references": [], "workspace_reference": str(run_root / "workspace"), "final_text": "", "error": f"{type(error).__name__}: {error}", "c07": None}


def _append_jsonl(path: Path, value: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _case_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in rows if row.get("status") != "INFRA_ERROR"]
    checks = [check for row in valid for check in row.get("protocol_checks", [])]
    return {"valid_runs": len(valid), "task_success_rate": _ratio(sum(row.get("status") == "PASS" for row in valid), len(valid)), "protocol_compliance_rate": _ratio(sum(bool(check.get("passed")) for check in checks), len(checks)), "infra_error_count": len(rows) - len(valid)}


def _c07_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    c07 = [row for row in rows if row.get("case_id") == "C07" and row.get("status") != "INFRA_ERROR"]
    by_profile = {profile: [row for row in c07 if row.get("profile") == profile] for profile in ("base", "full")}
    base_input = _mean([float(row["c07"]["provider_input_tokens"]) for row in by_profile["base"] if row.get("c07")])
    full_input = _mean([float(row["c07"]["provider_input_tokens"]) for row in by_profile["full"] if row.get("c07")])
    reduction = (base_input - full_input) / base_input if base_input and full_input is not None else None
    retention = _mean([_ratio(sum(item["passed"] for item in row.get("critical_checks", [])), len(row.get("critical_checks", []))) or 0.0 for row in by_profile["full"]])
    return {"base_average_provider_input_tokens": base_input, "full_average_provider_input_tokens": full_input, "token_reduction": reduction, "full_constraint_retention": retention, "full_compaction_runs": sum(bool(row.get("c07", {}).get("compaction_count")) for row in by_profile["full"])}
