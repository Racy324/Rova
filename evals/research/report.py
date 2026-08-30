from __future__ import annotations

import json
from pathlib import Path


def render_report(result_dir: Path) -> Path:
    summary = json.loads((result_dir / "summary.json").read_text(encoding="utf-8"))
    profiles = summary["profiles"]
    lines = ["# Rova Research Agent Evaluation", "", f"- Eval version: `{summary['eval_version']}`", f"- Runs: {summary['run_count']}", "", "## Aggregate metrics", "", "| Profile | Task success | Protocol compliance | Valid / infra | Avg tools | Avg total tokens |", "|---|---:|---:|---:|---:|---:|"]
    for name in ("base", "full"):
        value = profiles[name]
        lines.append(f"| {name} | {_percent(value['task_success_rate'])} | {_percent(value['protocol_compliance_rate'])} | {value['valid_runs']} / {value['infra_error_count']} | {_number(value['average_tool_calls'])} | {_number(value['average_total_tokens'])} |")
    lines.extend(["", "## Per-case metrics", "", "| Case | Base success | Full success | Base protocol | Full protocol |", "|---|---:|---:|---:|---:|"])
    for case_id, value in summary["cases"].items():
        lines.append(f"| {case_id} | {_percent(value['base']['task_success_rate'])} | {_percent(value['full']['task_success_rate'])} | {_percent(value['base']['protocol_compliance_rate'])} | {_percent(value['full']['protocol_compliance_rate'])} |")
    c07 = summary["c07"]
    lines.extend(["", "## C07 long-context evidence", "", f"- Base average actual provider input tokens: {_number(c07['base_average_provider_input_tokens'])}", f"- Full average actual provider input tokens: {_number(c07['full_average_provider_input_tokens'])}", f"- Token reduction: {_percent(c07['token_reduction'])}", f"- Full constraint retention: {_percent(c07['full_constraint_retention'])}", f"- Full runs with observed compaction: {c07['full_compaction_runs']}", "", "Raw records: `runs.jsonl`; semantic traces: `traces.jsonl`; design audit: `eval-design-audit.md`."])
    path = result_dir / "research-eval-report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return path


def _percent(value: float | None) -> str:
    return "N/A" if value is None else f"{value * 100:.1f}%"


def _number(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.1f}"
