from __future__ import annotations

import json
from pathlib import Path

from evals.research.graders import grade_case
from evals.research.runner import aggregate_records, write_design_audit
from evals.research.spec import CASES, EVAL_VERSION, get_case, select_cases


def test_research_cases_are_versioned_and_have_deterministic_checks():
    assert EVAL_VERSION == "research-eval-v1"
    assert [case.case_id for case in CASES] == [f"C0{index}" for index in range(1, 9)]
    for case in CASES:
        assert case.prompt
        assert case.critical_checks
        assert case.protocol_checks
        assert case.required_capabilities


def test_select_cases_preserves_benchmark_order_and_rejects_unknown_ids():
    assert [case.case_id for case in select_cases(("C07",))] == ["C07"]
    assert [case.case_id for case in select_cases(("C08", "C01"))] == ["C01", "C08"]
    with __import__("pytest").raises(ValueError, match="unknown"):
        select_cases(("C99",))


def test_evidence_writing_grader_is_profile_independent(tmp_path: Path):
    output = "AP50 improved from 82.4 to 84.1; baseline AP50_95 is 51.7 and the variant value is missing."
    base = grade_case(get_case("C03"), output, tmp_path, (), profile="base")
    full = grade_case(get_case("C03"), output, tmp_path, (), profile="full")
    assert base == full
    assert all(item["passed"] for item in base["critical_checks"])
    assert all(item["passed"] for item in base["protocol_checks"])


def test_aggregate_excludes_infrastructure_error_and_counts_protocol():
    records = [
        {"profile": "base", "status": "PASS", "critical_checks": [{"passed": True}], "protocol_checks": [{"passed": True}, {"passed": False}], "token_usage": {"total_tokens": 10}, "tool_calls": 2, "duration_ms": 5},
        {"profile": "base", "status": "INFRA_ERROR", "critical_checks": [], "protocol_checks": [], "token_usage": {"total_tokens": 0}, "tool_calls": 0, "duration_ms": 0},
    ]
    summary = aggregate_records(records)
    assert summary["profiles"]["base"]["valid_runs"] == 1
    assert summary["profiles"]["base"]["task_success_rate"] == 1.0
    assert summary["profiles"]["base"]["protocol_compliance_rate"] == 0.5
    assert summary["profiles"]["base"]["infra_error_count"] == 1


def test_design_audit_rejects_profile_specific_grading(tmp_path: Path):
    path = write_design_audit(tmp_path, grader_profile_dependent=True)
    payload = json.loads((tmp_path / "eval-design-audit.json").read_text(encoding="utf-8"))
    assert path.exists()
    assert payload["passed"] is False
    assert any(not item["passed"] for item in payload["checks"])
