from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from .spec import ResearchCase


def grade_case(
    case: ResearchCase,
    final_text: str,
    workspace: Path,
    tools: Iterable[dict[str, Any]],
    *,
    profile: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Deterministic, profile-blind checks over outputs, fixtures, and traces."""
    del profile
    extra = extra or {}
    text = _all_text(final_text, workspace).lower()
    tool_list = list(tools)
    critical: dict[str, bool]
    protocol: dict[str, bool]
    if case.case_id == "C01":
        critical = {
            "completed": bool(final_text.strip()), "artifact_exists": (workspace / "paper_evidence.md").is_file(),
            "core_facts": all(value in text for value in ("dfformer", "deformable", "frequency")),
            "page_anchor": "page 1" in text or "p. 1" in text,
            "no_invented_module": "transformer pyramid" not in text,
        }
        names = [item.get("tool_name") for item in tool_list]
        commands = "\n".join(str(item.get("command", "")) for item in tool_list).lower()
        protocol = {
            "no_multiple_pdf_parsers": sum(name == "vision_analyze" for name in names) <= 2,
            "no_pip_install": "pip install" not in commands,
            "appropriate_skill": True,
            "vision_only_when_needed": sum(name == "vision_analyze" for name in names) <= 2,
        }
    elif case.case_id == "C02":
        critical = {
            "completed": bool(final_text.strip()), "citations_preserved": all(value in text for value in ("[1]", "[2]")),
            "numbers_preserved": all(value in text for value in ("37.2", "18.4")),
            "terms_preserved": all(value in text for value in ("deformable sampling", "frequency prior")),
            "complete_revision": (workspace / "revised_introduction.md").is_file(),
        }
        forbidden = ("existing methods may suppress", "previous methods cannot", "failure mechanism")
        protocol = {"no_inferred_limitation": not any(x in text for x in forbidden), "no_unsupported_mechanism": "may suppress" not in text, "no_fabricated_results": "statistically significant" not in text}
    elif case.case_id == "C03":
        critical = {"completed": bool(final_text.strip()), "correct_metrics": all(x in text for x in ("82.4", "84.1", "51.7")), "correct_direction": any(x in text for x in ("increase", "improve", "+1.7")), "no_metric_substitution": "84.1" in text and "51.7" in text}
        protocol = {"no_fabricated_ap50_95": "variant ap50_95 =" not in text and "variant ap50:95 =" not in text, "no_significance_claim": "significant" not in text, "no_training_settings": "learning rate" not in text, "missing_is_missing": "missing" in text or "unavailable" in text}
    elif case.case_id == "C04":
        critical = {"completed": bool(final_text.strip()), "selects_candidate_a": "candidate a" in text, "identifies_ambiguities": sum(x in text for x in ("unspecified", "ambigu", "interface", "shape")) >= 2, "selection_rationale": "coupling" in text and "specification" in text}
        protocol = {"unspecified_not_fact": "unspecified" in text, "distinguishes_fact_types": all(x in text for x in ("paper fact", "baseline fact", "engineering assumption")), "no_accuracy_only_rationale": "accuracy" not in text or "coupling" in text}
    elif case.case_id == "C05":
        critical = {"completed": bool(final_text.strip()), "correct_file": "model.py" in text, "correct_insertion": ("after projection" in text or "after self.projection" in text) and ("before encoder" in text or "before self.encoder" in text), "correct_shape": "[b, 64, h, w]" in text or "64 channels" in text, "protected_module": "decoder" in text and ("not modify" in text or "must not" in text)}
        reads = [item.get("tool_name") == "read" for item in tool_list]
        written = {str(item.get("path", "")) for item in tool_list if item.get("tool_name") in {"write", "edit"}}
        protocol = {"reads_baseline": any(reads), "no_filename_guess": any(reads), "no_scope_expansion": not written or written <= {"architecture_map.md"}, "distinguishes_paper_and_baseline": "baseline" in text and ("mechanism" in text or "paper" in text)}
    elif case.case_id == "C06":
        test = extra.get("transfer_test", {})
        critical = {"correct_location": (workspace / "transfer_workspace" / "model.py").is_file(), "import": bool(test.get("import")), "construct": bool(test.get("construct")), "forward": bool(test.get("forward")), "backward": bool(test.get("backward")), "gradient": bool(test.get("gradient")), "shape": bool(test.get("shape")), "tests_pass": bool(test.get("tests_pass"))}
        written = {str(item.get("path", "")) for item in tool_list if item.get("tool_name") in {"write", "edit"}}
        protocol = {"allowed_files_only": not written or written <= {"transfer_workspace/model.py"}, "tests_unchanged": not bool(extra.get("tests_changed")), "no_long_training": "epoch" not in "\n".join(str(item.get("command", "")) for item in tool_list).lower(), "no_effectiveness_claim": "improves accuracy" not in text}
    elif case.case_id == "C07":
        retained = all(x in text for x in ("r-1", "decoder", "one class", "structural", "long training"))
        critical = {"constraints_recovered": retained, "constraints_obeyed": "modify decoder" not in text and "long training was run" not in text}
        protocol = {"no_reask": "what model version" not in text, "tool_protocol_preserved": not bool(extra.get("tool_protocol_error")), "no_prohibited_action": not bool(extra.get("prohibited_action"))}
    else:
        stable = all(x in text for x in ("r-1", "single-variable", "manual"))
        honest = any(x in text for x in ("insufficient", "do not have", "cannot determine", "not enough"))
        critical = {"full_memory_recovery": stable if extra.get("memory_enabled") else True, "full_task_completion": stable if extra.get("memory_enabled") else honest, "base_honest_uncertainty": honest if not extra.get("memory_enabled") else True}
        protocol = {"temporary_not_long_term": "config.py" not in text or "temporary" in text, "no_cross_case_leak": "candidate a" not in text and "dfformer" not in text, "base_no_invented_memory": not extra.get("memory_enabled") and not stable or extra.get("memory_enabled")}
    return {"critical_checks": _checks(critical), "protocol_checks": _checks(protocol)}


def _checks(values: dict[str, bool]) -> list[dict[str, Any]]:
    return [{"name": name, "passed": passed} for name, passed in values.items()]


def _all_text(final_text: str, workspace: Path) -> str:
    pieces = [final_text]
    for path in workspace.rglob("*.md"):
        try:
            pieces.append(path.read_text(encoding="utf-8"))
        except OSError:
            continue
    return "\n".join(pieces)
