from __future__ import annotations

from dataclasses import dataclass


EVAL_VERSION = "research-eval-v1"


@dataclass(frozen=True)
class ResearchCase:
    case_id: str
    name: str
    description: str
    prompt: str
    critical_checks: tuple[str, ...]
    protocol_checks: tuple[str, ...]
    required_capabilities: tuple[str, ...]
    max_turns: int = 16
    timeout_seconds: int = 240


CASES = (
    ResearchCase(
        "C01", "paper_evidence", "Extract source-grounded paper evidence from one fixed local PDF.",
        "Read only paper.pdf in this workspace. Extract the method and evidence into paper_evidence.md, including source page numbers. Do not invent modules. Finish with a concise source-grounded summary.",
        ("completed", "artifact_exists", "core_facts", "page_anchor", "no_invented_module"),
        ("no_multiple_pdf_parsers", "no_pip_install", "appropriate_skill", "vision_only_when_needed"),
        ("workspace", "shell"),
    ),
    ResearchCase(
        "C02", "constrained_revision", "Revise supplied introduction prose without exceeding evidence.",
        "Read introduction.txt. Produce a complete revised Introduction in revised_introduction.md. Preserve supplied citations, numbers, and core terms. Do not infer limitations of previous methods or invent evidence.",
        ("completed", "citations_preserved", "numbers_preserved", "terms_preserved", "complete_revision"),
        ("no_inferred_limitation", "no_unsupported_mechanism", "no_fabricated_results"),
        ("workspace",),
    ),
    ResearchCase(
        "C03", "evidence_writing", "Write an experimental analysis strictly from fixed metrics.",
        "Read experiment.json and write evidence_analysis.md. State only supported results, calculate the AP50 change, and explicitly mark missing values as unavailable. Do not fabricate statistics, settings, or conclusions.",
        ("completed", "correct_metrics", "correct_direction", "no_metric_substitution"),
        ("no_fabricated_ap50_95", "no_significance_claim", "no_training_settings", "missing_is_missing"),
        ("workspace",),
    ),
    ResearchCase(
        "C04", "candidate_screening", "Screen paper-card candidates for the first controlled transfer.",
        "Read candidates.json. Select the safer first controlled transfer candidate and write candidate_decision.md. Identify implementation ambiguities and distinguish paper facts, baseline facts, and engineering assumptions.",
        ("completed", "selects_candidate_a", "identifies_ambiguities", "selection_rationale"),
        ("unspecified_not_fact", "distinguishes_fact_types", "no_accuracy_only_rationale"),
        ("workspace",),
    ),
    ResearchCase(
        "C05", "architecture_mapping", "Map one paper mechanism into a real toy baseline without editing code.",
        "Inspect toy_detector and mechanism.md. Do not modify any code. Write architecture_map.md naming the exact file, class/function, insertion position, shapes/channels, and modules that must not change. Base claims on files you read.",
        ("completed", "correct_file", "correct_insertion", "correct_shape", "protected_module"),
        ("reads_baseline", "no_filename_guess", "no_scope_expansion", "distinguishes_paper_and_baseline"),
        ("workspace",),
    ),
    ResearchCase(
        "C06", "minimal_transfer", "Implement a minimal PyTorch transfer and structurally verify it.",
        "Inspect transfer_workspace. Implement the requested ResearchAdapter only in transfer_workspace/model.py, positioned between encoder and decoder. Do not edit tests or run long training. Run the provided structural tests and report what was verified.",
        ("correct_location", "import", "construct", "forward", "backward", "gradient", "shape", "tests_pass"),
        ("allowed_files_only", "tests_unchanged", "no_long_training", "no_effectiveness_claim"),
        ("workspace", "shell"),
        max_turns=20,
    ),
    ResearchCase(
        "C07", "long_context", "Maintain research constraints through an automatically compacted session.",
        "This session has fixed constraints: use model version R-1; never modify Decoder; dataset has exactly one class; structural verification only; never run long training. A later request will require these constraints. Acknowledge them briefly.",
        ("constraints_recovered", "constraints_obeyed"),
        ("no_reask", "tool_protocol_preserved", "no_prohibited_action"),
        ("workspace", "session", "compaction"),
        max_turns=20,
    ),
    ResearchCase(
        "C08", "cross_session_memory", "Carry stable research habits into a separate new session.",
        "Record these stable research habits for future work: baseline is R-1; prefer single-variable experiments; the user manually runs long GPU training. Today only inspect config.py; that last item is temporary. A later separate session will ask for transfer principles.",
        ("full_memory_recovery", "full_task_completion", "base_honest_uncertainty"),
        ("temporary_not_long_term", "no_cross_case_leak", "base_no_invented_memory"),
        ("workspace", "session", "memory"),
        max_turns=12,
    ),
)


def get_case(case_id: str) -> ResearchCase:
    try:
        return next(case for case in CASES if case.case_id == case_id)
    except StopIteration as error:
        raise ValueError(f"unknown research eval case: {case_id}") from error


def select_cases(case_ids: tuple[str, ...] | None = None) -> tuple[ResearchCase, ...]:
    if not case_ids:
        return CASES
    requested = set(case_ids)
    unknown = requested - {case.case_id for case in CASES}
    if unknown:
        raise ValueError(f"unknown research eval case ids: {sorted(unknown)}")
    return tuple(case for case in CASES if case.case_id in requested)
