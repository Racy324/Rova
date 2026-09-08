from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from rova.app.skill_candidate_review import CandidateReviewService
from rova.app.skill_candidates import (
    CandidateMaterializer,
    CandidateState,
    FileSkillCandidateStore,
)
from rova.app.skill_proposals import FileSkillProposalStore, SkillProposal
from rova.app.skill_promotion import SkillPromotionError, SkillPromotionService
from rova.app.skills import FileSkillStore


def _skill_markdown(name: str, description: str = "Test skill") -> str:
    return (
        "---\n"
        f"name: {name}\n"
        f"description: {description}\n"
        "---\n\n"
        f"# {name}\n"
    )


def _make_services(
    tmp_path: Path,
) -> tuple[
    FileSkillStore,
    FileSkillProposalStore,
    FileSkillCandidateStore,
    CandidateMaterializer,
    SkillPromotionService,
]:
    active_store = FileSkillStore(tmp_path / "skills")
    proposal_store = FileSkillProposalStore(tmp_path / "skill-proposals")
    candidate_store = FileSkillCandidateStore(tmp_path / "skill-candidates")
    review_service = CandidateReviewService(
        candidate_store=candidate_store,
        active_skill_store=active_store,
    )
    return (
        active_store,
        proposal_store,
        candidate_store,
        CandidateMaterializer(
            proposal_store=proposal_store,
            active_skill_store=active_store,
            candidate_store=candidate_store,
        ),
        SkillPromotionService(
            candidate_store=candidate_store,
            active_skill_store=active_store,
            review_service=review_service,
        ),
    )


def _tree_digest(root: Path) -> dict[str, str]:
    if not root.exists():
        return {}
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*")
        if path.is_file()
    }


def test_promote_create_writes_active_skill_then_marks_candidate_promoted(
    tmp_path: Path,
) -> None:
    active_store, proposal_store, candidate_store, materializer, service = _make_services(
        tmp_path
    )
    content = _skill_markdown("paper-review", "Candidate workflow")
    proposal = proposal_store.save(
        1,
        [
            SkillProposal(
                action="create",
                name="paper-review",
                content=content,
                rationale="Add a reusable workflow.",
            )
        ],
    )[0]
    candidate = materializer.materialize(proposal.proposal_id)

    promoted = service.promote(candidate.candidate_id, confirmed=True)

    assert active_store.read_main_document("paper-review") == content
    assert promoted.state is CandidateState.PROMOTED
    assert promoted.resolved_at
    assert promoted.rejection_reason is None
    assert candidate_store.read(candidate.candidate_id) == promoted


def test_promote_patch_replaces_only_active_main_document_after_fresh_baseline_check(
    tmp_path: Path,
) -> None:
    active_store, proposal_store, candidate_store, materializer, service = _make_services(
        tmp_path
    )
    original = _skill_markdown("paper-review", "Original workflow")
    candidate_content = _skill_markdown("paper-review", "Candidate workflow")
    active_store.create("paper-review", original)
    supporting_file = active_store.root / "paper-review" / "references" / "notes.md"
    supporting_file.parent.mkdir()
    supporting_file.write_text("Keep this file.", encoding="utf-8")
    proposal = proposal_store.save(
        1,
        [
            SkillProposal(
                action="patch",
                name="paper-review",
                content=candidate_content,
                rationale="Clarify the workflow.",
            )
        ],
    )[0]
    candidate = materializer.materialize(proposal.proposal_id)

    promoted = service.promote(candidate.candidate_id, confirmed=True)

    assert active_store.read_main_document("paper-review") == candidate_content
    assert supporting_file.read_text(encoding="utf-8") == "Keep this file."
    assert promoted.state is CandidateState.PROMOTED
    assert candidate_store.read(candidate.candidate_id) == promoted


def test_reject_marks_ready_candidate_without_modifying_active_or_proposal(
    tmp_path: Path,
) -> None:
    active_store, proposal_store, candidate_store, materializer, service = _make_services(
        tmp_path
    )
    proposal = proposal_store.save(
        1,
        [
            SkillProposal(
                action="create",
                name="paper-review",
                content=_skill_markdown("paper-review"),
                rationale="Add a reusable workflow.",
            )
        ],
    )[0]
    candidate = materializer.materialize(proposal.proposal_id)
    active_before = _tree_digest(active_store.root)
    proposal_before = _tree_digest(proposal_store.root)

    rejected = service.reject(candidate.candidate_id, reason="Needs a clearer scope.")

    assert rejected.state is CandidateState.REJECTED
    assert rejected.resolved_at
    assert rejected.rejection_reason == "Needs a clearer scope."
    assert candidate_store.read(candidate.candidate_id) == rejected
    assert _tree_digest(active_store.root) == active_before
    assert _tree_digest(proposal_store.root) == proposal_before


def test_promote_requires_explicit_confirmation_without_changing_any_state(
    tmp_path: Path,
) -> None:
    active_store, proposal_store, candidate_store, materializer, service = _make_services(
        tmp_path
    )
    proposal = proposal_store.save(
        1,
        [
            SkillProposal(
                action="create",
                name="paper-review",
                content=_skill_markdown("paper-review"),
                rationale="Add a reusable workflow.",
            )
        ],
    )[0]
    candidate = materializer.materialize(proposal.proposal_id)
    active_before = _tree_digest(active_store.root)
    candidate_before = _tree_digest(candidate_store.root)

    with pytest.raises(SkillPromotionError, match="confirmation"):
        service.promote(candidate.candidate_id, confirmed=False)

    assert _tree_digest(active_store.root) == active_before
    assert _tree_digest(candidate_store.root) == candidate_before
    assert candidate_store.read(candidate.candidate_id).state is CandidateState.READY


def test_promote_create_fails_closed_when_active_target_appears_after_materialization(
    tmp_path: Path,
) -> None:
    active_store, proposal_store, candidate_store, materializer, service = _make_services(
        tmp_path
    )
    proposal = proposal_store.save(
        1,
        [
            SkillProposal(
                action="create",
                name="paper-review",
                content=_skill_markdown("paper-review", "Candidate"),
                rationale="Add a reusable workflow.",
            )
        ],
    )[0]
    candidate = materializer.materialize(proposal.proposal_id)
    active_store.create("paper-review", _skill_markdown("paper-review", "Existing"))
    active_before = _tree_digest(active_store.root)
    candidate_before = _tree_digest(candidate_store.root)

    with pytest.raises(SkillPromotionError, match="already exists"):
        service.promote(candidate.candidate_id, confirmed=True)

    assert _tree_digest(active_store.root) == active_before
    assert _tree_digest(candidate_store.root) == candidate_before


@pytest.mark.parametrize("change", ["remove", "modify"])
def test_promote_patch_fails_closed_when_target_is_missing_or_baseline_is_stale(
    tmp_path: Path, change: str
) -> None:
    active_store, proposal_store, candidate_store, materializer, service = _make_services(
        tmp_path
    )
    active_store.create("paper-review", _skill_markdown("paper-review", "Original"))
    proposal = proposal_store.save(
        1,
        [
            SkillProposal(
                action="patch",
                name="paper-review",
                content=_skill_markdown("paper-review", "Candidate"),
                rationale="Update the workflow.",
            )
        ],
    )[0]
    candidate = materializer.materialize(proposal.proposal_id)
    if change == "remove":
        active_store.delete("paper-review")
    else:
        active_store.edit("paper-review", _skill_markdown("paper-review", "Changed"))
    active_before = _tree_digest(active_store.root)
    candidate_before = _tree_digest(candidate_store.root)

    with pytest.raises(SkillPromotionError, match="baseline"):
        service.promote(candidate.candidate_id, confirmed=True)

    assert _tree_digest(active_store.root) == active_before
    assert _tree_digest(candidate_store.root) == candidate_before


def test_promote_rejects_invalid_candidate_document_without_writing_active_or_state(
    tmp_path: Path,
) -> None:
    active_store, proposal_store, candidate_store, materializer, service = _make_services(
        tmp_path
    )
    proposal = proposal_store.save(
        1,
        [
            SkillProposal(
                action="create",
                name="paper-review",
                content=_skill_markdown("paper-review"),
                rationale="Add a reusable workflow.",
            )
        ],
    )[0]
    candidate = materializer.materialize(proposal.proposal_id)
    content_path = (
        candidate_store.root / candidate.candidate_id / "skill" / candidate.name / "SKILL.md"
    )
    content_path.write_text("# malformed\n", encoding="utf-8")
    active_before = _tree_digest(active_store.root)
    candidate_before = _tree_digest(candidate_store.root)

    with pytest.raises(SkillPromotionError, match="validation"):
        service.promote(candidate.candidate_id, confirmed=True)

    assert _tree_digest(active_store.root) == active_before
    assert _tree_digest(candidate_store.root) == candidate_before
    assert candidate_store.read_document(candidate.candidate_id).candidate.state is CandidateState.READY


def test_candidate_cannot_be_promoted_or_rejected_after_terminal_transition(
    tmp_path: Path,
) -> None:
    _, proposal_store, candidate_store, materializer, service = _make_services(tmp_path)
    proposal = proposal_store.save(
        1,
        [
            SkillProposal(
                action="create",
                name="paper-review",
                content=_skill_markdown("paper-review"),
                rationale="Add a reusable workflow.",
            )
        ],
    )[0]
    candidate = materializer.materialize(proposal.proposal_id)
    promoted = service.promote(candidate.candidate_id, confirmed=True)
    candidate_before = _tree_digest(candidate_store.root)

    with pytest.raises(SkillPromotionError, match="not ready"):
        service.promote(candidate.candidate_id, confirmed=True)
    with pytest.raises(SkillPromotionError, match="not ready"):
        service.reject(candidate.candidate_id, reason="Too late")

    assert _tree_digest(candidate_store.root) == candidate_before
    assert candidate_store.read(candidate.candidate_id) == promoted


@pytest.mark.parametrize("reason", ["", "   "])
def test_reject_requires_a_non_empty_reason(tmp_path: Path, reason: str) -> None:
    _, proposal_store, candidate_store, materializer, service = _make_services(tmp_path)
    proposal = proposal_store.save(
        1,
        [
            SkillProposal(
                action="create",
                name="paper-review",
                content=_skill_markdown("paper-review"),
                rationale="Add a reusable workflow.",
            )
        ],
    )[0]
    candidate = materializer.materialize(proposal.proposal_id)
    before = _tree_digest(candidate_store.root)

    with pytest.raises(SkillPromotionError, match="non-empty"):
        service.reject(candidate.candidate_id, reason=reason)

    assert _tree_digest(candidate_store.root) == before
