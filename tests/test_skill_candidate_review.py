from __future__ import annotations

import hashlib
from pathlib import Path

from rova.app.skill_candidate_review import CandidateReviewService, CandidateTargetStatus
from rova.app.skill_candidates import CandidateMaterializer, FileSkillCandidateStore
from rova.app.skill_proposals import FileSkillProposalStore, SkillProposal
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
    CandidateReviewService,
]:
    active_store = FileSkillStore(tmp_path / "skills")
    proposal_store = FileSkillProposalStore(tmp_path / "skill-proposals")
    candidate_store = FileSkillCandidateStore(tmp_path / "skill-candidates")
    return (
        active_store,
        proposal_store,
        candidate_store,
        CandidateMaterializer(
            proposal_store=proposal_store,
            active_skill_store=active_store,
            candidate_store=candidate_store,
        ),
        CandidateReviewService(
            candidate_store=candidate_store,
            active_skill_store=active_store,
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


def test_review_returns_complete_patch_candidate_and_preserves_all_stores(
    tmp_path: Path,
) -> None:
    active_store, proposal_store, candidate_store, materializer, service = _make_services(
        tmp_path
    )
    active_store.create("paper-review", _skill_markdown("paper-review", "Original"))
    proposal = proposal_store.save(
        4,
        [
            SkillProposal(
                action="patch",
                name="paper-review",
                content=_skill_markdown("paper-review", "Candidate revision"),
                rationale="Clarify the review sequence.",
            )
        ],
    )[0]
    candidate = materializer.materialize(proposal.proposal_id)
    before = {
        "active": _tree_digest(active_store.root),
        "proposal": _tree_digest(proposal_store.root),
        "candidate": _tree_digest(candidate_store.root),
    }

    review = service.review(candidate.candidate_id)

    assert review.candidate == candidate
    assert review.proposal_rationale == "Clarify the review sequence."
    assert review.content == _skill_markdown("paper-review", "Candidate revision")
    assert review.target_status is CandidateTargetStatus.PATCH_TARGET_PRESENT
    assert review.validation_errors == ()
    assert {
        "active": _tree_digest(active_store.root),
        "proposal": _tree_digest(proposal_store.root),
        "candidate": _tree_digest(candidate_store.root),
    } == before


def test_review_reports_create_target_absent_or_exists(tmp_path: Path) -> None:
    active_store, proposal_store, _, materializer, service = _make_services(tmp_path)
    proposal = proposal_store.save(
        1,
        [
            SkillProposal(
                action="create",
                name="paper-review",
                content=_skill_markdown("paper-review"),
                rationale="Add the workflow.",
            )
        ],
    )[0]
    candidate = materializer.materialize(proposal.proposal_id)

    assert service.review(candidate.candidate_id).target_status is (
        CandidateTargetStatus.CREATE_TARGET_ABSENT
    )

    active_store.create("paper-review", _skill_markdown("paper-review", "Existing"))

    assert service.review(candidate.candidate_id).target_status is (
        CandidateTargetStatus.CREATE_TARGET_EXISTS
    )


def test_review_reports_patch_target_missing_after_active_skill_is_removed(
    tmp_path: Path,
) -> None:
    active_store, proposal_store, _, materializer, service = _make_services(tmp_path)
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
    active_store.delete("paper-review")

    review = service.review(candidate.candidate_id)

    assert review.target_status is CandidateTargetStatus.PATCH_TARGET_MISSING
    assert review.validation_errors == ()


def test_review_reports_patch_baseline_changed_after_active_main_document_changes(
    tmp_path: Path,
) -> None:
    active_store, proposal_store, _, materializer, service = _make_services(tmp_path)
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
    active_store.edit("paper-review", _skill_markdown("paper-review", "Changed"))

    review = service.review(candidate.candidate_id)

    assert review.target_status is CandidateTargetStatus.PATCH_BASELINE_CHANGED
    assert review.validation_errors == ()


def test_review_reports_malformed_candidate_document_without_writing_any_store(
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
                rationale="Add the workflow.",
            )
        ],
    )[0]
    candidate = materializer.materialize(proposal.proposal_id)
    content_path = (
        candidate_store.root / candidate.candidate_id / "skill" / candidate.name / "SKILL.md"
    )
    content_path.write_text("# malformed\n", encoding="utf-8")
    before = {
        "active": _tree_digest(active_store.root),
        "proposal": _tree_digest(proposal_store.root),
        "candidate": _tree_digest(candidate_store.root),
    }

    review = service.review(candidate.candidate_id)

    assert review.content == "# malformed\n"
    assert "SKILL.md must begin with frontmatter" in review.validation_errors
    assert "Candidate SKILL.md hash does not match metadata" in review.validation_errors
    assert {
        "active": _tree_digest(active_store.root),
        "proposal": _tree_digest(proposal_store.root),
        "candidate": _tree_digest(candidate_store.root),
    } == before
