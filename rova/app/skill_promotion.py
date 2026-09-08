from __future__ import annotations

from datetime import datetime, timezone

from .skill_candidate_review import (
    CandidateReviewService,
    CandidateTargetStatus,
)
from .skill_candidates import CandidateState, FileSkillCandidateStore, SkillCandidate, SkillCandidateStoreError
from .skills import FileSkillStore, SkillStoreError


class SkillPromotionError(RuntimeError):
    """A Candidate did not satisfy the explicit promotion/rejection gate."""


class SkillPromotionService:
    """Applies explicit user decisions to isolated Candidates without automatic promotion."""

    def __init__(
        self,
        *,
        candidate_store: FileSkillCandidateStore,
        active_skill_store: FileSkillStore,
        review_service: CandidateReviewService,
    ) -> None:
        self._candidate_store = candidate_store
        self._active_skill_store = active_skill_store
        self._review_service = review_service

    def promote(self, candidate_id: str, *, confirmed: bool) -> SkillCandidate:
        review = self._review_service.review(candidate_id)
        candidate = review.candidate
        self._require_ready(candidate)
        if confirmed is not True:
            raise SkillPromotionError("explicit user confirmation is required")
        if review.validation_errors:
            raise SkillPromotionError("Candidate document validation failed")
        try:
            if candidate.action == "create":
                if review.target_status is not CandidateTargetStatus.CREATE_TARGET_ABSENT:
                    raise SkillPromotionError("Active Skill target already exists")
                self._active_skill_store.create(candidate.name, review.content)
            elif candidate.action == "patch":
                if review.target_status is not CandidateTargetStatus.PATCH_TARGET_PRESENT:
                    raise SkillPromotionError("Active Skill target is not at the Candidate baseline")
                self._active_skill_store.edit(
                    candidate.name,
                    review.content,
                    expected_main_document_sha256=candidate.active_baseline_sha256,
                )
            else:
                raise SkillPromotionError("Candidate action is invalid")
        except SkillStoreError as error:
            raise SkillPromotionError("could not promote Candidate to Active Skill") from error
        return self._transition(candidate, CandidateState.PROMOTED, rejection_reason=None)

    def reject(self, candidate_id: str, *, reason: str) -> SkillCandidate:
        if not isinstance(reason, str) or not reason.strip():
            raise SkillPromotionError("a non-empty rejection reason is required")
        try:
            candidate = self._candidate_store.read_document(candidate_id).candidate
        except SkillCandidateStoreError as error:
            raise SkillPromotionError("could not read Candidate") from error
        self._require_ready(candidate)
        return self._transition(
            candidate,
            CandidateState.REJECTED,
            rejection_reason=reason.strip(),
        )

    def _transition(
        self,
        candidate: SkillCandidate,
        state: CandidateState,
        *,
        rejection_reason: str | None,
    ) -> SkillCandidate:
        try:
            return self._candidate_store.transition(
                candidate.candidate_id,
                state=state,
                resolved_at=datetime.now(timezone.utc).isoformat(),
                rejection_reason=rejection_reason,
            )
        except SkillCandidateStoreError as error:
            raise SkillPromotionError("could not update Candidate state") from error

    @staticmethod
    def _require_ready(candidate: SkillCandidate) -> None:
        if candidate.state is not CandidateState.READY:
            raise SkillPromotionError("Candidate is not ready")
