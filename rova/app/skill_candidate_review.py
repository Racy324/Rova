from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib

from .skill_candidates import FileSkillCandidateStore, SkillCandidate
from .skills import FileSkillStore, SkillStoreError, validate_skill_document


class CandidateTargetStatus(str, Enum):
    CREATE_TARGET_ABSENT = "target_absent"
    CREATE_TARGET_EXISTS = "target_exists"
    PATCH_TARGET_PRESENT = "target_present"
    PATCH_TARGET_MISSING = "target_missing"
    PATCH_BASELINE_CHANGED = "baseline_changed"


@dataclass(frozen=True)
class CandidateReview:
    candidate: SkillCandidate
    proposal_rationale: str
    content: str
    target_status: CandidateTargetStatus
    validation_errors: tuple[str, ...]


class CandidateReviewService:
    """Read-only, deterministic validation for one isolated Skill Candidate."""

    def __init__(
        self,
        *,
        candidate_store: FileSkillCandidateStore,
        active_skill_store: FileSkillStore,
    ) -> None:
        self._candidate_store = candidate_store
        self._active_skill_store = active_skill_store

    def review(self, candidate_id: str) -> CandidateReview:
        document = self._candidate_store.read_document(candidate_id)
        candidate = document.candidate
        return CandidateReview(
            candidate=candidate,
            proposal_rationale=candidate.rationale,
            content=document.content,
            target_status=self._target_status(candidate),
            validation_errors=_document_validation_errors(candidate, document.content),
        )

    def _target_status(self, candidate: SkillCandidate) -> CandidateTargetStatus:
        if candidate.action == "create":
            target = self._active_skill_store.root / candidate.name
            return (
                CandidateTargetStatus.CREATE_TARGET_EXISTS
                if target.exists() or target.is_symlink()
                else CandidateTargetStatus.CREATE_TARGET_ABSENT
            )
        try:
            active_content = self._active_skill_store.read_main_document(candidate.name)
        except SkillStoreError:
            return CandidateTargetStatus.PATCH_TARGET_MISSING
        if _sha256(active_content) != candidate.active_baseline_sha256:
            return CandidateTargetStatus.PATCH_BASELINE_CHANGED
        return CandidateTargetStatus.PATCH_TARGET_PRESENT


def _document_validation_errors(candidate: SkillCandidate, content: str) -> tuple[str, ...]:
    errors: list[str] = []
    try:
        normalized = validate_skill_document(candidate.name, content)
    except SkillStoreError as error:
        errors.append(str(error))
    else:
        if content != normalized:
            errors.append("Candidate SKILL.md is not normalized")
    if _sha256(content) != candidate.content_sha256:
        errors.append("Candidate SKILL.md hash does not match metadata")
    return tuple(errors)


def _sha256(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()
