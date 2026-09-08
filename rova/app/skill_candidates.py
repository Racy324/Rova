from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import time
from typing import Literal
from uuid import uuid4

from .file_lock import FileLock, FileLockError
from .paths import RovaDataPaths
from .skill_proposals import FileSkillProposalStore, PendingSkillProposal, SkillProposalStoreError
from .skills import FileSkillStore, SkillStoreError, validate_skill_document


_SCHEMA_VERSION = 1
_CANDIDATE_ID = re.compile(r"^[0-9]{20}-[a-f0-9]{32}$")
_SHA256 = re.compile(r"^[a-f0-9]{64}$")
_SKILL_FILENAME = "SKILL.md"


class SkillCandidateStoreError(RuntimeError):
    """Expected failure while materializing or reading an isolated Skill Candidate."""


class CandidateState(str, Enum):
    READY = "ready"
    PROMOTED = "promoted"
    REJECTED = "rejected"


@dataclass(frozen=True)
class SkillCandidate:
    candidate_id: str
    proposal_id: str
    review_generation: int
    created_at: str
    action: Literal["create", "patch"]
    name: str
    rationale: str
    content_sha256: str
    active_baseline_sha256: str | None
    state: CandidateState
    resolved_at: str | None
    rejection_reason: str | None


@dataclass(frozen=True)
class CandidateSkillDocument:
    """Candidate metadata plus its stored main document, without document validation."""

    candidate: SkillCandidate
    content: str


class FileSkillCandidateStore:
    """Persistent isolated Candidates. It never exposes or writes Active Skills."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = RovaDataPaths.resolve().skill_candidates if root is None else Path(root)

    def list(self) -> tuple[SkillCandidate, ...]:
        self._ensure_root()
        try:
            with FileLock(self.root / ".skill-candidates.lock"):
                paths = sorted(
                    (
                        path
                        for path in self.root.iterdir()
                        if (
                            path.is_dir()
                            and not path.is_symlink()
                            and _CANDIDATE_ID.fullmatch(path.name)
                        )
                    ),
                    key=lambda path: path.name,
                )
                return tuple(self._read_candidate(path.name) for path in paths)
        except FileLockError as error:
            raise SkillCandidateStoreError("could not acquire Skill candidate store lock") from error
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as error:
            raise SkillCandidateStoreError("Skill candidate store is invalid") from error

    def read(self, candidate_id: str) -> SkillCandidate:
        _validate_candidate_id(candidate_id)
        self._ensure_root()
        try:
            with FileLock(self.root / ".skill-candidates.lock"):
                return self._read_candidate(candidate_id)
        except FileLockError as error:
            raise SkillCandidateStoreError("could not acquire Skill candidate store lock") from error
        except FileNotFoundError as error:
            raise SkillCandidateStoreError("Skill candidate does not exist") from error
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as error:
            raise SkillCandidateStoreError("Skill candidate is invalid") from error

    def read_content(self, candidate_id: str) -> str:
        candidate = self.read(candidate_id)
        return self._read_candidate_content(candidate)

    def read_document(self, candidate_id: str) -> CandidateSkillDocument:
        """Read a Candidate main document for read-only validation/reporting."""
        _validate_candidate_id(candidate_id)
        self._ensure_root()
        try:
            with FileLock(self.root / ".skill-candidates.lock"):
                candidate = self._read_candidate_metadata(candidate_id)
                return CandidateSkillDocument(candidate, self._read_candidate_content_raw(candidate))
        except FileLockError as error:
            raise SkillCandidateStoreError("could not acquire Skill candidate store lock") from error
        except FileNotFoundError as error:
            raise SkillCandidateStoreError("Skill candidate does not exist") from error
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as error:
            raise SkillCandidateStoreError("Skill candidate is invalid") from error

    def find_by_proposal_id(self, proposal_id: str) -> SkillCandidate | None:
        self._ensure_root()
        try:
            with FileLock(self.root / ".skill-candidates.lock"):
                return self._find_by_proposal_id(proposal_id)
        except FileLockError as error:
            raise SkillCandidateStoreError("could not acquire Skill candidate store lock") from error
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as error:
            raise SkillCandidateStoreError("Skill candidate store is invalid") from error

    def get_or_create(
        self,
        proposal_id: str,
        factory: Callable[[], tuple[SkillCandidate, str]],
    ) -> SkillCandidate:
        self._ensure_root()
        try:
            with FileLock(self.root / ".skill-candidates.lock"):
                existing = self._find_by_proposal_id(proposal_id)
                if existing is not None:
                    return existing
                candidate, content = factory()
                if candidate.proposal_id != proposal_id:
                    raise ValueError("Candidate proposal id does not match")
                self._publish(candidate, content)
                return candidate
        except FileLockError as error:
            raise SkillCandidateStoreError("could not acquire Skill candidate store lock") from error
        except SkillCandidateStoreError:
            raise
        except (OSError, UnicodeDecodeError, TypeError, ValueError) as error:
            raise SkillCandidateStoreError("could not materialize Skill candidate") from error

    def transition(
        self,
        candidate_id: str,
        *,
        state: CandidateState,
        resolved_at: str,
        rejection_reason: str | None,
    ) -> SkillCandidate:
        """Persist the only permitted Candidate state transitions from ready."""
        if state not in {CandidateState.PROMOTED, CandidateState.REJECTED}:
            raise SkillCandidateStoreError("Candidate transition target is invalid")
        self._ensure_root()
        try:
            with FileLock(self.root / ".skill-candidates.lock"):
                candidate = self._read_candidate_metadata(candidate_id)
                if candidate.state is not CandidateState.READY:
                    raise SkillCandidateStoreError("Candidate is no longer ready")
                updated = replace(
                    candidate,
                    state=state,
                    resolved_at=resolved_at,
                    rejection_reason=rejection_reason,
                )
                _validate_candidate(updated)
                _write_json_atomically(
                    self._candidate_directory(candidate_id) / "candidate.json",
                    _candidate_to_json(updated),
                )
                return updated
        except FileLockError as error:
            raise SkillCandidateStoreError("could not acquire Skill candidate store lock") from error
        except SkillCandidateStoreError:
            raise
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as error:
            raise SkillCandidateStoreError("could not update Skill candidate") from error

    def _find_by_proposal_id(self, proposal_id: str) -> SkillCandidate | None:
        for path in sorted(self.root.iterdir(), key=lambda item: item.name):
            if (
                not path.is_dir()
                or path.is_symlink()
                or not _CANDIDATE_ID.fullmatch(path.name)
            ):
                continue
            candidate = self._read_candidate(path.name)
            if candidate.proposal_id == proposal_id:
                return candidate
        return None

    def _read_candidate(self, candidate_id: str) -> SkillCandidate:
        candidate = self._read_candidate_metadata(candidate_id)
        self._read_candidate_content(candidate)
        return candidate

    def _read_candidate_metadata(self, candidate_id: str) -> SkillCandidate:
        _validate_candidate_id(candidate_id)
        directory = self._candidate_directory(candidate_id)
        metadata_path = directory / "candidate.json"
        if metadata_path.is_symlink() or not metadata_path.is_file():
            raise FileNotFoundError(metadata_path)
        raw = json.loads(metadata_path.read_text(encoding="utf-8"))
        candidate = _candidate_from_json(raw)
        if candidate.candidate_id != candidate_id:
            raise ValueError("Skill candidate id does not match its directory")
        return candidate

    def _read_candidate_content(self, candidate: SkillCandidate) -> str:
        content = self._read_candidate_content_raw(candidate)
        normalized = validate_skill_document(candidate.name, content)
        if content != normalized:
            raise ValueError("Candidate SKILL.md is not normalized")
        if _sha256(content) != candidate.content_sha256:
            raise ValueError("Candidate SKILL.md hash does not match metadata")
        return content

    def _read_candidate_content_raw(self, candidate: SkillCandidate) -> str:
        directory = self._candidate_directory(candidate.candidate_id)
        skill_directory = directory / "skill" / candidate.name
        content_path = skill_directory / _SKILL_FILENAME
        if (
            skill_directory.is_symlink()
            or not skill_directory.is_dir()
            or content_path.is_symlink()
            or not content_path.is_file()
        ):
            raise FileNotFoundError(content_path)
        _ensure_descendant(skill_directory, directory)
        _ensure_descendant(content_path, directory)
        return content_path.read_text(encoding="utf-8")

    def _publish(self, candidate: SkillCandidate, content: str) -> None:
        _validate_candidate(candidate)
        normalized = validate_skill_document(candidate.name, content)
        if normalized != content or _sha256(content) != candidate.content_sha256:
            raise ValueError("Candidate SKILL.md does not match metadata")
        final_directory = self.root / candidate.candidate_id
        if final_directory.exists() or final_directory.is_symlink():
            raise ValueError("Skill candidate already exists")
        staging = self.root / f".{candidate.candidate_id}.{uuid4().hex}.staging"
        try:
            skill_path = staging / "skill" / candidate.name / _SKILL_FILENAME
            skill_path.parent.mkdir(parents=True)
            _write_text_durably(skill_path, content)
            _write_json_durably(staging / "candidate.json", _candidate_to_json(candidate))
            os.replace(staging, final_directory)
        finally:
            if staging.exists() or staging.is_symlink():
                shutil.rmtree(staging, ignore_errors=True)

    def _candidate_directory(self, candidate_id: str) -> Path:
        directory = self.root / candidate_id
        if directory.is_symlink() or not directory.is_dir():
            raise FileNotFoundError(directory)
        _ensure_descendant(directory, self.root)
        return directory

    def _ensure_root(self) -> None:
        try:
            self.root.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            raise SkillCandidateStoreError("could not create Skill candidate directory") from error
        try:
            self.root.chmod(0o700)
        except OSError:
            pass


class CandidateMaterializer:
    """Turns one pending proposal into one complete, isolated Candidate."""

    def __init__(
        self,
        *,
        proposal_store: FileSkillProposalStore,
        active_skill_store: FileSkillStore,
        candidate_store: FileSkillCandidateStore,
    ) -> None:
        self._proposal_store = proposal_store
        self._active_skill_store = active_skill_store
        self._candidate_store = candidate_store

    def materialize(self, proposal_id: str) -> SkillCandidate:
        existing = self._candidate_store.find_by_proposal_id(proposal_id)
        if existing is not None:
            return existing
        try:
            proposal = self._proposal_store.read(proposal_id)
        except SkillProposalStoreError as error:
            raise SkillCandidateStoreError("pending Skill proposal is unavailable") from error
        return self._candidate_store.get_or_create(
            proposal_id,
            lambda: self._prepare_candidate(proposal),
        )

    def _prepare_candidate(self, proposal: PendingSkillProposal) -> tuple[SkillCandidate, str]:
        try:
            content = validate_skill_document(proposal.name, proposal.content)
            baseline = self._active_baseline(proposal)
        except SkillStoreError as error:
            raise SkillCandidateStoreError(str(error)) from error
        candidate = SkillCandidate(
            candidate_id=f"{time.time_ns():020d}-{uuid4().hex}",
            proposal_id=proposal.proposal_id,
            review_generation=proposal.review_generation,
            created_at=datetime.now(timezone.utc).isoformat(),
            action=proposal.action,
            name=proposal.name,
            rationale=proposal.rationale,
            content_sha256=_sha256(content),
            active_baseline_sha256=baseline,
            state=CandidateState.READY,
            resolved_at=None,
            rejection_reason=None,
        )
        return candidate, content

    def _active_baseline(self, proposal: PendingSkillProposal) -> str | None:
        if proposal.action == "create":
            target = self._active_skill_store.root / proposal.name
            if target.exists() or target.is_symlink():
                raise SkillCandidateStoreError(f"Active Skill already exists: {proposal.name}")
            return None
        try:
            active_content = self._active_skill_store.read_main_document(proposal.name)
        except SkillStoreError as error:
            raise SkillCandidateStoreError(f"Active Skill does not exist: {proposal.name}") from error
        return _sha256(active_content)


def _candidate_to_json(candidate: SkillCandidate) -> dict[str, object]:
    payload = asdict(candidate)
    payload["state"] = candidate.state.value
    return {"schema_version": _SCHEMA_VERSION, **payload}


def _candidate_from_json(raw: object) -> SkillCandidate:
    if not isinstance(raw, dict) or set(raw) != {
        "schema_version",
        "candidate_id",
        "proposal_id",
        "review_generation",
        "created_at",
        "action",
        "name",
        "rationale",
        "content_sha256",
        "active_baseline_sha256",
        "state",
        "resolved_at",
        "rejection_reason",
    }:
        raise ValueError("Skill candidate metadata is invalid")
    if raw.get("schema_version") != _SCHEMA_VERSION:
        raise ValueError("unsupported Skill candidate")
    candidate = SkillCandidate(
        candidate_id=raw.get("candidate_id"),
        proposal_id=raw.get("proposal_id"),
        review_generation=raw.get("review_generation"),
        created_at=raw.get("created_at"),
        action=raw.get("action"),
        name=raw.get("name"),
        rationale=raw.get("rationale"),
        content_sha256=raw.get("content_sha256"),
        active_baseline_sha256=raw.get("active_baseline_sha256"),
        state=CandidateState(raw.get("state")),
        resolved_at=raw.get("resolved_at"),
        rejection_reason=raw.get("rejection_reason"),
    )
    _validate_candidate(candidate)
    return candidate


def _validate_candidate(candidate: SkillCandidate) -> None:
    _validate_candidate_id(candidate.candidate_id)
    if not isinstance(candidate.proposal_id, str) or not candidate.proposal_id:
        raise ValueError("Candidate proposal id is invalid")
    if not isinstance(candidate.review_generation, int) or isinstance(candidate.review_generation, bool) or candidate.review_generation < 0:
        raise ValueError("Candidate review generation is invalid")
    if not isinstance(candidate.created_at, str) or not candidate.created_at:
        raise ValueError("Candidate created_at is invalid")
    if candidate.action not in {"create", "patch"}:
        raise ValueError("Candidate action is invalid")
    if not isinstance(candidate.rationale, str) or not candidate.rationale.strip():
        raise ValueError("Candidate rationale is invalid")
    validate_skill_document(candidate.name, "---\nname: " + candidate.name + "\ndescription: placeholder\n---\n")
    if not isinstance(candidate.content_sha256, str) or not _SHA256.fullmatch(candidate.content_sha256):
        raise ValueError("Candidate content hash is invalid")
    if candidate.active_baseline_sha256 is not None and (
        not isinstance(candidate.active_baseline_sha256, str)
        or not _SHA256.fullmatch(candidate.active_baseline_sha256)
    ):
        raise ValueError("Candidate active baseline hash is invalid")
    if candidate.action == "create" and candidate.active_baseline_sha256 is not None:
        raise ValueError("Create Candidate cannot have an Active baseline hash")
    if candidate.action == "patch" and candidate.active_baseline_sha256 is None:
        raise ValueError("Patch Candidate requires an Active baseline hash")
    if candidate.state is CandidateState.READY and (candidate.resolved_at is not None or candidate.rejection_reason is not None):
        raise ValueError("Ready Candidate cannot be resolved")
    if candidate.state is CandidateState.PROMOTED and (
        not isinstance(candidate.resolved_at, str)
        or not candidate.resolved_at
        or candidate.rejection_reason is not None
    ):
        raise ValueError("Promoted Candidate resolution is invalid")
    if candidate.state is CandidateState.REJECTED and (
        not isinstance(candidate.resolved_at, str)
        or not candidate.resolved_at
        or not isinstance(candidate.rejection_reason, str)
        or not candidate.rejection_reason.strip()
    ):
        raise ValueError("Rejected Candidate resolution is invalid")


def _validate_candidate_id(candidate_id: object) -> None:
    if not isinstance(candidate_id, str) or not _CANDIDATE_ID.fullmatch(candidate_id):
        raise ValueError("invalid Skill candidate id")


def _sha256(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _ensure_descendant(path: Path, root: Path) -> None:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as error:
        raise ValueError("Skill candidate path is outside the candidate store") from error


def _write_text_durably(path: Path, content: str) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())


def _write_json_durably(path: Path, payload: dict[str, object]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _write_json_atomically(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        _write_json_durably(temporary, payload)
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
