from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import time
from typing import Literal
from uuid import uuid4

from .file_lock import FileLock, FileLockError
from .paths import RovaDataPaths


_SCHEMA_VERSION = 1
_PROPOSAL_ID = re.compile(r"^[0-9]{20}-[a-f0-9]{32}$")


class SkillProposalStoreError(RuntimeError):
    """Expected failure while persisting pending Skill proposals."""


@dataclass(frozen=True)
class SkillProposal:
    action: Literal["create", "patch"]
    name: str
    content: str
    rationale: str


@dataclass(frozen=True)
class PendingSkillProposal:
    proposal_id: str
    review_generation: int
    created_at: str
    action: Literal["create", "patch"]
    name: str
    content: str
    rationale: str


class FileSkillProposalStore:
    """Append-only local store for pending proposals; it never writes Active Skills."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = RovaDataPaths.resolve().skill_proposals if root is None else Path(root)

    def save(
        self,
        generation: int,
        proposals: Sequence[SkillProposal],
    ) -> tuple[PendingSkillProposal, ...]:
        _non_negative_int(generation, "generation")
        timestamp = time.time_ns()
        prepared = tuple(
            _pending_proposal(generation, proposal, timestamp=timestamp + index)
            for index, proposal in enumerate(proposals)
        )
        if not prepared:
            return ()
        self._ensure_root()
        try:
            with FileLock(self.root / ".skill-proposals.lock"):
                for proposal in prepared:
                    _write_json_atomically(self.root / f"{proposal.proposal_id}.json", _proposal_to_json(proposal))
            return prepared
        except FileLockError as error:
            raise SkillProposalStoreError("could not acquire skill proposal file lock") from error
        except (OSError, TypeError, ValueError) as error:
            raise SkillProposalStoreError("could not persist Skill proposal") from error

    def list(self) -> tuple[PendingSkillProposal, ...]:
        self._ensure_root()
        try:
            with FileLock(self.root / ".skill-proposals.lock"):
                paths = sorted(self.root.glob("*.json"), key=lambda path: path.name)
                return tuple(_read_proposal(path) for path in paths)
        except FileLockError as error:
            raise SkillProposalStoreError("could not acquire skill proposal file lock") from error
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as error:
            raise SkillProposalStoreError("Skill proposal store is invalid") from error

    def read(self, proposal_id: str) -> PendingSkillProposal:
        if not isinstance(proposal_id, str) or not _PROPOSAL_ID.fullmatch(proposal_id):
            raise SkillProposalStoreError("invalid Skill proposal id")
        self._ensure_root()
        try:
            with FileLock(self.root / ".skill-proposals.lock"):
                return _read_proposal(self.root / f"{proposal_id}.json")
        except FileLockError as error:
            raise SkillProposalStoreError("could not acquire skill proposal file lock") from error
        except FileNotFoundError as error:
            raise SkillProposalStoreError("Skill proposal does not exist") from error
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as error:
            raise SkillProposalStoreError("Skill proposal is invalid") from error

    def _ensure_root(self) -> None:
        try:
            self.root.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            raise SkillProposalStoreError("could not create Skill proposal directory") from error
        try:
            self.root.chmod(0o700)
        except OSError:
            pass


def _pending_proposal(generation: int, proposal: SkillProposal, *, timestamp: int) -> PendingSkillProposal:
    _validate_skill_proposal(proposal)
    proposal_id = f"{timestamp:020d}-{uuid4().hex}"
    return PendingSkillProposal(
        proposal_id=proposal_id,
        review_generation=generation,
        created_at=datetime.now(timezone.utc).isoformat(),
        action=proposal.action,
        name=proposal.name,
        content=proposal.content,
        rationale=proposal.rationale,
    )


def _validate_skill_proposal(proposal: SkillProposal) -> None:
    if not isinstance(proposal, SkillProposal):
        raise ValueError("Skill proposal is invalid")
    if proposal.action not in {"create", "patch"}:
        raise ValueError("Skill proposal action is invalid")
    if not all(isinstance(value, str) and value.strip() for value in (proposal.name, proposal.content, proposal.rationale)):
        raise ValueError("Skill proposal fields are invalid")


def _proposal_to_json(proposal: PendingSkillProposal) -> dict[str, object]:
    return {"schema_version": _SCHEMA_VERSION, **asdict(proposal)}


def _read_proposal(path: Path) -> PendingSkillProposal:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or set(raw) != {
        "schema_version",
        "proposal_id",
        "review_generation",
        "created_at",
        "action",
        "name",
        "content",
        "rationale",
    }:
        raise ValueError("Skill proposal is invalid")
    if raw.get("schema_version") != _SCHEMA_VERSION:
        raise ValueError("unsupported Skill proposal")
    proposal_id = raw.get("proposal_id")
    if not isinstance(proposal_id, str) or not _PROPOSAL_ID.fullmatch(proposal_id):
        raise ValueError("Skill proposal id is invalid")
    review_generation = _non_negative_int(raw.get("review_generation"), "review_generation")
    created_at = raw.get("created_at")
    action = raw.get("action")
    name = raw.get("name")
    content = raw.get("content")
    rationale = raw.get("rationale")
    if not isinstance(created_at, str) or not created_at:
        raise ValueError("Skill proposal created_at is invalid")
    proposal = SkillProposal(action, name, content, rationale)
    _validate_skill_proposal(proposal)
    return PendingSkillProposal(proposal_id, review_generation, created_at, action, name, content, rationale)


def _non_negative_int(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _write_json_atomically(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
