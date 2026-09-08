from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from rova.ai.events import StreamDone
from rova.ai.messages import AssistantMessage, TextBlock
from rova.ai.models import Model
from rova.ai.messages import ToolCall
from rova.agent_core.tools import ToolRegistry, ToolRuntime
from rova.app.runtime import build_rova_runtime
from rova.app.skill_candidates import (
    CandidateMaterializer,
    CandidateState,
    FileSkillCandidateStore,
    SkillCandidateStoreError,
)
from rova.app.skill_proposals import FileSkillProposalStore, SkillProposal
from rova.app.skills import FileSkillStore, create_skill_tools
from rova.app import skill_candidates as skill_candidates_module


def _skill_markdown(name: str, description: str = "Test skill") -> str:
    return (
        "---\n"
        f"name: {name}\n"
        f"description: {description}\n"
        "---\n\n"
        f"# {name}\n"
    )


def _make_materializer(tmp_path: Path) -> tuple[
    FileSkillStore,
    FileSkillProposalStore,
    FileSkillCandidateStore,
    CandidateMaterializer,
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
    )


def test_materialize_create_proposal_writes_isolated_complete_candidate(
    tmp_path: Path,
) -> None:
    active_store, proposal_store, candidate_store, materializer = _make_materializer(
        tmp_path
    )
    proposal = proposal_store.save(
        7,
        [
            SkillProposal(
                action="create",
                name="paper-review",
                content=_skill_markdown("paper-review"),
                rationale="Capture a repeatable review workflow.",
            )
        ],
    )[0]

    candidate = materializer.materialize(proposal.proposal_id)

    assert candidate.proposal_id == proposal.proposal_id
    assert candidate.review_generation == 7
    assert candidate.action == "create"
    assert candidate.state is CandidateState.READY
    assert candidate.name == "paper-review"
    assert candidate.active_baseline_sha256 is None
    assert candidate_store.read(candidate.candidate_id) == candidate
    assert candidate_store.read_content(candidate.candidate_id) == _skill_markdown(
        "paper-review"
    )
    assert not (active_store.root / "paper-review").exists()
    assert "paper-review" not in {
        skill.name for skill in active_store.discover_catalog().skills
    }


def test_materialize_same_proposal_returns_existing_candidate(tmp_path: Path) -> None:
    _, proposal_store, candidate_store, materializer = _make_materializer(tmp_path)
    proposal = proposal_store.save(
        1,
        [
            SkillProposal(
                action="create",
                name="paper-review",
                content=_skill_markdown("paper-review"),
                rationale="Capture a repeatable review workflow.",
            )
        ],
    )[0]

    first = materializer.materialize(proposal.proposal_id)
    second = materializer.materialize(proposal.proposal_id)

    assert second == first
    assert len(candidate_store.list()) == 1


def test_materialize_requires_an_existing_pending_proposal(tmp_path: Path) -> None:
    _, _, _, materializer = _make_materializer(tmp_path)

    with pytest.raises(SkillCandidateStoreError, match="pending Skill proposal is unavailable"):
        materializer.materialize("00000000000000000000-00000000000000000000000000000000")


@pytest.mark.asyncio
async def test_candidate_is_absent_from_new_runtime_catalog_and_system_context(
    tmp_path: Path,
) -> None:
    active_store, proposal_store, _, materializer = _make_materializer(tmp_path)
    proposal = proposal_store.save(
        1,
        [
            SkillProposal(
                action="create",
                name="paper-review",
                content=_skill_markdown("paper-review"),
                rationale="Capture a repeatable review workflow.",
            )
        ],
    )[0]
    materializer.materialize(proposal.proposal_id)
    prompts: list[str] = []

    async def stream(_model, context, _options):
        prompts.append(context.system_prompt)
        yield StreamDone(AssistantMessage([TextBlock("done")]))

    runtime = build_rova_runtime(
        model=Model(provider="mock"),
        stream_fn=stream,
        skill_root=active_store.root,
        experience_review_enabled=False,
        session_root=tmp_path / "sessions",
        artifact_root=tmp_path / "artifacts",
    )
    try:
        await runtime.prompt("hello")
    finally:
        await runtime.close()

    assert runtime.skill_catalog_snapshot.skills == ()
    assert "paper-review" not in prompts[0]
    result = await ToolRuntime(ToolRegistry(create_skill_tools(active_store))).execute(
        ToolCall("candidate-view", "skill_view", {"name": "paper-review"})
    )
    assert result.is_error is True
    assert "not found" in result.text


def test_materialize_patch_records_current_active_skill_baseline_hash(
    tmp_path: Path,
) -> None:
    active_store, proposal_store, candidate_store, materializer = _make_materializer(tmp_path)
    active_store.create("paper-review", _skill_markdown("paper-review", "Original"))
    proposal = proposal_store.save(
        3,
        [
            SkillProposal(
                action="patch",
                name="paper-review",
                content=_skill_markdown("paper-review", "Revised"),
                rationale="Clarify the workflow.",
            )
        ],
    )[0]

    candidate = materializer.materialize(proposal.proposal_id)

    assert candidate.action == "patch"
    assert candidate.active_baseline_sha256 == hashlib.sha256(
        active_store.read_main_document("paper-review").encode("utf-8")
    ).hexdigest()
    assert candidate_store.read_content(candidate.candidate_id) == _skill_markdown(
        "paper-review", "Revised"
    )
    assert active_store.read("paper-review") == _skill_markdown("paper-review", "Original")


def test_materialize_create_rejects_active_collision(tmp_path: Path) -> None:
    active_store, proposal_store, candidate_store, materializer = _make_materializer(tmp_path)
    active_store.create("paper-review", _skill_markdown("paper-review"))
    proposal = proposal_store.save(
        1,
        [
            SkillProposal(
                action="create",
                name="paper-review",
                content=_skill_markdown("paper-review", "Candidate"),
                rationale="Conflicting create.",
            )
        ],
    )[0]

    with pytest.raises(SkillCandidateStoreError, match="already exists"):
        materializer.materialize(proposal.proposal_id)

    assert candidate_store.list() == ()
    assert active_store.read("paper-review") == _skill_markdown("paper-review")


def test_materialize_patch_rejects_missing_active_target(tmp_path: Path) -> None:
    _, proposal_store, candidate_store, materializer = _make_materializer(tmp_path)
    proposal = proposal_store.save(
        1,
        [
            SkillProposal(
                action="patch",
                name="paper-review",
                content=_skill_markdown("paper-review"),
                rationale="Target does not exist.",
            )
        ],
    )[0]

    with pytest.raises(SkillCandidateStoreError, match="does not exist"):
        materializer.materialize(proposal.proposal_id)

    assert candidate_store.list() == ()


def test_candidate_read_detects_skill_content_hash_tampering(tmp_path: Path) -> None:
    _, proposal_store, candidate_store, materializer = _make_materializer(tmp_path)
    proposal = proposal_store.save(
        1,
        [
            SkillProposal(
                action="create",
                name="paper-review",
                content=_skill_markdown("paper-review"),
                rationale="Capture a repeatable review workflow.",
            )
        ],
    )[0]
    candidate = materializer.materialize(proposal.proposal_id)
    content_path = (
        candidate_store.root / candidate.candidate_id / "skill" / "paper-review" / "SKILL.md"
    )
    content_path.write_text(_skill_markdown("paper-review", "Tampered"), encoding="utf-8")

    with pytest.raises(SkillCandidateStoreError, match="invalid"):
        candidate_store.read(candidate.candidate_id)


def test_materialize_publish_failure_removes_staging_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, proposal_store, candidate_store, materializer = _make_materializer(tmp_path)
    proposal = proposal_store.save(
        1,
        [
            SkillProposal(
                action="create",
                name="paper-review",
                content=_skill_markdown("paper-review"),
                rationale="Capture a repeatable review workflow.",
            )
        ],
    )[0]

    def fail_publish(_source: Path, _destination: Path) -> None:
        raise OSError("publish failed")

    monkeypatch.setattr(skill_candidates_module.os, "replace", fail_publish)

    with pytest.raises(SkillCandidateStoreError, match="could not materialize"):
        materializer.materialize(proposal.proposal_id)

    assert candidate_store.list() == ()
    assert not [
        path
        for path in candidate_store.root.iterdir()
        if path.name != ".skill-candidates.lock"
    ]


def test_materialize_rejects_invalid_candidate_without_publishing_partial_directory(
    tmp_path: Path,
) -> None:
    _, proposal_store, candidate_store, materializer = _make_materializer(tmp_path)
    proposal = proposal_store.save(
        1,
        [
            SkillProposal(
                action="create",
                name="paper-review",
                content="# Missing frontmatter\n",
                rationale="Invalid candidate body.",
            )
        ],
    )[0]

    with pytest.raises(SkillCandidateStoreError, match="frontmatter"):
        materializer.materialize(proposal.proposal_id)

    assert candidate_store.list() == ()
    assert not [
        path
        for path in candidate_store.root.iterdir()
        if path.name != ".skill-candidates.lock"
    ]
