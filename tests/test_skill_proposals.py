from __future__ import annotations

from pathlib import Path

from rova.app.skill_proposals import FileSkillProposalStore, SkillProposal
from rova.app.skills import FileSkillStore


def test_pending_skill_proposals_are_durable_listable_and_readable_without_touching_active_skills(tmp_path: Path):
    proposal_store = FileSkillProposalStore(tmp_path / "proposals")
    skill_store = FileSkillStore(tmp_path / "skills")
    original = "---\nname: existing\ndescription: Existing skill.\n---\n\nOriginal.\n"
    skill_store.create("existing", original)

    saved = proposal_store.save(
        7,
        (
            SkillProposal("create", "new-skill", "# New skill", "new reusable workflow"),
            SkillProposal("patch", "existing", "# Proposed revision", "clarify a step"),
        ),
    )

    assert [item.proposal_id for item in proposal_store.list()] == [item.proposal_id for item in saved]
    assert proposal_store.read(saved[1].proposal_id) == saved[1]
    assert saved[0].review_generation == 7
    assert skill_store.read("existing") == original
    assert not (skill_store.root / "new-skill").exists()
