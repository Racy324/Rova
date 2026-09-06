from __future__ import annotations

import pytest

from rova.ai.context import Context
from rova.ai.events import StreamDone
from rova.ai.messages import AssistantMessage, TextBlock, UserMessage
from rova.ai.models import Model
from rova.agent_core.agent import Agent
from rova.agent_session.agent_session import AgentSession
from rova.agent_session.agent_session import PreRunContextTooLarge
from rova.agent_session.compaction import CompactionPolicy


class CharacterEstimator:
    def estimate_messages(self, messages):
        return sum(len(message.content) for message in messages if isinstance(message, UserMessage))


@pytest.mark.asyncio
async def test_proactive_compaction_rebuilds_full_provider_context_before_first_provider_step(tmp_path) -> None:
    seen_contexts = []

    async def stream(_model, context, _options):
        seen_contexts.append(context)
        yield StreamDone(AssistantMessage([TextBlock("done")]))

    agent = Agent(Model("mock", context_window=3_000), "stable", [], stream)
    session = AgentSession.create(
        agent,
        session_root=tmp_path,
        compaction_policy=CompactionPolicy(100, 2_720),
        token_estimator=CharacterEstimator(),
        provider_context_estimator=lambda context: 100 + CharacterEstimator().estimate_messages(context.messages),
        summary_fn=lambda _request: _summary(),
    )
    session._persist_user_message(UserMessage("h" * 2_000))

    async def prepare(base_context: Context) -> Context:
        return await session.prepare_provider_context(
            Context(base_context.system_prompt, list(base_context.messages), list(base_context.tools)),
            rebuild_context=lambda: Context(
                agent.create_context_snapshot().system_prompt,
                list(agent.messages),
                list(base_context.tools),
            ),
        )

    agent.set_context_preparer(prepare)
    await session.prompt("x" * 1_000)

    assert len(seen_contexts) == 1
    assert "<SUMMARY>\nsummary\n</SUMMARY>" in seen_contexts[0].messages[0].content
    assert seen_contexts[0].messages[1] == UserMessage("x" * 1_000)
    assert any(event.type == "compaction_started" and event.metadata["trigger"] == "proactive" for event in agent.events)


async def _summary() -> str:
    return "summary"


@pytest.mark.asyncio
async def test_non_compactable_provider_context_fails_without_attempting_compaction(tmp_path) -> None:
    async def stream(_model, _context, _options):
        yield StreamDone(AssistantMessage([TextBlock("unused")]))

    agent = Agent(Model("mock", context_window=100), "stable", [], stream)
    session = AgentSession.create(
        agent,
        session_root=tmp_path,
        compaction_policy=CompactionPolicy(10, 10),
        provider_context_estimator=lambda _context: 91,
    )

    with pytest.raises(PreRunContextTooLarge, match="non-compactable"):
        await session.prepare_provider_context(Context("fixed", [], []), rebuild_context=lambda: Context("fixed", [], []))

    assert not any(event.type.startswith("compaction_") for event in agent.events)
