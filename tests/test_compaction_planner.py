import pytest

from rova.agent_session.compaction import (
    CompactionPlanningError,
    ConservativeTokenEstimator,
    NoSafeCompactionBoundary,
    find_compaction_plan,
)
from rova.agent_session.context_builder import ProjectedMessage
from rova.ai.messages import AssistantMessage, TextBlock, ToolCall, ToolResultMessage, UserMessage


class FakeTokenEstimator:
    def __init__(self, costs):
        self.costs = costs
        self.calls = []

    def estimate_messages(self, messages):
        self.calls.append(tuple(messages))
        return sum(self.costs[id(message)] for message in messages)


def raw(entry_id, message):
    return ProjectedMessage(message, entry_id)


def synthetic(text):
    return ProjectedMessage(UserMessage(text), None)


def estimate_for(*projected):
    return FakeTokenEstimator({id(item.message): 1 for item in projected})


def test_default_estimator_is_deterministic_non_negative_and_accounts_for_tool_data():
    estimator = ConservativeTokenEstimator()
    basic = [UserMessage("read")]
    with_tool = [
        AssistantMessage([ToolCall("call-1", "read_file", {"path": "long/path.py", "line": 42})], stop_reason="tool_calls")
    ]

    assert estimator.estimate_messages(basic) == estimator.estimate_messages(basic) >= 0
    assert estimator.estimate_messages(with_tool) > 0


def test_normal_plan_keeps_maximum_recent_complete_raw_turns_and_summarizes_previous_summary():
    items = [
        synthetic("S1"),
        raw("e2", UserMessage("U2")), raw("e3", AssistantMessage([TextBlock("A2")])),
        raw("e4", UserMessage("U3")), raw("e5", AssistantMessage([TextBlock("A3")])),
        raw("e6", UserMessage("U4")), raw("e7", AssistantMessage([TextBlock("A4")])),
    ]

    plan = find_compaction_plan(items, retained_token_budget=4, token_estimator=estimate_for(*items))

    assert plan is not None
    assert plan.messages_to_summarize == tuple(item.message for item in items[:3])
    assert plan.turn_prefix_messages == ()
    assert plan.first_kept_entry_id == "e4"
    assert plan.is_split_turn is False
    assert plan.estimated_retained_tokens == 4


def test_synthetic_user_summary_does_not_start_a_user_level_turn_or_become_first_kept():
    items = [synthetic("S1"), raw("e1", UserMessage("U1")), raw("e2", AssistantMessage([TextBlock("A1")]))]

    plan = find_compaction_plan(items, retained_token_budget=2, token_estimator=estimate_for(*items))

    assert plan is not None
    assert plan.first_kept_entry_id == "e1"
    assert plan.messages_to_summarize == (items[0].message,)


def test_no_plan_is_created_when_all_logical_messages_already_fit():
    items = [raw("e1", UserMessage("U1")), raw("e2", AssistantMessage([TextBlock("A1")]))]

    assert find_compaction_plan(items, retained_token_budget=2, token_estimator=estimate_for(*items)) is None


def test_extreme_latest_turn_can_split_immediately_after_its_initial_user_message():
    user = raw("e1", UserMessage("large user"))
    assistant = raw("e2", AssistantMessage([TextBlock("recent suffix")]))
    estimator = FakeTokenEstimator({id(user.message): 10, id(assistant.message): 2})

    plan = find_compaction_plan([user, assistant], retained_token_budget=2, token_estimator=estimator)

    assert plan is not None
    assert plan.messages_to_summarize == ()
    assert plan.turn_prefix_messages == (user.message,)
    assert plan.first_kept_entry_id == "e2"
    assert plan.is_split_turn is True
    assert plan.estimated_retained_tokens == 2


def test_extreme_turn_chooses_earliest_fitting_boundary_after_complete_tool_group():
    items = [
        raw("e1", UserMessage("U")),
        raw("e2", AssistantMessage([ToolCall("read", "read_file", {})], stop_reason="tool_calls")),
        raw("e3", ToolResultMessage("read", "read_file", [TextBlock("read result")])),
        raw("e4", AssistantMessage([ToolCall("search", "search_code", {})], stop_reason="tool_calls")),
        raw("e5", ToolResultMessage("search", "search_code", [TextBlock("search result")])),
        raw("e6", AssistantMessage([TextBlock("final")])),
    ]

    plan = find_compaction_plan(items, retained_token_budget=3, token_estimator=estimate_for(*items))

    assert plan is not None
    assert plan.turn_prefix_messages == tuple(item.message for item in items[:3])
    assert plan.first_kept_entry_id == "e4"
    assert plan.is_split_turn is True
    assert plan.estimated_retained_tokens == 3


def test_multiple_tool_calls_are_atomic_until_every_result_arrives():
    items = [
        raw("e1", UserMessage("U")),
        raw("e2", AssistantMessage([ToolCall("one", "a", {}), ToolCall("two", "b", {})], stop_reason="tool_calls")),
        raw("e3", ToolResultMessage("one", "a", [TextBlock("one")])),
        raw("e4", ToolResultMessage("two", "b", [TextBlock("two")])),
        raw("e5", AssistantMessage([TextBlock("final")])),
    ]

    plan = find_compaction_plan(items, retained_token_budget=1, token_estimator=estimate_for(*items))

    assert plan is not None
    assert plan.turn_prefix_messages == tuple(item.message for item in items[:4])
    assert plan.first_kept_entry_id == "e5"


@pytest.mark.parametrize(
    "items, error",
    [
        ([raw("e1", UserMessage("U")), raw("e2", ToolResultMessage("unknown", "tool", [TextBlock("bad")]))], CompactionPlanningError),
        ([raw("e1", UserMessage("U")), raw("e2", AssistantMessage([ToolCall("open", "tool", {})], stop_reason="tool_calls"))], NoSafeCompactionBoundary),
    ],
)
def test_planner_rejects_protocol_invalid_tool_histories(items, error):
    with pytest.raises(error):
        find_compaction_plan(items, retained_token_budget=0, token_estimator=estimate_for(*items))


def test_zero_budget_can_plan_full_history_summary_with_no_first_kept():
    items = [raw("e1", UserMessage("U")), raw("e2", AssistantMessage([TextBlock("A")]))]

    plan = find_compaction_plan(items, retained_token_budget=0, token_estimator=estimate_for(*items))

    assert plan is not None
    assert plan.messages_to_summarize == tuple(item.message for item in items)
    assert plan.turn_prefix_messages == ()
    assert plan.first_kept_entry_id is None
    assert plan.is_split_turn is False
    assert plan.estimated_retained_tokens == 0


def test_planner_rejects_negative_budget_and_negative_estimate():
    item = raw("e1", UserMessage("U"))

    with pytest.raises(CompactionPlanningError, match="budget"):
        find_compaction_plan([item], retained_token_budget=-1, token_estimator=estimate_for(item))
    with pytest.raises(CompactionPlanningError, match="non-negative"):
        find_compaction_plan([item], retained_token_budget=0, token_estimator=FakeTokenEstimator({id(item.message): -1}))
