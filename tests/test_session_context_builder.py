from rova.ai.messages import AssistantMessage, TextBlock, UserMessage
from rova.agent_session.context_builder import build_session_messages
from rova.agent_session.session_store import MessageEntry


def test_builder_projects_linear_message_entries_in_path_order():
    messages = [UserMessage("one"), AssistantMessage([TextBlock("two")]), UserMessage("three")]
    entries = [
        MessageEntry("e1", None, messages[0]),
        MessageEntry("e2", "e1", messages[1]),
        MessageEntry("e3", "e2", messages[2]),
    ]

    assert build_session_messages(entries) == messages


def test_builder_projects_only_the_selected_branch_path():
    root = UserMessage("root")
    shared = AssistantMessage([TextBlock("shared")])
    selected = AssistantMessage([TextBlock("selected")])
    entries = [
        MessageEntry("e1", None, root),
        MessageEntry("e2", "e1", shared),
        MessageEntry("e4", "e2", selected),
    ]

    assert build_session_messages(entries) == [root, shared, selected]


def test_builder_returns_a_new_list_container_without_deepcopying_messages():
    first_message = UserMessage("root")
    second_message = AssistantMessage([TextBlock("reply")])
    entries = [MessageEntry("e1", None, first_message), MessageEntry("e2", "e1", second_message)]

    built = build_session_messages(entries)
    built.append(UserMessage("outside"))

    assert built[0] is first_message
    assert built[1] is second_message
    assert [entry.message for entry in entries] == [first_message, second_message]


def test_builder_projects_an_empty_path_to_an_empty_new_list():
    first = build_session_messages([])
    second = build_session_messages([])

    assert first == []
    assert first is not second
