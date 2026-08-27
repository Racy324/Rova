from pathlib import Path

import pytest

from rova.app.context.local import (
    MAX_CONTEXT_FILES,
    MAX_CONTEXT_FILE_BYTES,
    LocalContextError,
    load_local_research_context,
)


def test_local_context_loads_explicit_utf8_files_with_stable_labels(tmp_path: Path):
    first = tmp_path / "method.md"
    second = tmp_path / "data.csv"
    first.write_text("Method assumption A.", encoding="utf-8")
    second.write_text("metric,value\naccuracy,0.9\n", encoding="utf-8")

    context = load_local_research_context([first, second])

    assert [(item.label, item.filename, item.content) for item in context.items] == [
        ("L1", "method.md", "Method assumption A."),
        ("L2", "data.csv", "metric,value\naccuracy,0.9\n"),
    ]
    assert all(item.byte_count > 0 and len(item.sha256) == 64 for item in context.items)
    attachment = context.render_user_attachment()
    assert "Attached Context" in attachment
    assert "[L1] method.md" in attachment
    assert "[L2] data.csv" in attachment
    assert "[S#]" in attachment


def test_local_context_rejects_directories_and_non_utf8_files(tmp_path: Path):
    directory = tmp_path / "notes"
    directory.mkdir()
    invalid = tmp_path / "broken.txt"
    invalid.write_bytes(b"\xff\xfe")

    with pytest.raises(LocalContextError, match="regular file"):
        load_local_research_context([directory])
    with pytest.raises(LocalContextError, match="UTF-8"):
        load_local_research_context([invalid])


def test_local_context_enforces_file_count_and_size_limits(tmp_path: Path):
    paths = []
    for index in range(MAX_CONTEXT_FILES + 1):
        path = tmp_path / f"note-{index}.txt"
        path.write_text("x", encoding="utf-8")
        paths.append(path)
    oversized = tmp_path / "oversized.txt"
    oversized.write_bytes(b"x" * (MAX_CONTEXT_FILE_BYTES + 1))

    with pytest.raises(LocalContextError, match="at most"):
        load_local_research_context(paths)
    with pytest.raises(LocalContextError, match="too large"):
        load_local_research_context([oversized])
