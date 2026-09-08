from __future__ import annotations

from pathlib import Path

import pytest


def test_fresh_workspace_removes_successful_copy_and_preserves_failed_copy_only_when_requested(tmp_path: Path) -> None:
    from evals.runtime_v1.fixtures import fresh_workspace, smoke_fixtures

    fixture = smoke_fixtures()[0]
    successful_root = tmp_path / "success"
    with fresh_workspace(fixture, successful_root) as workspace:
        (workspace / "generated.txt").write_text("temporary", encoding="utf-8")
    assert not successful_root.exists()

    failed_root = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="smoke failure"):
        with fresh_workspace(fixture, failed_root, keep_failed=True):
            raise RuntimeError("smoke failure")
    retained = list(failed_root.glob("CM01_fixture_probe-*"))
    assert len(retained) == 1
    assert (retained[0] / "reference.txt").is_file()


def test_fixture_hash_and_workspace_copy_ignore_python_cache_files(tmp_path: Path) -> None:
    from evals.runtime_v1.fixtures import SmokeFixture, fresh_workspace

    source = tmp_path / "fixture"
    source.mkdir()
    (source / "tracked.txt").write_text("fixture authority", encoding="utf-8")
    cache = source / "__pycache__"
    cache.mkdir()
    (cache / "generated.pyc").write_bytes(b"first-machine-cache")
    fixture = SmokeFixture("fixture", source)

    first_hash = fixture.sha256
    (cache / "generated.pyc").write_bytes(b"second-machine-cache")

    assert fixture.sha256 == first_hash
    with fresh_workspace(fixture, tmp_path / "workspaces") as workspace:
        assert (workspace / "tracked.txt").read_text(encoding="utf-8") == "fixture authority"
        assert not (workspace / "__pycache__").exists()
