from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from .freeze_v2 import _context_cases
from .freeze_v3 import SUITE_VERSION, _authority_hashes


class FormalPreflightError(RuntimeError):
    pass


@dataclass(frozen=True)
class FormalPreflight:
    suite_version: str
    runtime_commit: str
    eval_suite_commit: str
    manifest_sha256: str
    manifest_path: str
    formal_execution_started: bool = False


def _default_git(arguments: tuple[str, ...]) -> str:
    completed = subprocess.run(["git", *arguments], check=True, capture_output=True, text=True)
    return completed.stdout.strip()


def _manifest_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verify_authority_hashes(manifest: dict[str, object], repository_root: Path) -> None:
    actual = _authority_hashes(repository_root)
    for relative, expected_hash in manifest["authority_file_sha256"].items():
        if actual.get(relative) != expected_hash:
            raise FormalPreflightError(f"authority hash mismatch: {relative}")
    for area in ("tool_parallelism", "fault_injection"):
        for relative, expected_hash in manifest[area]["authority_file_sha256"].items():
            if actual.get(relative) != expected_hash:
                raise FormalPreflightError(f"authority hash mismatch: {relative}")


def preflight_formal_suite(
    manifest_path: Path,
    *,
    repository_root: Path | None = None,
    git_runner: Callable[[tuple[str, ...]], str] = _default_git,
    docker_image_digest: Callable[[str], str] | None = None,
) -> FormalPreflight:
    """Validate v3 reproducibility facts without starting a benchmark run."""
    path = Path(manifest_path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("suite_version") != SUITE_VERSION:
        raise FormalPreflightError("unexpected formal suite version")
    if "eval_suite_commit" in manifest:
        raise FormalPreflightError("freeze manifest must not contain eval_suite_commit")
    if git_runner(("status", "--porcelain")):
        raise FormalPreflightError("working tree is not clean")
    root = Path(repository_root or Path(__file__).resolve().parents[2]).resolve()
    _verify_authority_hashes(manifest, root)
    if _context_cases() != manifest["context_cases"]:
        raise FormalPreflightError("Context fixture, prompt, validator, or allowed-change contract mismatch")
    image = manifest["sandbox"]["image"]
    expected_digest = manifest["sandbox"]["image_digest"]
    if docker_image_digest is not None and docker_image_digest(image) != expected_digest:
        raise FormalPreflightError("Sandbox image digest mismatch")
    runtime_commit = manifest["runtime_commit"]
    if len(runtime_commit) != 40:
        raise FormalPreflightError("runtime_commit is not a full SHA")
    git_runner(("merge-base", "--is-ancestor", runtime_commit, "HEAD"))
    return FormalPreflight(
        suite_version=manifest["suite_version"],
        runtime_commit=runtime_commit,
        eval_suite_commit=git_runner(("rev-parse", "HEAD")),
        manifest_sha256=_manifest_sha256(path),
        manifest_path=path.as_posix(),
    )


def write_formal_metadata(path: Path, preflight: FormalPreflight) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(asdict(preflight), handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
    return destination
