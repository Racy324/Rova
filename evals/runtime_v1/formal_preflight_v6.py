"""Clean-HEAD preflight for the Runtime V1 Evaluation v6 contract."""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .freeze_v2 import _context_cases
from .freeze_v6 import FORMAL_EXECUTION_CONTRACT, SUITE_VERSION, authority_hashes
from .frozen_manifest import load_frozen_manifest


class FormalPreflightError(RuntimeError):
    pass


@dataclass(frozen=True)
class FormalPreflight:
    suite_version: str
    runtime_commit: str
    eval_suite_commit: str
    manifest_sha256: str
    manifest_path: str
    execution_counts: dict[str, int]
    formal_execution_started: bool = False


def _default_git(arguments: tuple[str, ...]) -> str:
    completed = subprocess.run(["git", *arguments], check=True, capture_output=True, text=True)
    return completed.stdout.strip()


def _verify_contract(manifest: dict[str, object]) -> dict[str, int]:
    if manifest.get("formal_execution_contract") != FORMAL_EXECUTION_CONTRACT:
        raise FormalPreflightError("Formal execution contract mismatch")
    expected = {
        "context_ab": 12,
        "tool_parallelism_warmups": 36,
        "tool_parallelism_measurements": 360,
        "fault_injection": 36,
    }
    if manifest.get("formal_runs") != expected:
        raise FormalPreflightError("Formal execution counts mismatch")
    return expected


def preflight_formal_suite(
    manifest_path: Path,
    *,
    repository_root: Path | None = None,
    git_runner: Callable[[tuple[str, ...]], str] = _default_git,
    docker_image_digest: Callable[[str], str] | None = None,
) -> FormalPreflight:
    frozen = load_frozen_manifest(manifest_path)
    manifest = frozen.document
    if manifest.get("suite_version") != SUITE_VERSION:
        raise FormalPreflightError("unexpected formal suite version")
    if "eval_suite_commit" in manifest:
        raise FormalPreflightError("freeze manifest must not contain eval_suite_commit")
    if git_runner(("status", "--porcelain")):
        raise FormalPreflightError("working tree is not clean")
    root = Path(repository_root or Path(__file__).resolve().parents[2]).resolve()
    if authority_hashes(root) != manifest.get("authority_file_sha256"):
        raise FormalPreflightError("authority hash mismatch")
    if _context_cases() != manifest["context_cases"]:
        raise FormalPreflightError("Context fixture, prompt, validator, or allowed-change contract mismatch")
    counts = _verify_contract(manifest)
    sandbox = manifest["sandbox"]
    if docker_image_digest is not None and docker_image_digest(str(sandbox["image"])) != sandbox["image_digest"]:
        raise FormalPreflightError("Sandbox image digest mismatch")
    runtime_commit = str(manifest["runtime_commit"])
    if len(runtime_commit) != 40:
        raise FormalPreflightError("runtime_commit is not a full SHA")
    git_runner(("merge-base", "--is-ancestor", runtime_commit, "HEAD"))
    return FormalPreflight(
        suite_version=SUITE_VERSION,
        runtime_commit=runtime_commit,
        eval_suite_commit=git_runner(("rev-parse", "HEAD")),
        manifest_sha256=frozen.sha256,
        manifest_path=frozen.path.as_posix(),
        execution_counts=counts,
    )
