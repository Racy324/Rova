"""Clean-HEAD reproducibility preflight for Runtime V1 Evaluation v4."""
from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .freeze_v2 import _context_cases
from .freeze_v4 import FORMAL_EXECUTION_CONTRACT, SUITE_VERSION, authority_hashes


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


def _manifest_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verify_hashes(manifest: dict[str, object], root: Path) -> None:
    actual = authority_hashes(root)
    expected = manifest["authority_file_sha256"]
    if set(actual) != set(expected):
        raise FormalPreflightError("Formal execution authority list mismatch")
    for relative, expected_hash in expected.items():
        if actual.get(relative) != expected_hash:
            raise FormalPreflightError(f"authority hash mismatch: {relative}")


def _verify_contract(manifest: dict[str, object]) -> dict[str, int]:
    if manifest.get("formal_execution_contract") != FORMAL_EXECUTION_CONTRACT:
        raise FormalPreflightError("Formal execution contract mismatch")
    counts = manifest.get("formal_runs")
    expected = {
        "context_ab": 12,
        "tool_parallelism_warmups": 36,
        "tool_parallelism_measurements": 360,
        "fault_injection": 36,
    }
    if counts != expected:
        raise FormalPreflightError("Formal execution counts mismatch")
    observation_schema = manifest.get("fault_injection", {}).get("observation_schema", [])
    required_observation = {
        "provider_attempt_count",
        "recovered",
        "terminated",
        "termination_reason",
        "compaction_count",
        "tool_execution_count",
        "side_effect_execution_count",
        "partial_tool_execution_count",
        "duplicate_side_effect_count",
        "partial_commit_violations",
        "transparent_tool_retry_violations",
        "unexpected_retry_violations",
    }
    if not isinstance(observation_schema, list) or not required_observation.issubset(observation_schema):
        raise FormalPreflightError("Fault observation schema mismatch")
    profiles = manifest.get("context_profiles")
    if not isinstance(profiles, dict) or profiles.get("base") != {
        "compaction": False,
        "overflow_recovery": False,
        "tool_result_externalization": False,
    }:
        raise FormalPreflightError("Base Context profile mismatch")
    full = profiles.get("full")
    if not isinstance(full, dict) or full.get("compaction_policy") != {
        "context_window": 64_000,
        "reserve_tokens": 12_000,
        "keep_recent_tokens": 20_000,
    }:
        raise FormalPreflightError("Full Context profile mismatch")
    return expected


def preflight_formal_suite(
    manifest_path: Path,
    *,
    repository_root: Path | None = None,
    git_runner: Callable[[tuple[str, ...]], str] = _default_git,
    docker_image_digest: Callable[[str], str] | None = None,
) -> FormalPreflight:
    """Check v4 authority and clean-HEAD facts without starting Formal execution."""
    path = Path(manifest_path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("suite_version") != SUITE_VERSION:
        raise FormalPreflightError("unexpected formal suite version")
    if "eval_suite_commit" in manifest:
        raise FormalPreflightError("freeze manifest must not contain eval_suite_commit")
    if git_runner(("status", "--porcelain")):
        raise FormalPreflightError("working tree is not clean")
    root = Path(repository_root or Path(__file__).resolve().parents[2]).resolve()
    _verify_hashes(manifest, root)
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
        manifest_sha256=_manifest_sha256(path),
        manifest_path=path.as_posix(),
        execution_counts=counts,
    )
