from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class WorkspaceValidation:
    passed: bool
    reason: str


@dataclass(frozen=True)
class ContextCaseContract:
    case_id: str
    allowed_change_paths: tuple[str, ...]
    expected_target_content: str


_CONTRACTS = {
    "CM01_large_tool_output_repair": ContextCaseContract(
        "CM01_large_tool_output_repair",
        ("src/rule_engine.py",),
        'def selected_mapping() -> str:\n    return "violet-47"\n',
    ),
    "CM02_long_history_followthrough": ContextCaseContract(
        "CM02_long_history_followthrough",
        ("src/followthrough.py",),
        'def required_constraint() -> str:\n    return "keep-violet-3"\n',
    ),
}


def contract_for(case_id: str) -> ContextCaseContract:
    try:
        return _CONTRACTS[case_id]
    except KeyError as error:
        raise ValueError(f"unknown Runtime V1 Context case: {case_id}") from error


def expected_workspace_content(case_id: str) -> str:
    return contract_for(case_id).expected_target_content


def prompt_script(case_id: str, workspace: Path) -> list[str]:
    if case_id == "CM01_large_tool_output_repair":
        return [
            "Read reference.txt in full, find ROVA_EVAL_TARGET_MAPPING, and update only "
            "src/rule_engine.py so selected_mapping returns that mapping. Do not change other files."
        ]
    prompts = [
        (workspace / "constraints" / f"constraint-{index}.txt").read_text(encoding="utf-8")
        + f"\nAcknowledge constraint block {index}; do not modify files yet."
        for index in range(1, 5)
    ]
    prompts.append(
        "FINAL_IMPLEMENT: Based on the prior immutable constraint blocks, update only "
        "src/followthrough.py so required_constraint returns keep-violet-3. Do not modify other files."
    )
    return prompts


def snapshot_workspace(workspace: Path) -> dict[str, str]:
    root = Path(workspace)
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and "__pycache__" not in path.relative_to(root).parts
        and ".git" not in path.relative_to(root).parts
    }


def validate_workspace(
    case_id: str,
    workspace: Path,
    baseline: dict[str, str],
) -> WorkspaceValidation:
    contract = contract_for(case_id)
    current = snapshot_workspace(workspace)
    changed = {
        path
        for path in set(baseline) | set(current)
        if baseline.get(path) != current.get(path)
    }
    unexpected = sorted(changed - set(contract.allowed_change_paths))
    if unexpected:
        return WorkspaceValidation(False, f"unexpected workspace changes: {', '.join(unexpected)}")
    target = Path(workspace) / contract.allowed_change_paths[0]
    if not target.is_file() or target.read_text(encoding="utf-8") != contract.expected_target_content:
        return WorkspaceValidation(False, f"target content does not satisfy {case_id}")
    return WorkspaceValidation(True, "deterministic workspace contract satisfied")


def prompt_sha256(case_id: str, workspace: Path) -> str:
    payload = json.dumps(prompt_script(case_id, workspace), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validator_sha256(case_id: str) -> str:
    contract = contract_for(case_id)
    payload = json.dumps(
        {
            "allowed_change_paths": contract.allowed_change_paths,
            "expected_target_content": contract.expected_target_content,
            "validator_version": 1,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
