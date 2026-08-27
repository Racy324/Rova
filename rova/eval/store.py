from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Protocol

from .models import EvalResult


SCHEMA_VERSION = 1


class EvalStoreError(RuntimeError):
    pass


class EvalCorruptionError(EvalStoreError):
    def __init__(self, line_number: int, reason: str) -> None:
        super().__init__(f"eval store is corrupt at line {line_number}: {reason}")


class EvalStore(Protocol):
    def append(self, result: EvalResult) -> None: ...
    def load_all(self) -> list[EvalResult]: ...


class JsonlEvalStore:
    """Append-only persistence for completed, independently evaluable results."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def append(self, result: EvalResult) -> None:
        _validate_finalized(result)
        existing = {item.eval_run_id for item in self.load_all()} if self.path.exists() else set()
        if result.eval_run_id in existing:
            raise EvalStoreError(f"duplicate eval_run_id: {result.eval_run_id}")
        try:
            record = {
                "schema_version": SCHEMA_VERSION,
                "result": _redact_environment_credentials(result.to_dict()),
            }
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(
                    json.dumps(record, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
                    + "\n"
                )
                handle.flush()
                os.fsync(handle.fileno())
        except (OSError, TypeError, ValueError) as error:
            raise EvalStoreError(f"could not append eval result: {error}") from error

    def load_all(self) -> list[EvalResult]:
        if not self.path.exists():
            return []
        try:
            raw = self.path.read_text(encoding="utf-8")
        except OSError as error:
            raise EvalStoreError(f"could not read eval store: {error}") from error
        if raw and not raw.endswith("\n"):
            raise EvalCorruptionError(raw.count("\n") + 1, "missing final newline")

        results: list[EvalResult] = []
        seen: set[str] = set()
        for line_number, line in enumerate(raw.splitlines(), start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise EvalCorruptionError(line_number, "invalid JSON") from error
            if (
                not isinstance(record, dict)
                or set(record) != {"schema_version", "result"}
                or record.get("schema_version") != SCHEMA_VERSION
            ):
                raise EvalCorruptionError(line_number, "unsupported schema_version")
            try:
                result = EvalResult.from_dict(record["result"])
                _validate_finalized(result)
            except (KeyError, TypeError, ValueError) as error:
                raise EvalCorruptionError(line_number, str(error)) from error
            if result.eval_run_id in seen:
                raise EvalCorruptionError(line_number, "duplicate eval_run_id")
            seen.add(result.eval_run_id)
            results.append(result)
        return results


def _validate_finalized(result: EvalResult) -> None:
    if not isinstance(result, EvalResult):
        raise EvalStoreError("only EvalResult values can be persisted")
    if not result.eval_run_id or not result.case_id:
        raise EvalStoreError("finalized EvalResult requires non-empty eval_run_id and case_id")
    if result.ended_at < result.started_at:
        raise EvalStoreError("finalized EvalResult cannot end before it starts")
    if result.duration_ms < 0:
        raise EvalStoreError("finalized EvalResult duration_ms must be non-negative")


def _redact_environment_credentials(value: object) -> object:
    """Keep result artifacts useful without copying configured credentials into them."""
    credentials = sorted(
        {
            configured
            for name, configured in os.environ.items()
            if any(marker in name.upper() for marker in ("API_KEY", "TOKEN", "SECRET", "PASSWORD"))
            and configured
        },
        key=len,
        reverse=True,
    )
    if isinstance(value, str):
        for credential in credentials:
            value = value.replace(credential, "[REDACTED]")
        return value
    if isinstance(value, list):
        return [_redact_environment_credentials(item) for item in value]
    if isinstance(value, dict):
        return {
            _redact_environment_credentials(key) if isinstance(key, str) else key:
            _redact_environment_credentials(item)
            for key, item in value.items()
        }
    return value
