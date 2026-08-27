from __future__ import annotations

from collections.abc import Mapping
import os
from pathlib import Path


def project_root() -> Path:
    """Return the repository root independently of the current working directory."""
    return Path(__file__).resolve().parent.parent


def load_project_env(
    env: Mapping[str, str] | None = None,
    *,
    dotenv_path: Path | None = None,
) -> dict[str, str]:
    """Merge the project .env with explicit environment values without mutating os.environ.

    Normal application calls omit ``env`` and therefore load ``<project-root>/.env``.
    An injected mapping preserves the existing test/configuration seam; it only reads a
    dotenv file when ``dotenv_path`` is explicitly supplied.
    """
    path = dotenv_path if dotenv_path is not None else (project_root() / ".env" if env is None else None)
    dotenv_values = _read_dotenv(path) if path is not None else {}
    explicit_values = os.environ if env is None else env
    return {**dotenv_values, **dict(explicit_values)}


def _read_dotenv(path: Path) -> dict[str, str]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return {}

    values: dict[str, str] = {}
    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            raise ValueError(f"invalid .env entry at line {line_number}")
        name, value = line.split("=", maxsplit=1)
        name = name.strip()
        if not name:
            raise ValueError(f"invalid .env entry at line {line_number}")
        values[name] = _parse_value(value.strip())
    return values


def _parse_value(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value
