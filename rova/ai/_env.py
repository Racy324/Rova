from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from rova.config import load_project_env


_API_KEY_ENV = {
    "openai_compatible": "OPENAI_API_KEY",
}


def resolve_api_key(
    provider: str,
    env: Mapping[str, str] | None = None,
    *,
    dotenv_path: Path | None = None,
) -> str:
    variable = _API_KEY_ENV.get(provider)
    if variable is None:
        raise ValueError(f"No API key environment variable is configured for provider: {provider}")
    source = load_project_env(env, dotenv_path=dotenv_path)
    api_key = source.get(variable)
    if not api_key:
        raise ValueError(f"Missing API credential: {variable}")
    return api_key
