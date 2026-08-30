from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from rova.config import load_project_env


@dataclass(frozen=True)
class VisionSettings:
    """Configuration for the optional, standalone auxiliary vision model."""

    model: str = "qwen3-vl-flash"
    base_url: str | None = None
    api_key: str | None = field(default=None, repr=False)
    timeout: float = 60.0

    @classmethod
    def from_env(
        cls,
        env: Mapping[str, str] | None = None,
        *,
        dotenv_path: Path | None = None,
    ) -> "VisionSettings":
        source = load_project_env(env, dotenv_path=dotenv_path)
        timeout_value = source.get("ROVA_VISION_TIMEOUT")
        timeout = cls.timeout if not timeout_value else float(timeout_value)
        if timeout <= 0:
            raise ValueError("ROVA_VISION_TIMEOUT must be positive")
        return cls(
            model=source.get("ROVA_VISION_MODEL") or cls.model,
            base_url=source.get("ROVA_VISION_BASE_URL") or None,
            api_key=source.get("ROVA_VISION_API_KEY") or None,
            timeout=timeout,
        )

    @property
    def is_configured(self) -> bool:
        return bool(self.base_url and self.api_key)
