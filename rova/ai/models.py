from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Model:
    model: str = "mock"
    provider: str = "mock"
    base_url: str | None = None
    api: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    context_window: int | None = None
    provider_timeout: float = 60.0

    def __post_init__(self) -> None:
        if self.context_window is not None and (
            not isinstance(self.context_window, int)
            or isinstance(self.context_window, bool)
            or self.context_window <= 0
        ):
            raise ValueError("context_window must be a positive integer or None")
        if (
            not isinstance(self.provider_timeout, (int, float))
            or isinstance(self.provider_timeout, bool)
            or self.provider_timeout <= 0
        ):
            raise ValueError("provider_timeout must be a positive number")

    @property
    def name(self) -> str:
        """Compatibility alias for the Phase 1 model identifier."""
        return self.model
