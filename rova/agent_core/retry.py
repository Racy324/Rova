from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field


Sleep = Callable[[float], Awaitable[None]]
RandomSource = Callable[[], float]


@dataclass(frozen=True)
class ProviderRetryPolicy:
    """Small, bounded retry timing policy for one Agent model step."""

    max_retries: int = 2
    base_delay_seconds: float = 0.5
    max_delay_seconds: float = 4.0
    jitter_ratio: float = 0.2
    sleep: Sleep = asyncio.sleep
    random_source: RandomSource = field(default=random.random, compare=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.max_retries, int) or isinstance(self.max_retries, bool) or self.max_retries < 0:
            raise ValueError("max_retries must be a non-negative integer")

    def delay_for_retry(self, retry_number: int) -> float:
        if not isinstance(retry_number, int) or isinstance(retry_number, bool) or retry_number < 1:
            raise ValueError("retry_number must be a positive integer")
        delay = min(self.max_delay_seconds, self.base_delay_seconds * (2 ** (retry_number - 1)))
        return delay * (1 - self.jitter_ratio + (2 * self.jitter_ratio * self.random_source()))

    async def wait(self, retry_number: int) -> None:
        await self.sleep(self.delay_for_retry(retry_number))
