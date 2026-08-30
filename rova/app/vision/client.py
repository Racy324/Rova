from __future__ import annotations

from typing import Protocol

import httpx

from rova.agent_core.tools import ToolExecutionError

from .settings import VisionSettings


VISION_SYSTEM_PROMPT = """You are the visual analysis component of an AI agent.

Analyze only the supplied image and answer the requested question.

Report visible evidence precisely.
Do not invent text, values, labels, relationships, or structures that cannot be determined from the image.

When information is unclear or unreadable, explicitly state the uncertainty.
"""


class VisionClientError(ToolExecutionError):
    """A provider-safe error that can become a normal tool result."""


class VisionClient(Protocol):
    async def analyze(self, *, question: str, image_data_url: str) -> str: ...


class VisionHttpClient(Protocol):
    async def post(self, url: str, *, headers: dict[str, str], json: dict) -> httpx.Response: ...


class OpenAICompatibleVisionClient:
    """One-shot, text-only adapter for an auxiliary OpenAI-compatible vision model."""

    def __init__(self, settings: VisionSettings, *, http_client: VisionHttpClient | None = None) -> None:
        if not settings.is_configured:
            raise ValueError("vision settings are not configured")
        self._settings = settings
        self._http_client = http_client

    async def analyze(self, *, question: str, image_data_url: str) -> str:
        payload = {
            "model": self._settings.model,
            "messages": [
                {"role": "system", "content": VISION_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                        {"type": "image_url", "image_url": {"url": image_data_url}},
                    ],
                },
            ],
            "temperature": 0,
        }
        try:
            if self._http_client is not None:
                response = await self._http_client.post(self._chat_completions_url(), headers=self._headers(), json=payload)
            else:
                async with httpx.AsyncClient(timeout=self._settings.timeout) as http_client:
                    response = await http_client.post(self._chat_completions_url(), headers=self._headers(), json=payload)
            response.raise_for_status()
        except httpx.TimeoutException as error:
            raise VisionClientError("auxiliary model timeout") from error
        except httpx.HTTPError as error:
            raise VisionClientError("vision provider request failed") from error
        try:
            return _observation_from_response(response.json())
        except (TypeError, ValueError) as error:
            raise VisionClientError("vision provider returned an invalid response") from error

    def _chat_completions_url(self) -> str:
        assert self._settings.base_url is not None
        return f"{self._settings.base_url.rstrip('/')}/chat/completions"

    def _headers(self) -> dict[str, str]:
        assert self._settings.api_key is not None
        return {"Authorization": f"Bearer {self._settings.api_key}", "Content-Type": "application/json"}


def _observation_from_response(response: object) -> str:
    if not isinstance(response, dict):
        raise ValueError("response must be an object")
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise ValueError("response has no choices")
    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise ValueError("response has no message")
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("response has no text observation")
    return content.strip()
