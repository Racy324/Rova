"""Optional auxiliary vision capability for workspace images."""

from .client import OpenAICompatibleVisionClient, VisionClient, VisionClientError
from .settings import VisionSettings
from .tool import create_vision_analyze_tool

__all__ = [
    "OpenAICompatibleVisionClient",
    "VisionClient",
    "VisionClientError",
    "VisionSettings",
    "create_vision_analyze_tool",
]
