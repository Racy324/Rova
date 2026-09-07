from __future__ import annotations

import asyncio
import base64
from pathlib import Path

from rova.ai.messages import TextBlock
from rova.ai.tools import Tool
from rova.agent_core.tools import AgentTool, AgentToolResult, ToolExecutionError, ToolExecutionMode
from rova.app.workspace import Workspace
from rova.app.workspace.environment import WorkspaceFileSystem, workspace_filesystem

from .client import VisionClient


MAX_IMAGE_BYTES = 10 * 1024 * 1024
SUPPORTED_IMAGE_MIME_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
}


class VisionToolError(ToolExecutionError):
    """A user-facing failure while obtaining a workspace image."""


def create_vision_analyze_tool(
    workspace: Workspace | WorkspaceFileSystem,
    vision_client: VisionClient,
    *,
    max_image_bytes: int = MAX_IMAGE_BYTES,
) -> AgentTool:
    filesystem = workspace_filesystem(workspace)

    async def execute(_tool_call_id: str, params: dict) -> AgentToolResult:
        image_data_url = await _load_image_data_url(filesystem, params["image_path"], max_image_bytes)
        observation = await vision_client.analyze(question=params["question"], image_data_url=image_data_url)
        return AgentToolResult([TextBlock(observation)])

    return AgentTool(
        Tool(
            "vision_analyze",
            "Analyze a supported image in the workspace using an auxiliary vision model",
            {"image_path": str, "question": str},
            required=("image_path", "question"),
        ),
        execute,
        execution_mode=ToolExecutionMode.PARALLEL,
    )


async def _load_image_data_url(filesystem: WorkspaceFileSystem, image_path: str, max_image_bytes: int) -> str:
    if image_path.lower().startswith(("http://", "https://")):
        raise VisionToolError("remote image URLs are not supported")
    resolved = filesystem.resolve(image_path)
    if not resolved.exists():
        raise VisionToolError("image file not found")
    if not resolved.is_file():
        raise VisionToolError("image path is not a regular file")
    mime_type = SUPPORTED_IMAGE_MIME_TYPES.get(resolved.suffix.lower())
    if mime_type is None:
        raise VisionToolError("unsupported image type")
    try:
        if resolved.stat().st_size > max_image_bytes:
            raise VisionToolError("image is too large")
        image_bytes = await filesystem.read_bytes(image_path, max_bytes=max_image_bytes)
        if len(image_bytes) > max_image_bytes:
            raise VisionToolError("image is too large")
    except VisionToolError:
        raise
    except OSError as error:
        raise VisionToolError("unable to read image") from error
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"
