from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from rova.ai.events import StreamDone
from rova.ai.messages import AssistantMessage, TextBlock, ToolCall, ToolResultMessage
from rova.ai.models import Model
from rova.agent_core.tools import ToolRegistry, ToolRuntime
from rova.app import cli
from rova.app.runtime import build_rova_runtime
from rova.app.settings import AppSettings
from rova.app.vision.client import OpenAICompatibleVisionClient, VisionClientError, VISION_SYSTEM_PROMPT
from rova.app.vision.settings import VisionSettings
from rova.app.vision.tool import MAX_IMAGE_BYTES, create_vision_analyze_tool
from rova.app.workspace import Workspace


@pytest.fixture(autouse=True)
def _isolate_default_rova_data_dir(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("ROVA_DATA_DIR", str(tmp_path / "rova-data"))


class FakeVisionClient:
    def __init__(self, observation: str = "The visible diagram has two boxes.") -> None:
        self.observation = observation
        self.calls: list[dict[str, str]] = []

    async def analyze(self, *, question: str, image_data_url: str) -> str:
        self.calls.append({"question": question, "image_data_url": image_data_url})
        return self.observation


class FakeVisionResponse:
    def __init__(self, payload: object, *, status_code: int = 200, error: Exception | None = None) -> None:
        self._payload = payload
        self.status_code = status_code
        self._error = error

    def raise_for_status(self) -> None:
        if self._error is not None:
            raise self._error

    def json(self) -> object:
        return self._payload


class FakeVisionHttpClient:
    def __init__(self, response: FakeVisionResponse | None = None, *, error: Exception | None = None) -> None:
        self.response = response
        self.error = error
        self.calls: list[dict[str, object]] = []

    async def post(self, url: str, *, headers: dict[str, str], json: dict) -> FakeVisionResponse:
        self.calls.append({"url": url, "headers": headers, "json": json})
        if self.error is not None:
            raise self.error
        assert self.response is not None
        return self.response


@pytest.fixture
def workspace_root(tmp_path: Path) -> Path:
    root = tmp_path / "workspace"
    root.mkdir()
    return root


def test_vision_settings_are_disabled_without_auxiliary_configuration() -> None:
    settings = VisionSettings.from_env({})

    assert settings.model == "qwen3-vl-flash"
    assert settings.is_configured is False


def test_vision_settings_read_dedicated_environment_values() -> None:
    settings = VisionSettings.from_env({
        "ROVA_VISION_MODEL": "vision-test",
        "ROVA_VISION_BASE_URL": "https://vision.example/v1",
        "ROVA_VISION_API_KEY": "test-key",
        "ROVA_VISION_TIMEOUT": "12.5",
    })

    assert settings.model == "vision-test"
    assert settings.base_url == "https://vision.example/v1"
    assert settings.timeout == 12.5
    assert settings.is_configured is True


@pytest.mark.asyncio
async def test_vision_client_sends_only_question_and_image_to_auxiliary_provider() -> None:
    http_client = FakeVisionHttpClient(FakeVisionResponse({
        "choices": [{"message": {"content": "Visible: a blue square."}}],
    }))
    client = OpenAICompatibleVisionClient(
        VisionSettings(model="vision-test", base_url="https://vision.example/v1", api_key="test-key"),
        http_client=http_client,
    )

    observation = await client.analyze(question="What shape is visible?", image_data_url="data:image/png;base64,AAE=")

    assert observation == "Visible: a blue square."
    request = http_client.calls[0]
    assert request["url"] == "https://vision.example/v1/chat/completions"
    assert request["headers"] == {"Authorization": "Bearer test-key", "Content-Type": "application/json"}
    payload = request["json"]
    assert isinstance(payload, dict)
    assert payload["model"] == "vision-test"
    assert payload["temperature"] == 0
    assert payload["messages"] == [
        {"role": "system", "content": VISION_SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "text", "text": "What shape is visible?"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAE="}},
        ]},
    ]


@pytest.mark.asyncio
async def test_vision_client_reports_timeout_without_leaking_api_key() -> None:
    secret = "top-secret-vision-key"
    client = OpenAICompatibleVisionClient(
        VisionSettings(base_url="https://vision.example/v1", api_key=secret),
        http_client=FakeVisionHttpClient(error=httpx.TimeoutException(secret)),
    )

    with pytest.raises(VisionClientError, match="auxiliary model timeout") as error:
        await client.analyze(question="inspect", image_data_url="data:image/png;base64,AAE=")

    assert secret not in str(error.value)


@pytest.mark.asyncio
async def test_vision_client_redacts_provider_failure_details() -> None:
    secret = "top-secret-vision-key"
    request = httpx.Request("POST", "https://vision.example/v1/chat/completions")
    response = httpx.Response(503, request=request)
    client = OpenAICompatibleVisionClient(
        VisionSettings(base_url="https://vision.example/v1", api_key=secret),
        http_client=FakeVisionHttpClient(FakeVisionResponse(
            {},
            error=httpx.HTTPStatusError(f"provider rejected bearer {secret}", request=request, response=response),
        )),
    )

    with pytest.raises(VisionClientError, match="provider request failed") as error:
        await client.analyze(question="inspect", image_data_url="data:image/png;base64,AAE=")

    assert secret not in str(error.value)


@pytest.mark.asyncio
@pytest.mark.parametrize(("filename", "mime_type"), [
    ("diagram.png", "image/png"),
    ("photo.jpg", "image/jpeg"),
    ("photo.jpeg", "image/jpeg"),
    ("diagram.webp", "image/webp"),
])
async def test_vision_tool_reads_supported_workspace_images(
    workspace_root: Path,
    filename: str,
    mime_type: str,
) -> None:
    (workspace_root / filename).write_bytes(b"image-bytes")
    vision_client = FakeVisionClient()
    tool = create_vision_analyze_tool(Workspace(workspace_root), vision_client)

    result = await tool.execute("vision-call", {"image_path": filename, "question": "Describe the visible structure."})

    assert result.content == [TextBlock("The visible diagram has two boxes.")]
    assert vision_client.calls == [{
        "question": "Describe the visible structure.",
        "image_data_url": f"data:{mime_type};base64,aW1hZ2UtYnl0ZXM=",
    }]


@pytest.mark.asyncio
@pytest.mark.parametrize(("image_path", "expected_message"), [
    ("../outside.png", "escapes workspace root"),
    ("missing.png", "image file not found"),
    ("notes.txt", "unsupported image type"),
    ("folder.png", "image path is not a regular file"),
    ("https://example.test/image.png", "remote image URLs are not supported"),
])
async def test_vision_tool_returns_normal_tool_errors_for_invalid_image_paths(
    workspace_root: Path,
    image_path: str,
    expected_message: str,
) -> None:
    (workspace_root / "notes.txt").write_text("not an image", encoding="utf-8")
    (workspace_root / "folder.png").mkdir()
    registry = ToolRegistry([create_vision_analyze_tool(Workspace(workspace_root), FakeVisionClient())])

    result = await ToolRuntime(registry).execute(ToolCall("vision-call", "vision_analyze", {
        "image_path": image_path,
        "question": "Inspect it",
    }))

    assert result.role == "tool"
    assert result.is_error is True
    assert expected_message in result.text


@pytest.mark.asyncio
async def test_vision_tool_rejects_images_over_the_size_limit(workspace_root: Path) -> None:
    (workspace_root / "large.png").write_bytes(b"0" * (MAX_IMAGE_BYTES + 1))
    registry = ToolRegistry([create_vision_analyze_tool(Workspace(workspace_root), FakeVisionClient())])

    result = await ToolRuntime(registry).execute(ToolCall("vision-call", "vision_analyze", {
        "image_path": "large.png",
        "question": "Inspect it",
    }))

    assert result.is_error is True
    assert "image is too large" in result.text


def test_cli_does_not_read_optional_vision_settings_without_workspace(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    class UnexpectedVisionSettings:
        @classmethod
        def from_env(cls):
            raise AssertionError("Vision settings must not affect runs without a workspace")

    monkeypatch.setattr(cli, "VisionSettings", UnexpectedVisionSettings)
    monkeypatch.setattr(cli, "build_rova_runtime", lambda **kwargs: captured.update(kwargs) or object())

    cli._build_runtime_from_args(
        cli.parse_rova_cli_args(["plain text task"]),
        app_settings=AppSettings(data_dir=tmp_path / "runtime-data"),
    )

    assert captured["workspace_root"] is None
    assert captured["vision_client"] is None


def test_runtime_injects_vision_only_when_workspace_and_client_are_available(workspace_root: Path, tmp_path: Path) -> None:
    async def stream(_model, _context, _options):
        yield StreamDone(AssistantMessage([TextBlock("done")]))

    without_vision = build_rova_runtime(
        model=Model(provider="mock"), stream_fn=stream, workspace_root=workspace_root,
        session_root=tmp_path / "sessions-without", artifact_root=tmp_path / "artifacts-without",
    )
    with_vision = build_rova_runtime(
        model=Model(provider="mock"), stream_fn=stream, workspace_root=workspace_root,
        vision_client=FakeVisionClient(), session_root=tmp_path / "sessions-with", artifact_root=tmp_path / "artifacts-with",
    )

    assert "vision_analyze" not in {tool.name for tool in without_vision.agent.registry.schemas}
    with_tool_names = {tool.name for tool in with_vision.agent.registry.schemas}
    assert "vision_analyze" in with_tool_names
    assert {"read", "write", "edit", "shell", "skill_view", "skill_manage"} <= with_tool_names


def test_runtime_rejects_a_vision_client_without_workspace(tmp_path: Path) -> None:
    async def stream(_model, _context, _options):
        yield StreamDone(AssistantMessage([TextBlock("done")]))

    with pytest.raises(ValueError, match="vision_client requires workspace_root"):
        build_rova_runtime(
            model=Model(provider="mock"), stream_fn=stream, vision_client=FakeVisionClient(),
            session_root=tmp_path / "sessions", artifact_root=tmp_path / "artifacts",
        )


@pytest.mark.asyncio
async def test_vision_observation_enters_existing_session_as_a_tool_result(workspace_root: Path, tmp_path: Path) -> None:
    (workspace_root / "diagram.png").write_bytes(b"image-bytes")

    async def stream(_model, context, _options):
        if not any(isinstance(message, ToolResultMessage) for message in context.messages):
            yield StreamDone(AssistantMessage([
                ToolCall("vision-call", "vision_analyze", {"image_path": "diagram.png", "question": "What is visible?"}),
            ], stop_reason="tool_calls"))
        else:
            yield StreamDone(AssistantMessage([TextBlock("I used the visual observation.")]))

    runtime = build_rova_runtime(
        model=Model(provider="mock"), stream_fn=stream, workspace_root=workspace_root,
        vision_client=FakeVisionClient("A visible arrow connects the two boxes."),
        session_root=tmp_path / "sessions", artifact_root=tmp_path / "artifacts",
    )

    responses = await runtime.session.prompt("Analyze diagram.png")
    result = next(message for message in runtime.agent.messages if isinstance(message, ToolResultMessage))

    assert responses[-1].text == "I used the visual observation."
    assert result.role == "tool"
    assert result.tool_name == "vision_analyze"
    assert result.text == "A visible arrow connects the two boxes."
