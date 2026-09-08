from __future__ import annotations

from pathlib import Path

import pytest

from rova.ai.events import StreamDone, StreamError
from rova.ai.messages import AssistantMessage, TextBlock, Usage
from rova.ai.models import Model


@pytest.mark.asyncio
async def test_provider_capability_smoke_records_accepted_and_typed_overflow_without_coding_tools(
    tmp_path: Path,
) -> None:
    from evals.runtime_v1.provider_capability import run_provider_capability_smoke
    from rova.ai.events import ProviderFailure

    async def stream(_model, context, _options):
        assert context.tools == []
        if "boundary-overflow" in context.messages[0].content:
            yield StreamError(
                "error",
                AssistantMessage([TextBlock("context limit")], stop_reason="error"),
                ProviderFailure("context_overflow", status_code=400),
            )
        else:
            yield StreamDone(AssistantMessage([TextBlock("CAPABILITY_ACK")], usage=Usage(123, 4, 127)))

    report = await run_provider_capability_smoke(
        tmp_path / "results",
        model=Model("capability-test", context_window=64_000),
        stream_fn=stream,
        probes=(("accepted", 1_000), ("boundary-overflow", 2_000)),
    )

    assert report.provider_request_count == 2
    assert report.probes[0].accepted is True
    assert report.probes[0].response_contract_satisfied is True
    assert report.probes[0].actual_input_tokens == 123
    assert report.probes[1].accepted is False
    assert report.probes[1].failure_category == "context_overflow"
    assert (tmp_path / "results" / "provider-capability.json").is_file()
