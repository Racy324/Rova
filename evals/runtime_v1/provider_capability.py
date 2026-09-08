from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable

from rova.ai.context import Context
from rova.ai.events import StreamDone, StreamError
from rova.ai.messages import UserMessage
from rova.ai.models import Model
from rova.ai.stream import stream_simple


@dataclass(frozen=True)
class ProviderCapabilityProbe:
    probe_id: str
    payload_chars: int
    accepted: bool
    response_contract_satisfied: bool
    actual_input_tokens: int | None
    actual_output_tokens: int | None
    failure_category: str | None
    status_code: int | None


@dataclass(frozen=True)
class ProviderCapabilityReport:
    model_name: str
    provider: str
    configured_context_window: int | None
    provider_request_count: int
    probes: tuple[ProviderCapabilityProbe, ...]


async def run_provider_capability_smoke(
    result_root: Path,
    *,
    model: Model,
    stream_fn: Callable = stream_simple,
    probes: Iterable[tuple[str, int]] = (("accepted-50k", 300_000), ("accepted-72k", 430_000), ("overflow-128k-plus", 800_000)),
) -> ProviderCapabilityReport:
    """Issue deterministic no-tool requests to measure provider input limits."""
    observations: list[ProviderCapabilityProbe] = []
    for probe_id, payload_chars in probes:
        if not probe_id or payload_chars <= 0:
            raise ValueError("capability probes require a name and positive payload size")
        payload = _payload(probe_id, payload_chars)
        context = Context(
            system_prompt="Return exactly CAPABILITY_ACK. This is a context-capability probe, not a coding task.",
            messages=[UserMessage(payload)],
            tools=[],
        )
        done = None
        failure = None
        async for event in stream_fn(model, context, None):
            if isinstance(event, StreamDone):
                done = event.message
                break
            if isinstance(event, StreamError):
                failure = event.failure
                break
        observations.append(ProviderCapabilityProbe(
            probe_id=probe_id,
            payload_chars=payload_chars,
            # Transport acceptance is the capability fact.  Exact text is
            # recorded separately because provider/model instruction following
            # must not turn into a false context-window rejection.
            accepted=done is not None,
            response_contract_satisfied=done is not None and done.text.strip() == "CAPABILITY_ACK",
            actual_input_tokens=done.usage.input_tokens if done and done.usage else None,
            actual_output_tokens=done.usage.output_tokens if done and done.usage else None,
            failure_category=failure.category if failure else None,
            status_code=failure.status_code if failure else None,
        ))
    report = ProviderCapabilityReport(
        model_name=model.model,
        provider=model.provider,
        configured_context_window=model.context_window,
        provider_request_count=len(observations),
        probes=tuple(observations),
    )
    root = Path(result_root)
    root.mkdir(parents=True, exist_ok=True)
    (root / "provider-capability.json").write_text(
        json.dumps(
            {
                "model_name": report.model_name,
                "provider": report.provider,
                "configured_context_window": report.configured_context_window,
                "provider_request_count": report.provider_request_count,
                "probes": [asdict(item) for item in report.probes],
            },
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    return report


def _payload(probe_id: str, payload_chars: int) -> str:
    prefix = f"{probe_id}\n"
    unit = "deterministic capability payload keeps this request semantically inert.\n"
    repeats = max(1, (payload_chars - len(prefix) + len(unit) - 1) // len(unit))
    return (prefix + unit * repeats)[:payload_chars]
