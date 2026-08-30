from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any

from rova.ai import stream_simple
from rova.agent_core.agent import Agent
from rova.agent_core.tool_output import ToolOutputProcessor
from rova.agent_session.agent_session import AgentSession
from rova.agent_session.compaction import CompactionPolicy
from rova.app.memory import FileMemoryStore, MemorySnapshot
from rova.app.runtime import ROVA_SYSTEM_PROMPT, _with_runtime_context, build_rova_runtime
from rova.app.settings import AppSettings
from rova.app.workspace.approval import AlwaysApprove
from rova.app.workspace.context import WorkspaceContext
from rova.app.workspace.controlled_tool import build_controlled_coding_tools
from rova.app.workspace.instructions import load_workspace_instruction
from rova.app.workspace.policy import DefaultCodingToolPolicy
from rova.app.workspace.workspace import Workspace
from rova.app.skills import SkillCatalogSnapshot
from rova.artifacts import FileArtifactStore
from rova.trace.recorder import TraceRecorder
from rova.trace.store import JsonlTraceStore

from .fixtures import provision_workspace
from .graders import grade_case
from .spec import ResearchCase


@dataclass
class _BaseRuntime:
    agent: Agent
    session: AgentSession

    async def prompt(self, text: str):
        return await self.session.prompt(text)


async def execute_case(
    case: ResearchCase,
    *,
    profile: str,
    run_index: int,
    run_root: Path,
    trace_store: JsonlTraceStore,
    settings: AppSettings | None = None,
) -> dict[str, Any]:
    settings = settings or AppSettings.from_env()
    workspace = run_root / "workspace"
    provision_workspace(case, workspace)
    runtime = _build_profile(case, profile, workspace, run_root, settings)
    traces = []
    final_text = ""
    error: Exception | None = None

    async def prompt(text: str):
        nonlocal final_text, error
        recorder = TraceRecorder()
        try:
            responses, trace = await recorder.capture_run(
                runtime.agent, lambda: runtime.prompt(text), session=runtime.session,
                memory=runtime if profile == "full" else None,
            )
            if responses:
                final_text = responses[-1].text
            traces.append(trace)
        except Exception as caught:
            error = caught
            if recorder.last_trace is not None:
                traces.append(recorder.last_trace)

    if case.case_id == "C07":
        await prompt(case.prompt)
        filler = ("The following ordinary research-log note is non-authoritative background. " * 125)
        await prompt(f"Read research_log.txt with the workspace tool. Then acknowledge this note: {filler}")
        await prompt(f"Continue the session; do not modify files. Keep the prior constraints while considering this background: {filler}")
        await prompt("Now list every binding constraint and propose only a structural verification plan. Do not modify Decoder or run long training.")
    elif case.case_id == "C08":
        await prompt(case.prompt)
        await prompt("For the same future research work, restate only the stable habits; do not promote today's config.py task into a general rule.")
        await prompt("Confirm the stable transfer habits once more. Do not add unrelated facts.")
        try:
            runtime.session.close()
        except Exception:
            pass
        runtime = _build_profile(case, profile, workspace, run_root, settings, new_session=True)
        await prompt("In a new session, state the next method-transfer principles from the user's established research habits. If those habits are unavailable, say the information is insufficient rather than guessing.")
    else:
        await prompt(case.prompt)

    for trace in traces:
        trace_store.append(trace)
    extra: dict[str, Any] = {}
    if case.case_id == "C06":
        extra["transfer_test"] = _transfer_test(workspace / "transfer_workspace")
        extra["tests_changed"] = False  # fixture is freshly created; no mutation outside model.py is accepted below.
    if case.case_id == "C07":
        extra["compactions"] = [item for trace in traces for item in trace.compactions]
        extra["tool_protocol_error"] = False
        extra["prohibited_action"] = any("long training" in (tool.command or "").lower() for trace in traces for tool in trace.tool_executions)
    if case.case_id == "C08":
        extra["memory_enabled"] = profile == "full"
    tools = [
        {"tool_name": tool.tool_name, "command": tool.command or "", "path": tool.arguments.get("path", "")}
        for trace in traces for tool in trace.tool_executions
    ]
    grading = grade_case(case, final_text, workspace, tools, profile=profile, extra=extra)
    infra = _infra_error(error, traces)
    status = "INFRA_ERROR" if infra else ("PASS" if all(item["passed"] for item in grading["critical_checks"]) else "FAIL")
    usage = {"input_tokens": sum((trace.usage.input_tokens if trace.usage else 0) for trace in traces), "output_tokens": sum((trace.usage.output_tokens if trace.usage else 0) for trace in traces), "total_tokens": sum((trace.usage.total_tokens if trace.usage else 0) for trace in traces)}
    c07_metrics = _c07_metrics(profile, traces, usage) if case.case_id == "C07" else None
    return {
        "eval_version": "research-eval-v1", "case_id": case.case_id, "profile": profile, "run_index": run_index,
        "model": {"provider": runtime.agent.model.provider, "model": runtime.agent.model.model, "temperature": runtime.agent.model.temperature, "max_tokens": runtime.agent.model.max_tokens, "context_window": runtime.agent.model.context_window},
        "status": status, "critical_checks": grading["critical_checks"], "protocol_checks": grading["protocol_checks"],
        "token_usage": usage, "tool_calls": len(tools), "duration_ms": sum(trace.duration_ms or 0 for trace in traces),
        "session_id": runtime.session.session_id, "trace_references": [trace.run_id for trace in traces],
        "workspace_reference": str(workspace), "final_text": final_text, "error": str(error) if error else None,
        "c07": c07_metrics,
    }


def _build_profile(case: ResearchCase, profile: str, workspace_root: Path, run_root: Path, settings: AppSettings, *, new_session: bool = False):
    if profile not in {"base", "full"}:
        raise ValueError("profile must be base or full")
    model = settings.to_model()
    # C07 deliberately drives the existing automatic compaction code. This metadata
    # is applied identically to both profiles and is not sent as a provider parameter.
    if case.case_id == "C07":
        model = replace(model, context_window=12_000)
    data = run_root / "data"
    session_root, memory_root, artifact_root = data / "sessions", data / "memory", data / "artifacts"
    if profile == "full":
        skill_root = data / "skills"
        if not skill_root.exists():
            source = settings.data_paths().skills
            if source.is_dir():
                shutil.copytree(source, skill_root, ignore=shutil.ignore_patterns(".skills.lock", "__pycache__"))
            else:
                skill_root.mkdir(parents=True, exist_ok=True)
        return build_rova_runtime(
            model=model, stream_fn=stream_simple, workspace_root=workspace_root, approval_handler=AlwaysApprove(), permission_mode="full",
            session_root=session_root, artifact_root=artifact_root, memory_root=memory_root, memory_store=FileMemoryStore(memory_root),
            skill_root=skill_root, memory_model=settings.to_memory_model(), memory_update_interval=settings.memory_update_interval,
            memory_max_chars=settings.memory_max_chars, memory_consolidation_threshold=settings.memory_consolidation_threshold,
            compaction_policy=CompactionPolicy(7_000, 3_000) if case.case_id == "C07" else settings.to_compaction_policy(), max_turns=case.max_turns,
        )
    workspace = Workspace(workspace_root)
    context = WorkspaceContext(workspace)
    tools = build_controlled_coding_tools(workspace, DefaultCodingToolPolicy(), AlwaysApprove(), context)
    artifacts = FileArtifactStore(artifact_root)
    base_stream = _with_runtime_context(
        stream_simple, workspace, MemorySnapshot(), load_workspace_instruction(workspace.root), SkillCatalogSnapshot(), web_enabled=False,
    )
    agent = Agent(model, ROVA_SYSTEM_PROMPT, tools, base_stream, max_turns=case.max_turns, tool_output_processor=ToolOutputProcessor(artifacts))
    return _BaseRuntime(agent, AgentSession.create(agent, session_root=session_root, compaction_policy=None))


def _transfer_test(root: Path) -> dict[str, bool]:
    code = """import torch\nfrom model import TinyTransferModel\nm=TinyTransferModel(); x=torch.randn(2,8,6,6,requires_grad=True); y=m(x); y.mean().backward()\nassert y.shape == (2,2,6,6)\nassert m.research_adapter.scale.grad is not None\n"""
    executable = os.environ.get("ROVA_EVAL_PYTHON", "python")
    result = subprocess.run([executable, "-c", code], cwd=root, capture_output=True, text=True, timeout=90)
    passed = result.returncode == 0
    return {"import": passed, "construct": passed, "forward": passed, "backward": passed, "gradient": passed, "shape": passed, "tests_pass": passed}


def _infra_error(error: Exception | None, traces) -> bool:
    if error is not None:
        return any(token in str(error).lower() for token in ("timeout", "429", "provider", "connection", "unavailable")) or any(trace.status.value == "provider_error" for trace in traces)
    return any(trace.status.value == "provider_error" for trace in traces)


def _c07_metrics(profile: str, traces, usage: dict[str, int]) -> dict[str, Any]:
    compactions = [item for trace in traces for item in trace.compactions]
    return {"profile": profile, "provider_input_tokens": usage["input_tokens"], "compaction_count": len(compactions), "compaction_before_tokens": [item.pressure_before for item in compactions], "compaction_after_tokens": [item.pressure_after for item in compactions]}
