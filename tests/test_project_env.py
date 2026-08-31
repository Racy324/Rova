from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from rova.ai._env import resolve_api_key
from rova.app.settings import AppSettings
from rova.trace import RunStatus, RunTrace, TerminationReason, run_trace_to_dict


def test_project_dotenv_configures_settings_and_provider_credential(tmp_path):
    dotenv = tmp_path / ".env"
    dotenv.write_text(
        "ROVA_PROVIDER=openai_compatible\n"
        "ROVA_MODEL=dotenv-model\n"
        "ROVA_BASE_URL=https://provider.example/v1\n"
        "OPENAI_API_KEY=dotenv-secret\n",
        encoding="utf-8",
    )

    settings = AppSettings.from_env({}, dotenv_path=dotenv)

    assert settings.provider == "openai_compatible"
    assert settings.model == "dotenv-model"
    assert settings.base_url == "https://provider.example/v1"
    assert resolve_api_key("openai_compatible", {}, dotenv_path=dotenv) == "dotenv-secret"


def test_runtime_loads_dotenv_from_project_root(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text(
        "ROVA_PROVIDER=openai_compatible\n"
        "ROVA_MODEL=dotenv-model\n"
        "OPENAI_API_KEY=dotenv-secret\n",
        encoding="utf-8",
    )
    for name in ("ROVA_PROVIDER", "ROVA_MODEL", "OPENAI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr("rova.config.project_root", lambda: tmp_path)

    settings = AppSettings.from_env()

    assert settings.provider == "openai_compatible"
    assert settings.model == "dotenv-model"
    assert resolve_api_key("openai_compatible") == "dotenv-secret"


def test_explicit_environment_overrides_project_dotenv(tmp_path):
    dotenv = tmp_path / ".env"
    dotenv.write_text(
        "ROVA_MODEL=dotenv-model\nOPENAI_API_KEY=dotenv-secret\n",
        encoding="utf-8",
    )

    settings = AppSettings.from_env({"ROVA_MODEL": "system-model"}, dotenv_path=dotenv)

    assert settings.model == "system-model"
    assert resolve_api_key(
        "openai_compatible",
        {"OPENAI_API_KEY": "system-secret"},
        dotenv_path=dotenv,
    ) == "system-secret"


def test_missing_project_dotenv_is_ignored(tmp_path):
    missing = tmp_path / ".env"

    assert AppSettings.from_env({}, dotenv_path=missing) == AppSettings()
    with pytest.raises(ValueError, match="Missing API credential: OPENAI_API_KEY"):
        resolve_api_key("openai_compatible", {}, dotenv_path=missing)


def test_project_api_key_is_not_exposed_by_settings_or_trace_serialization(tmp_path):
    secret = "dotenv-secret-must-not-leak"
    dotenv = tmp_path / ".env"
    dotenv.write_text(f"OPENAI_API_KEY={secret}\n", encoding="utf-8")

    settings = AppSettings.from_env({}, dotenv_path=dotenv)
    trace = RunTrace(
        run_id="run-1",
        started_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        ended_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        status=RunStatus.COMPLETED,
        termination_reason=TerminationReason.FINAL_RESPONSE,
    )

    assert secret not in repr(settings)
    assert secret not in json.dumps(run_trace_to_dict(trace))


def test_experience_review_settings_are_enabled_by_default_and_accept_explicit_override(tmp_path):
    settings = AppSettings.from_env(
        {
            "ROVA_PROVIDER": "openai_compatible",
            "ROVA_MODEL": "main-model",
            "ROVA_BASE_URL": "https://provider.example/v1",
            "ROVA_EXPERIENCE_REVIEW_ENABLED": "false",
            "ROVA_EXPERIENCE_REVIEW_TOOL_THRESHOLD": "12",
            "ROVA_EXPERIENCE_REVIEW_TASK_THRESHOLD": "6",
        },
        dotenv_path=tmp_path / ".env",
    )

    assert settings.to_memory_model() == settings.to_model()
    assert not settings.experience_review_enabled
    assert settings.experience_review_tool_threshold == 12
    assert settings.experience_review_task_threshold == 6
    assert AppSettings.from_env({}, dotenv_path=tmp_path / ".env").experience_review_enabled


def test_memory_model_can_override_only_its_model_identity(tmp_path):
    settings = AppSettings.from_env(
        {
            "ROVA_PROVIDER": "openai_compatible",
            "ROVA_MODEL": "main-model",
            "ROVA_BASE_URL": "https://provider.example/v1",
            "ROVA_MEMORY_MODEL": "memory-model",
        },
        dotenv_path=tmp_path / ".env",
    )

    memory_model = settings.to_memory_model()

    assert memory_model.model == "memory-model"
    assert memory_model.provider == "openai_compatible"
    assert memory_model.base_url == "https://provider.example/v1"


def test_provider_timeout_defaults_to_60_and_is_carried_by_models(tmp_path):
    settings = AppSettings.from_env({}, dotenv_path=tmp_path / ".env")

    assert settings.provider_timeout == 60.0
    assert settings.to_model().provider_timeout == 60.0
    assert settings.to_memory_model().provider_timeout == 60.0


def test_provider_timeout_accepts_integer_and_float_values(tmp_path):
    integer = AppSettings.from_env({"ROVA_PROVIDER_TIMEOUT": "120"}, dotenv_path=tmp_path / ".env")
    decimal = AppSettings.from_env({"ROVA_PROVIDER_TIMEOUT": "12.5"}, dotenv_path=tmp_path / ".env")

    assert integer.provider_timeout == 120.0
    assert decimal.provider_timeout == 12.5


@pytest.mark.parametrize("value", ["0", "-1", "not-a-number"])
def test_provider_timeout_rejects_non_positive_or_invalid_values(tmp_path, value):
    with pytest.raises(ValueError, match="ROVA_PROVIDER_TIMEOUT"):
        AppSettings.from_env({"ROVA_PROVIDER_TIMEOUT": value}, dotenv_path=tmp_path / ".env")
