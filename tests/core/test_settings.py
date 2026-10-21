import pytest
from pydantic import ValidationError
from sqlalchemy.engine import make_url

from claimcheck.core.settings import Settings, get_settings


def test_settings_reads_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_BACKEND", "anthropic")
    assert Settings(_env_file=None).llm_backend == "anthropic"


def test_settings_rejects_unknown_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_BACKEND", "gpt")
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_defaults() -> None:
    s = Settings(_env_file=None)
    assert s.llm_backend == "ollama"
    assert s.ollama_url == "http://localhost:11434"
    assert s.replay_dir == "evals/cassettes"
    assert s.replay_record is False
    assert s.replay_inner is None
    assert s.storage_dir == "var/storage"
    assert s.anthropic_api_key is None
    assert s.openai_api_key is None
    assert s.otel_endpoint is None
    assert s.anthropic_model == "claude-haiku-5-5"
    assert s.openai_model == "gpt-5-mini"
    assert s.ollama_model == "qwen2.5:7b-instruct"


def test_each_service_has_its_own_database_login() -> None:
    s = Settings(_env_file=None)
    users = {
        name: make_url(getattr(s, name)).username
        for name in ("database_url", "app_database_url", "worker_database_url")
    }
    assert users == {
        "database_url": "claimcheck",
        "app_database_url": "claimcheck_app",
        "worker_database_url": "claimcheck_worker",
    }


def test_worker_database_url_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    url = "postgresql+psycopg://claimcheck_worker:pw@db:5432/claimcheck"
    monkeypatch.setenv("WORKER_DATABASE_URL", url)
    assert Settings(_env_file=None).worker_database_url == url


def test_jwt_secret_is_optional_for_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    # Only the API and the token helper need it; they refuse to run without it
    # (claimcheck.api.auth).
    monkeypatch.delenv("JWT_SECRET")
    assert Settings(_env_file=None).jwt_secret is None


def test_secrets_are_masked(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-value")
    s = Settings(_env_file=None)
    assert s.anthropic_api_key is not None
    assert "sk-test-value" not in repr(s)
    assert "test-secret-not-for-production-0123456789-0123456789-0123456789-abcdef" not in repr(s)
    assert s.jwt_secret is not None
    assert (
        s.jwt_secret.get_secret_value()
        == "test-secret-not-for-production-0123456789-0123456789-0123456789-abcdef"
    )


def test_replay_inner_cannot_be_replay(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REPLAY_INNER", "replay")
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_get_settings_is_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    assert get_settings() is get_settings()


def test_blank_optional_values_mean_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REPLAY_INNER", "")
    monkeypatch.setenv("OTEL_ENDPOINT", "")
    s = Settings(_env_file=None)
    assert s.replay_inner is None
    assert s.otel_endpoint is None
