"""Runtime configuration, read from the environment (and an optional .env file)."""

from functools import lru_cache
from typing import Literal

from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # database_url is the owner, for migrations, seeding and policy ingest. The API and the
    # worker log in as their own roles, which have only the privileges they use and are
    # subject to row-level security.
    database_url: str = "postgresql+psycopg://claimcheck:claimcheck@localhost:5432/claimcheck"
    app_database_url: str = (
        "postgresql+psycopg://claimcheck_app:claimcheck_app@localhost:5432/claimcheck"
    )
    worker_database_url: str = (
        "postgresql+psycopg://claimcheck_worker:claimcheck_worker@localhost:5432/claimcheck"
    )
    # The passwords migration 0001 gives those two roles. They must match the passwords in the
    # two URLs above; docker compose builds both from the same variables.
    app_db_password: SecretStr = SecretStr("claimcheck_app")
    worker_db_password: SecretStr = SecretStr("claimcheck_worker")

    # "replay" serves recorded responses and exists only for CI and tests.
    llm_backend: Literal["anthropic", "openai", "ollama", "replay"] = "ollama"
    anthropic_api_key: SecretStr | None = None
    openai_api_key: SecretStr | None = None
    anthropic_model: str = "claude-haiku-5-5"
    openai_model: str = "gpt-5-mini"
    ollama_model: str = "qwen2.5:7b-instruct"
    ollama_url: str = "http://localhost:11434"
    # A 7B model on a laptop can take over a minute for one extraction call.
    ollama_timeout_seconds: float = 300.0

    replay_dir: str = "evals/cassettes"
    replay_record: bool = False
    # Backend that fills cassette misses while recording.
    replay_inner: Literal["anthropic", "openai", "ollama"] | None = None

    storage_dir: str = "var/storage"
    # Only the API and the token helper use it, and they refuse to run without it
    # (claimcheck.api.auth), so the evals, the seed command and the worker do not need one.
    jwt_secret: SecretStr | None = None
    otel_endpoint: str | None = None

    @field_validator("replay_inner", "otel_endpoint", mode="before")
    @classmethod
    def _blank_is_unset(cls, value: object) -> object:
        # An empty `KEY=` line in .env should mean "not set", not an invalid literal.
        return None if value == "" else value


@lru_cache
def get_settings() -> Settings:
    return Settings()
