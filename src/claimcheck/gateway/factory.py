"""Build the gateway named by LLM_BACKEND."""

from pathlib import Path

from claimcheck.core.settings import Settings
from claimcheck.gateway.anthropic import AnthropicGateway
from claimcheck.gateway.base import LLMGateway
from claimcheck.gateway.ollama import OllamaGateway
from claimcheck.gateway.openai import OpenAIGateway
from claimcheck.gateway.redaction import RedactingGateway
from claimcheck.gateway.replay import ReplayGateway
from claimcheck.gateway.tracing import TracingGateway


def get_gateway(settings: Settings) -> LLMGateway:
    """The gateway for LLM_BACKEND, with redaction in front of anything that is a cloud service.

    Redaction sits outside the replay layer, so cassettes only ever hold redacted requests
    and responses, and recording and replaying see the same text. A local backend is never
    wrapped in redaction. Tracing is outermost, around everything, so a call's span covers
    redaction and replay and records the usage the caller gets back.
    """
    if settings.llm_backend == "replay":
        gateway = _replay(settings)
    else:
        gateway = _backend(settings.llm_backend, settings)
    if gateway.is_cloud:
        gateway = RedactingGateway(gateway)
    return TracingGateway(gateway, system=settings.llm_backend)


def _backend(name: str, settings: Settings) -> LLMGateway:
    if name == "anthropic":
        key = _secret(settings.anthropic_api_key, "ANTHROPIC_API_KEY", name)
        return AnthropicGateway(api_key=key, model=settings.anthropic_model)
    if name == "openai":
        key = _secret(settings.openai_api_key, "OPENAI_API_KEY", name)
        return OpenAIGateway(api_key=key, model=settings.openai_model)
    if name == "ollama":
        return OllamaGateway(
            base_url=settings.ollama_url,
            model=settings.ollama_model,
            timeout=settings.ollama_timeout_seconds,
        )
    raise ValueError(f"unknown LLM backend {name!r}")


def _secret(value: object, variable: str, backend: str) -> str:
    # SecretStr or None; an empty string in .env should not pass as a key.
    text = value.get_secret_value() if hasattr(value, "get_secret_value") else None
    if not text:
        raise ValueError(f"{variable} is required when the {backend} backend is used")
    return str(text)


def _replay(settings: Settings) -> LLMGateway:
    directory = Path(settings.replay_dir)
    if not settings.replay_record:
        if not directory.is_dir() or not any(directory.glob("*.json")):
            raise ValueError(
                f"LLM_BACKEND=replay found no cassettes in {directory}; "
                "record some with REPLAY_RECORD=true and REPLAY_INNER=<backend>"
            )
        return ReplayGateway(None, directory, record=False)
    if settings.replay_inner is None:
        raise ValueError("REPLAY_RECORD=true needs REPLAY_INNER=anthropic|openai|ollama")
    return ReplayGateway(_backend(settings.replay_inner, settings), directory, record=True)
