from pathlib import Path

import pytest

from claimcheck.core.settings import Settings
from claimcheck.gateway.anthropic import AnthropicGateway
from claimcheck.gateway.factory import get_gateway
from claimcheck.gateway.ollama import OllamaGateway
from claimcheck.gateway.openai import OpenAIGateway
from claimcheck.gateway.redaction import RedactingGateway
from claimcheck.gateway.replay import ReplayGateway
from claimcheck.gateway.tracing import TracingGateway, unwrap


def make_settings(monkeypatch: pytest.MonkeyPatch, **env: str) -> Settings:
    for name, value in env.items():
        monkeypatch.setenv(name.upper(), value)
    return Settings(_env_file=None)


def test_ollama_timeout_comes_from_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    gw = unwrap(
        get_gateway(make_settings(monkeypatch, llm_backend="ollama", ollama_timeout_seconds="240"))
    )
    assert isinstance(gw, OllamaGateway)
    assert gw.timeout == 240.0


def test_factory_selects_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    gw = get_gateway(make_settings(monkeypatch, llm_backend="ollama"))
    assert isinstance(gw, TracingGateway)
    assert isinstance(gw.inner, OllamaGateway)
    assert gw.is_cloud is False
    assert gw.model == "qwen2.5:7b-instruct"

    gw = get_gateway(make_settings(monkeypatch, llm_backend="anthropic", anthropic_api_key="k"))
    # Tracing outermost, redaction around the backend.
    assert isinstance(gw, TracingGateway)
    assert isinstance(gw.inner, RedactingGateway)
    assert isinstance(gw.inner.inner, AnthropicGateway)
    assert gw.is_cloud is True
    assert gw.model == "claude-haiku-5-5"

    gw = get_gateway(
        make_settings(monkeypatch, llm_backend="openai", openai_api_key="k", openai_model="gpt-5")
    )
    assert isinstance(gw, TracingGateway)
    assert isinstance(gw.inner, RedactingGateway)
    assert isinstance(gw.inner.inner, OpenAIGateway)
    assert gw.model == "gpt-5"


@pytest.mark.parametrize(
    ("backend", "variable"),
    [("anthropic", "ANTHROPIC_API_KEY"), ("openai", "OPENAI_API_KEY")],
)
def test_missing_api_key_is_a_clear_error(
    monkeypatch: pytest.MonkeyPatch, backend: str, variable: str
) -> None:
    with pytest.raises(ValueError, match=variable):
        get_gateway(make_settings(monkeypatch, llm_backend=backend))


def test_blank_api_key_counts_as_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        get_gateway(make_settings(monkeypatch, llm_backend="anthropic", anthropic_api_key=""))


def test_replay_needs_cassettes_or_record_mode(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    empty = tmp_path / "empty"
    with pytest.raises(ValueError, match="cassette"):
        get_gateway(make_settings(monkeypatch, llm_backend="replay", replay_dir=str(empty)))
    empty.mkdir()
    with pytest.raises(ValueError, match="cassette"):
        get_gateway(make_settings(monkeypatch, llm_backend="replay", replay_dir=str(empty)))


def test_replay_with_existing_cassettes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    (tmp_path / "k.json").write_text(
        '{"key": "k", "request_sha256": "x", "kind": "chat", "model": "m",'
        ' "is_cloud": false, "response": {}}'
    )
    gw = get_gateway(make_settings(monkeypatch, llm_backend="replay", replay_dir=str(tmp_path)))
    assert isinstance(gw, TracingGateway)
    assert isinstance(unwrap(gw), ReplayGateway)


def test_replay_record_requires_inner(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="REPLAY_INNER"):
        get_gateway(
            make_settings(
                monkeypatch, llm_backend="replay", replay_dir=str(tmp_path), replay_record="true"
            )
        )


def test_replay_record_builds_inner_backend(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    gw = get_gateway(
        make_settings(
            monkeypatch,
            llm_backend="replay",
            replay_dir=str(tmp_path),
            replay_record="true",
            replay_inner="ollama",
        )
    )
    assert isinstance(unwrap(gw), ReplayGateway)
    assert gw.is_cloud is False
    assert gw.model == "qwen2.5:7b-instruct"


def test_replay_record_with_cloud_inner_needs_its_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        get_gateway(
            make_settings(
                monkeypatch,
                llm_backend="replay",
                replay_dir=str(tmp_path),
                replay_record="true",
                replay_inner="openai",
            )
        )
