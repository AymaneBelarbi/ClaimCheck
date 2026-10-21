"""JWT_SECRET: optional for Settings, required by whatever signs or checks a token."""

import logging
import re
from pathlib import Path

import jwt
import pytest
from fastapi.testclient import TestClient

from api import tokens
from api.auth import COMPOSE_DEV_SECRET, PLACEHOLDER_SECRETS, MissingSecretError, signing_secret
from api.main import app
from core.settings import get_settings

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def no_secret(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("JWT_SECRET")
    # Settings also read ./.env; a developer's own must not supply one.
    monkeypatch.chdir(tmp_path)
    get_settings.cache_clear()


def test_signing_needs_a_secret(no_secret: None) -> None:
    with pytest.raises(MissingSecretError, match="JWT_SECRET is not set"):
        signing_secret()


def test_the_api_does_not_start_without_a_secret(no_secret: None) -> None:
    with pytest.raises(MissingSecretError), TestClient(app):
        pass


@pytest.mark.parametrize("placeholder", sorted(PLACEHOLDER_SECRETS))
def test_the_api_warns_when_the_secret_is_a_published_placeholder(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, placeholder: str
) -> None:
    monkeypatch.setenv("JWT_SECRET", placeholder)
    get_settings.cache_clear()

    with caplog.at_level(logging.WARNING, logger="api.main"), TestClient(app):
        pass

    assert "JWT_SECRET" in caplog.text and "placeholder" in caplog.text


def test_no_warning_for_a_secret_of_its_own(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="api.main"), TestClient(app):
        pass

    assert "JWT_SECRET" not in caplog.text


def test_the_placeholders_are_the_ones_the_repository_publishes() -> None:
    compose = (ROOT / "docker-compose.yml").read_text()
    example = (ROOT / ".env.example").read_text()
    assert f"${{JWT_SECRET:-{COMPOSE_DEV_SECRET}}}" in compose
    (published,) = re.findall(r"^JWT_SECRET=(.*)$", example, re.MULTILINE)
    assert published in PLACEHOLDER_SECRETS


def test_token_cli_without_a_secret_says_so(no_secret: None) -> None:
    with pytest.raises(SystemExit) as exc:
        tokens.main(["--role", "adjuster", "--sub", "adj-1"])

    assert "JWT_SECRET is not set" in str(exc.value.code)


def test_token_cli_can_sign_for_the_compose_default(
    no_secret: None, capsys: pytest.CaptureFixture[str]
) -> None:
    tokens.main(["--role", "adjuster", "--sub", "adj-1", "--compose-default"])

    token = capsys.readouterr().out.strip()
    assert jwt.decode(token, COMPOSE_DEV_SECRET, algorithms=["HS256"])["sub"] == "adj-1"


def test_the_compose_default_never_replaces_a_secret_that_is_set(
    capsys: pytest.CaptureFixture[str],
) -> None:
    tokens.main(["--role", "adjuster", "--sub", "adj-1", "--compose-default"])

    token = capsys.readouterr().out.strip()
    secret = get_settings().jwt_secret
    assert secret is not None
    assert jwt.decode(token, secret.get_secret_value(), algorithms=["HS256"])["sub"] == "adj-1"
