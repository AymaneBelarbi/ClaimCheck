import time
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import jwt
import pytest
from fastapi.testclient import TestClient

from api import tokens
from api.auth import signing_secret

pytestmark = pytest.mark.db


def _secret() -> str:
    return signing_secret()


def _encode(claims: dict[str, Any], key: str | None = None, algorithm: str = "HS256") -> str:
    return jwt.encode(claims, key or _secret(), algorithm=algorithm)


def _valid_claims(**overrides: Any) -> dict[str, Any]:
    claims: dict[str, Any] = {
        "sub": "adj-1",
        "role": "adjuster",
        "exp": datetime.now(UTC) + timedelta(hours=1),
    }
    claims.update(overrides)
    return {k: v for k, v in claims.items() if v is not None}


def _get_claims(client: TestClient, token: str | None) -> Any:
    headers = {"Authorization": f"Bearer {token}"} if token is not None else {}
    return client.get("/claims", headers=headers)


def test_no_token_is_401_with_challenge(client: TestClient) -> None:
    response = client.get("/claims")
    assert response.status_code == 401
    assert response.headers["WWW-Authenticate"] == "Bearer"


def test_other_scheme_is_401(client: TestClient) -> None:
    response = client.get("/claims", headers={"Authorization": "Basic YWRqOnB3"})
    assert response.status_code == 401
    assert response.headers["WWW-Authenticate"] == "Bearer"


def test_valid_token_is_accepted(client: TestClient) -> None:
    assert _get_claims(client, _encode(_valid_claims())).status_code == 200


@pytest.mark.parametrize(
    "claims",
    [
        _valid_claims(exp=datetime.now(UTC) - timedelta(seconds=5)),
        _valid_claims(exp=None),
        _valid_claims(sub=None),
        _valid_claims(sub=""),
        _valid_claims(role=None),
        _valid_claims(role="root"),
    ],
    ids=["expired", "no-exp", "no-sub", "empty-sub", "no-role", "unknown-role"],
)
def test_bad_claims_are_401(client: TestClient, claims: dict[str, Any]) -> None:
    response = _get_claims(client, _encode(claims))
    assert response.status_code == 401
    assert response.headers["WWW-Authenticate"] == "Bearer"


def test_token_signed_with_another_key_is_401(client: TestClient) -> None:
    assert _get_claims(client, _encode(_valid_claims(), key="x" * 40)).status_code == 401


def test_other_algorithm_is_401(client: TestClient) -> None:
    token = _encode(_valid_claims(), algorithm="HS512")
    assert _get_claims(client, token).status_code == 401


def test_unsigned_token_is_401(client: TestClient) -> None:
    token = jwt.encode(_valid_claims(), "", algorithm="none")
    assert _get_claims(client, token).status_code == 401


def test_garbage_token_is_401(client: TestClient) -> None:
    assert _get_claims(client, "not.a.jwt").status_code == 401
    assert _get_claims(client, "").status_code == 401


def test_customer_cannot_use_adjuster_endpoints(client: TestClient) -> None:
    token = _encode(_valid_claims(sub="cust-1", role="customer"))
    headers = {"Authorization": f"Bearer {token}"}
    assert client.get(f"/claims/{uuid.uuid4()}/review", headers=headers).status_code == 403


def test_token_cli_prints_a_token_the_api_accepts(
    client: TestClient, capsys: pytest.CaptureFixture[str]
) -> None:
    tokens.main(["--role", "customer", "--sub", "cust-9", "--ttl-hours", "2"])
    token = capsys.readouterr().out.strip()

    claims = jwt.decode(token, _secret(), algorithms=["HS256"])
    assert claims["sub"] == "cust-9"
    assert claims["role"] == "customer"
    assert 2 * 3600 - 60 < claims["exp"] - time.time() <= 2 * 3600
    assert _get_claims(client, token).status_code == 200


def test_token_cli_defaults_to_twelve_hours(capsys: pytest.CaptureFixture[str]) -> None:
    tokens.main(["--role", "adjuster", "--sub", "adj-1"])
    claims = jwt.decode(capsys.readouterr().out.strip(), _secret(), algorithms=["HS256"])
    assert 12 * 3600 - 60 < claims["exp"] - time.time() <= 12 * 3600


def test_token_cli_rejects_unknown_role() -> None:
    with pytest.raises(SystemExit):
        tokens.main(["--role", "root", "--sub", "x"])
