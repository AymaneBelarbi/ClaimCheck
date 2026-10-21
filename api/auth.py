"""Bearer-token authentication and role checks.

Tokens are HS256 JWTs carrying `sub`, `role` and `exp`; all three are required. They are
minted outside the API (see api.tokens for the development helper).
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Annotated

import jwt
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from starlette.datastructures import Headers
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from core.settings import get_settings
from core.types import Role

ALGORITHM = "HS256"

# What docker-compose.yml signs with when JWT_SECRET is not set, so `docker compose up` works on
# a fresh clone. Like the value in .env.example, it is published, so anyone can sign with it.
COMPOSE_DEV_SECRET = "dev-only-change-me-not-a-secret-0123456789"
PLACEHOLDER_SECRETS = frozenset({COMPOSE_DEV_SECRET, "replace-me-with-a-long-random-string"})


class MissingSecretError(RuntimeError):
    """JWT_SECRET is not set, so no token can be signed or checked."""


def signing_secret() -> str:
    secret = get_settings().jwt_secret
    if secret is None or not secret.get_secret_value():
        raise MissingSecretError(
            "JWT_SECRET is not set; the API signs and checks tokens with it (see .env.example)"
        )
    return secret.get_secret_value()


# auto_error is off so that a missing header gets the same 401 as a bad token.
_bearer = HTTPBearer(auto_error=False)


@dataclass(frozen=True)
class Principal:
    sub: str
    role: Role


def _unauthorized(detail: str) -> HTTPException:
    return HTTPException(status_code=401, detail=detail, headers={"WWW-Authenticate": "Bearer"})


def decode_token(token: str) -> Principal:
    try:
        claims = jwt.decode(
            token,
            signing_secret(),
            algorithms=[ALGORITHM],
            options={"require": ["sub", "role", "exp"]},
        )
    except jwt.InvalidTokenError as exc:
        raise _unauthorized("invalid or expired token") from exc
    sub = claims["sub"]
    if not isinstance(sub, str) or not sub:
        raise _unauthorized("invalid token subject")
    try:
        role = Role(claims["role"])
    except ValueError as exc:
        raise _unauthorized("invalid token role") from exc
    return Principal(sub=sub, role=role)


def current_principal(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
) -> Principal:
    if credentials is None:
        raise _unauthorized("missing bearer token")
    return decode_token(credentials.credentials)


def require_role(*roles: Role) -> Callable[[Principal], Principal]:
    """A dependency that lets only the given roles through."""

    def check(principal: Annotated[Principal, Depends(current_principal)]) -> Principal:
        if principal.role not in roles:
            raise HTTPException(status_code=403, detail="not permitted for this role")
        return principal

    return check


class BearerPrecheckMiddleware:
    """Turn away unauthenticated writes before their bodies are read.

    FastAPI parses a multipart form (spooling uploads to disk) before it resolves dependencies,
    so the `current_principal` check alone would come after a large upload had been accepted.
    This only checks the token; the dependencies stay the authority on who may do what. GET,
    HEAD and OPTIONS are left alone: they carry no body, and OPTIONS is the CORS preflight.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http" and scope["method"] not in {"GET", "HEAD", "OPTIONS"}:
            scheme, _, token = Headers(scope=scope).get("authorization", "").partition(" ")
            try:
                if scheme.lower() != "bearer" or not token:
                    raise _unauthorized("missing bearer token")
                decode_token(token)
            except HTTPException as exc:
                response = JSONResponse(
                    {"detail": exc.detail}, status_code=exc.status_code, headers=exc.headers
                )
                await response(scope, receive, send)
                return
        await self.app(scope, receive, send)
