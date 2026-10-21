"""Development token helper.

    python -m claimcheck.api.tokens --role adjuster --sub adj-1 [--ttl-hours 12] [--compose-default]

Signs with JWT_SECRET, so a token is only as good as that secret. Real deployments mint
tokens in their identity provider; this exists so the API can be exercised locally.

With --compose-default and no JWT_SECRET set (in the environment or .env), it signs with the
key docker-compose.yml falls back to in the same situation, so a token works against a stack
started without any configuration. `make token` passes it.
"""

import argparse
import sys
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta

import jwt

from claimcheck.api.auth import ALGORITHM, COMPOSE_DEV_SECRET, MissingSecretError, signing_secret
from claimcheck.core.types import Role


def compose_secret() -> str:
    """The key the compose stack checks tokens with: JWT_SECRET, else its development default."""
    try:
        return signing_secret()
    except MissingSecretError:
        return COMPOSE_DEV_SECRET


def issue_token(
    sub: str, role: Role, ttl: timedelta = timedelta(hours=12), *, secret: str | None = None
) -> str:
    claims = {"sub": sub, "role": role.value, "exp": datetime.now(UTC) + ttl}
    return jwt.encode(claims, secret or signing_secret(), algorithm=ALGORITHM)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m claimcheck.api.tokens", description=__doc__)
    parser.add_argument("--role", required=True, choices=[r.value for r in Role])
    parser.add_argument("--sub", required=True, help="subject, e.g. adj-1 or a policy holder id")
    parser.add_argument("--ttl-hours", type=float, default=12.0)
    parser.add_argument(
        "--compose-default",
        action="store_true",
        help="without JWT_SECRET, sign with docker-compose.yml's development default",
    )
    args = parser.parse_args(argv)
    secret = compose_secret() if args.compose_default else None
    try:
        token = issue_token(
            args.sub, Role(args.role), timedelta(hours=args.ttl_hours), secret=secret
        )
    except MissingSecretError as exc:
        sys.exit(str(exc))
    print(token)


if __name__ == "__main__":
    main()
