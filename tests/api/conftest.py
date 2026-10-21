"""API fixtures: the app on the worker tests' throwaway database, with storage in tmp_path.

The upload endpoint commits and the idempotency tests need a second request to see the first,
so these tests use the committed, truncate-per-test database rather than rolled-back sessions.
The app connects as claimcheck_app, exactly as deployed: only the engine is swapped, so the
request sessions carry the caller's identity and row-level security applies.
"""

import uuid
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Engine

from claimcheck.api.deps import get_engine, get_storage
from claimcheck.api.main import app
from claimcheck.api.tokens import issue_token
from claimcheck.core.types import Role
from claimcheck.db.models import Policy
from claimcheck.db.session import SessionFactory
from claimcheck.worker.storage import Storage
from tests.worker.conftest import (  # noqa: F401
    app_role_engine,
    engine,
    sessions,
    worker_db_url,
)

Headers = dict[str, str]


@pytest.fixture
def storage(tmp_path: Path) -> Storage:
    return Storage(tmp_path / "storage")


@pytest.fixture
def client(
    app_role_engine: Engine,  # noqa: F811
    sessions: SessionFactory,  # noqa: F811
    storage: Storage,
) -> Iterator[TestClient]:
    # `sessions` empties the tables before the test.
    app.dependency_overrides[get_engine] = lambda: app_role_engine
    app.dependency_overrides[get_storage] = lambda: storage
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.clear()


@pytest.fixture
def auth() -> Callable[[str, Role], Headers]:
    def headers(sub: str, role: Role) -> Headers:
        return {"Authorization": f"Bearer {issue_token(sub, role)}"}

    return headers


@pytest.fixture
def adjuster(auth: Callable[[str, Role], Headers]) -> Headers:
    return auth("adj-1", Role.ADJUSTER)


@pytest.fixture
def cust1(auth: Callable[[str, Role], Headers]) -> Headers:
    return auth("cust-1", Role.CUSTOMER)


@pytest.fixture
def cust2(auth: Callable[[str, Role], Headers]) -> Headers:
    return auth("cust-2", Role.CUSTOMER)


@pytest.fixture
def make_policy(sessions: SessionFactory) -> Callable[[str], str]:  # noqa: F811
    """Insert a policy for `holder_id` and return its number."""

    def make(holder_id: str) -> str:
        number = f"AUT-{uuid.uuid4().hex[:6]}"
        with sessions() as session:
            session.add(Policy(policy_number=number, holder_id=holder_id))
        return number

    return make


@pytest.fixture
def policy1(make_policy: Callable[[str], str]) -> str:
    return make_policy("cust-1")


@pytest.fixture
def policy2(make_policy: Callable[[str], str]) -> str:
    return make_policy("cust-2")
