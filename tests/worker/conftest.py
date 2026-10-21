"""Worker fixtures: a database of their own, emptied before every test.

The pipeline commits between its phases and the queue leases whatever job is due, so these
tests cannot share the rolled-back sessions the other database tests use.

`sessions` connects as the owner and is for arranging and inspecting rows. Code under test
runs as the role its service logs in as: `worker_sessions` (claimcheck_worker, the worker's
`system` identity) and `adjuster_sessions` (claimcheck_app as an adjuster), so the grants and
row-level security of migration 0002 apply to it as they do in production.
"""

import os
import uuid
from collections.abc import Iterator

import pytest
from alembic import command
from sqlalchemy import Engine, create_engine, text
from sqlalchemy.engine import URL, make_url

from core.types import Role
from db.session import SYSTEM, SessionFactory, session_factory
from tests.db.conftest import alembic_config, as_role

TABLES = (
    "audit_log, jobs, validation_results, extracted_fields, claim_documents, claims, "
    "policy_clauses, policy_versions, policies"
)


@pytest.fixture(scope="session")
def worker_db_url() -> Iterator[URL]:
    server = make_url(os.environ["TEST_DATABASE_URL"])
    admin = create_engine(server, isolation_level="AUTOCOMMIT")
    name = f"claimcheck_worker_test_{uuid.uuid4().hex[:12]}"
    with admin.connect() as conn:
        conn.execute(text(f'CREATE DATABASE "{name}"'))
    url = server.set(database=name)
    try:
        command.upgrade(alembic_config(url), "head")
        yield url
    finally:
        with admin.connect() as conn:
            conn.execute(text(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)'))
        admin.dispose()


@pytest.fixture(scope="session")
def engine(worker_db_url: URL) -> Iterator[Engine]:
    eng = create_engine(worker_db_url)
    yield eng
    eng.dispose()


@pytest.fixture(scope="session")
def app_role_engine(worker_db_url: URL) -> Iterator[Engine]:
    eng = create_engine(as_role(worker_db_url, "claimcheck_app"))
    yield eng
    eng.dispose()


@pytest.fixture(scope="session")
def worker_role_engine(worker_db_url: URL) -> Iterator[Engine]:
    eng = create_engine(as_role(worker_db_url, "claimcheck_worker"))
    yield eng
    eng.dispose()


@pytest.fixture
def sessions(engine: Engine) -> SessionFactory:
    with engine.begin() as conn:
        conn.execute(text(f"TRUNCATE {TABLES} RESTART IDENTITY"))
    # Every session carries an identity; the owner, a superuser here, is not bound by it.
    return session_factory(engine, SYSTEM, "test-setup")


@pytest.fixture
def worker_sessions(sessions: SessionFactory, worker_role_engine: Engine) -> SessionFactory:
    """What the worker process uses. Depends on `sessions` so the tables are emptied first."""
    return session_factory(worker_role_engine, SYSTEM, "worker")


@pytest.fixture
def adjuster_sessions(sessions: SessionFactory, app_role_engine: Engine) -> SessionFactory:
    """What the API's agent tools and the MCP server use."""
    return session_factory(app_role_engine, Role.ADJUSTER, "adj-1")
