"""Database fixtures: a throwaway database per test session, migrated with Alembic.

TEST_DATABASE_URL points at the server with a role that may create databases; the tests
never touch the database named in it.
"""

import os
import uuid
from collections.abc import Callable, Iterator

import pytest
from alembic import command
from alembic.config import Config
from pydantic import SecretStr
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL, make_url

from core.settings import Settings

# The migration gives each login role the password in these settings, so reading them the
# same way here keeps the tests and the server in step.
ROLE_PASSWORD_SETTING = {
    "claimcheck_app": "app_db_password",
    "claimcheck_worker": "worker_db_password",
}


def as_role(url: URL, role: str) -> URL:
    """`url` with the login of one of the roles the migration creates."""
    password: SecretStr = getattr(Settings(), ROLE_PASSWORD_SETTING[role])
    return url.set(username=role, password=password.get_secret_value())


def alembic_config(url: URL) -> Config:
    config = Config("alembic.ini")
    # configparser treats % as interpolation syntax.
    rendered = url.render_as_string(hide_password=False)
    config.set_main_option("sqlalchemy.url", rendered.replace("%", "%%"))
    return config


def _server_url() -> URL:
    return make_url(os.environ["TEST_DATABASE_URL"])


@pytest.fixture
def make_database() -> Iterator[Callable[[], URL]]:
    created: list[str] = []
    admin = create_engine(_server_url(), isolation_level="AUTOCOMMIT")

    def factory() -> URL:
        name = f"claimcheck_test_{uuid.uuid4().hex[:12]}"
        with admin.connect() as conn:
            conn.execute(text(f'CREATE DATABASE "{name}"'))
        created.append(name)
        return _server_url().set(database=name)

    yield factory

    with admin.connect() as conn:
        for name in created:
            conn.execute(text(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)'))
    admin.dispose()


@pytest.fixture(scope="session")
def migrated_url() -> Iterator[URL]:
    """One database, migrated to head, shared by every test that only needs the schema."""
    admin = create_engine(_server_url(), isolation_level="AUTOCOMMIT")
    name = f"claimcheck_test_{uuid.uuid4().hex[:12]}"
    with admin.connect() as conn:
        conn.execute(text(f'CREATE DATABASE "{name}"'))
    url = _server_url().set(database=name)
    try:
        command.upgrade(alembic_config(url), "head")
        yield url
    finally:
        with admin.connect() as conn:
            conn.execute(text(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)'))
        admin.dispose()


@pytest.fixture(scope="session")
def app_url(migrated_url: URL) -> URL:
    return as_role(migrated_url, "claimcheck_app")


@pytest.fixture(scope="session")
def worker_url(migrated_url: URL) -> URL:
    return as_role(migrated_url, "claimcheck_worker")


@pytest.fixture
def config_for() -> Callable[[URL], Config]:
    return alembic_config
