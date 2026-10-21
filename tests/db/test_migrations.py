import datetime as dt
import uuid
from collections.abc import Callable

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import URL, Connection
from sqlalchemy.exc import IntegrityError, ProgrammingError

from core.types import Role
from db.session import session_scope
from tests.db.conftest import as_role

pytestmark = pytest.mark.db

EXPECTED_TABLES = {
    "policies",
    "policy_versions",
    "policy_clauses",
    "claims",
    "claim_documents",
    "extracted_fields",
    "validation_results",
    "audit_log",
    "jobs",
}


def _url(url: URL) -> str:
    return url.render_as_string(hide_password=False)


def test_upgrade_creates_all_tables(migrated_url: URL) -> None:
    engine = create_engine(migrated_url)
    try:
        tables = set(inspect(engine).get_table_names()) - {"alembic_version"}
    finally:
        engine.dispose()
    assert tables == EXPECTED_TABLES


def test_downgrade_roundtrip(
    make_database: Callable[[], URL], config_for: Callable[[URL], Config]
) -> None:
    config = config_for(make_database())
    command.upgrade(config, "head")
    command.downgrade(config, "base")
    command.upgrade(config, "head")


def _insert_policy(conn: Connection, number: str) -> uuid.UUID:
    row = conn.execute(
        text("INSERT INTO policies (policy_number, holder_id) VALUES (:n, 'h-1') RETURNING id"),
        {"n": number},
    ).one()
    pid: uuid.UUID = row.id
    return pid


_INSERT_VERSION = text(
    "INSERT INTO policy_versions (policy_id, version_no, effective_from, effective_to)"
    " VALUES (:p, :v, :f, :t)"
)


def test_overlapping_versions_rejected(migrated_url: URL) -> None:
    engine = create_engine(migrated_url)
    try:
        with engine.connect() as conn:
            pid = _insert_policy(conn, f"P-{uuid.uuid4().hex[:8]}")
            conn.execute(
                _INSERT_VERSION,
                {"p": pid, "v": 1, "f": dt.date(2025, 1, 1), "t": dt.date(2026, 1, 1)},
            )
            with pytest.raises(IntegrityError):
                conn.execute(
                    _INSERT_VERSION,
                    {"p": pid, "v": 2, "f": dt.date(2025, 6, 1), "t": None},
                )
            conn.rollback()
    finally:
        engine.dispose()


def test_adjacent_versions_allowed(migrated_url: URL) -> None:
    # [from, to) ranges: a version may start on the day the previous one ends.
    engine = create_engine(migrated_url)
    try:
        with engine.connect() as conn:
            pid = _insert_policy(conn, f"P-{uuid.uuid4().hex[:8]}")
            conn.execute(
                _INSERT_VERSION,
                {"p": pid, "v": 1, "f": dt.date(2025, 1, 1), "t": dt.date(2026, 1, 1)},
            )
            conn.execute(
                _INSERT_VERSION,
                {"p": pid, "v": 2, "f": dt.date(2026, 1, 1), "t": None},
            )
            conn.rollback()
    finally:
        engine.dispose()


def test_audit_log_is_append_only(app_url: URL) -> None:
    engine = create_engine(app_url)
    try:
        with engine.connect() as conn:
            conn.execute(
                text(
                    "INSERT INTO audit_log (actor, action, entity, entity_id)"
                    " VALUES ('t', 'a', 'claim', '1')"
                )
            )
            conn.commit()
            with pytest.raises(ProgrammingError, match="permission denied"):
                conn.execute(text("UPDATE audit_log SET actor = 'x'"))
            conn.rollback()
            with pytest.raises(ProgrammingError, match="permission denied"):
                conn.execute(text("DELETE FROM audit_log"))
            conn.rollback()
    finally:
        engine.dispose()


def test_check_constraints_reject_unknown_values(migrated_url: URL) -> None:
    engine = create_engine(migrated_url)
    try:
        with engine.connect() as conn:
            pid = _insert_policy(conn, f"P-{uuid.uuid4().hex[:8]}")
            with pytest.raises(IntegrityError):
                conn.execute(
                    text(
                        "INSERT INTO claims (packet_id, policy_id, claim_type, incident_date,"
                        " created_by) VALUES ('pk', :p, 'flood', '2026-01-01', 'u')"
                    ),
                    {"p": pid},
                )
            conn.rollback()
    finally:
        engine.dispose()


def test_session_scope_sets_transaction_local_identity(app_url: URL) -> None:
    probe = text("SELECT current_setting('app.role', true), current_setting('app.user_id', true)")
    with session_scope(Role.CUSTOMER, "cust-1", url=_url(app_url)) as session:
        assert session.execute(probe).one() == ("customer", "cust-1")
    # A later transaction on the pooled connection must not inherit it.
    with session_scope(Role.ADJUSTER, "adj-1", url=_url(app_url)) as session:
        assert session.execute(probe).one() == ("adjuster", "adj-1")


def test_models_match_migration(migrated_url: URL) -> None:
    """Autogenerate against the migrated schema finds nothing to change."""
    from alembic.autogenerate import compare_metadata
    from alembic.migration import MigrationContext

    from db.models import Base

    engine = create_engine(migrated_url)
    try:
        with engine.connect() as conn:
            context = MigrationContext.configure(conn)
            diff = compare_metadata(context, Base.metadata)
    finally:
        engine.dispose()
    assert diff == []


def test_role_passwords_come_from_the_settings(
    make_database: Callable[[], URL],
    config_for: Callable[[URL], Config],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """APP_DB_PASSWORD is read like every other setting, so .env applies to `make migrate`."""
    url = make_database()
    usual = as_role(url, "claimcheck_app").password
    monkeypatch.setenv("APP_DB_PASSWORD", "app-password-from-settings")
    try:
        command.upgrade(config_for(url), "0001")
        login = url.set(username="claimcheck_app", password="app-password-from-settings")
        engine = create_engine(login)
        with engine.connect() as conn:
            assert conn.execute(text("SELECT current_user")).scalar_one() == "claimcheck_app"
        engine.dispose()
    finally:
        # Roles belong to the whole server; put back the password the other tests log in with.
        owner = create_engine(url)
        with owner.begin() as conn:
            conn.execute(text(f"ALTER ROLE claimcheck_app PASSWORD '{usual}'"))
        owner.dispose()
