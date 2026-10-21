"""Row-level security, checked in SQL as the roles the services log in as.

The owner (a superuser in development and CI) arranges the rows; every assertion runs as
claimcheck_app or claimcheck_worker with the `app.role` / `app.user_id` settings the services
put on each transaction.
"""

import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import pytest
from sqlalchemy import Engine, create_engine, text
from sqlalchemy.engine import URL, Connection
from sqlalchemy.exc import ProgrammingError

pytestmark = pytest.mark.db

RLS_TABLES = ("claims", "claim_documents", "extracted_fields", "validation_results")


@dataclass(frozen=True)
class Claim:
    id: uuid.UUID
    document_id: uuid.UUID
    field_id: uuid.UUID
    result_id: uuid.UUID


@dataclass(frozen=True)
class Pair:
    """Two customers, each with one claim, one document, one field and one rule result."""

    a_sub: str
    b_sub: str
    a: Claim
    b: Claim


@pytest.fixture(scope="module")
def owner(migrated_url: URL) -> Iterator[Engine]:
    engine = create_engine(migrated_url)
    yield engine
    engine.dispose()


@pytest.fixture(scope="module")
def app(app_url: URL) -> Iterator[Engine]:
    engine = create_engine(app_url)
    yield engine
    engine.dispose()


@pytest.fixture(scope="module")
def worker(worker_url: URL) -> Iterator[Engine]:
    engine = create_engine(worker_url)
    yield engine
    engine.dispose()


def _add_claim(conn: Connection, policy_id: uuid.UUID, created_by: str) -> Claim:
    claim_id = conn.execute(
        text(
            "INSERT INTO claims (packet_id, policy_id, claim_type, incident_date, created_by)"
            " VALUES (:k, :p, 'glass', '2026-01-01', :by) RETURNING id"
        ),
        {"k": f"PKT-{uuid.uuid4().hex[:10]}", "p": policy_id, "by": created_by},
    ).scalar_one()
    document_id = conn.execute(
        text(
            "INSERT INTO claim_documents (claim_id, filename, storage_key)"
            " VALUES (:c, 'form.pdf', 'k') RETURNING id"
        ),
        {"c": claim_id},
    ).scalar_one()
    field_id = conn.execute(
        text(
            "INSERT INTO extracted_fields (claim_document_id, field_name, confidence, verdict)"
            " VALUES (:d, 'claimant_name', 0.9, 'found') RETURNING id"
        ),
        {"d": document_id},
    ).scalar_one()
    result_id = conn.execute(
        text(
            "INSERT INTO validation_results (claim_id, rule_id, outcome)"
            " VALUES (:c, 'R1', 'found') RETURNING id"
        ),
        {"c": claim_id},
    ).scalar_one()
    return Claim(claim_id, document_id, field_id, result_id)


@pytest.fixture
def pair(owner: Engine) -> Pair:
    a_sub, b_sub = f"cust-{uuid.uuid4().hex[:8]}", f"cust-{uuid.uuid4().hex[:8]}"
    with owner.begin() as conn:
        policy_id = conn.execute(
            text("INSERT INTO policies (policy_number, holder_id) VALUES (:n, :h) RETURNING id"),
            {"n": f"AUT-{uuid.uuid4().hex[:8]}", "h": a_sub},
        ).scalar_one()
        return Pair(
            a_sub, b_sub, _add_claim(conn, policy_id, a_sub), _add_claim(conn, policy_id, b_sub)
        )


@contextmanager
def acting_as(engine: Engine, role: str | None, user_id: str | None) -> Iterator[Connection]:
    """A transaction with the identity settings a service sets, or none at all; rolled back."""
    with engine.connect() as conn, conn.begin() as txn:
        if role is not None:
            conn.execute(text("SELECT set_config('app.role', :r, true)"), {"r": role})
        if user_id is not None:
            conn.execute(text("SELECT set_config('app.user_id', :u, true)"), {"u": user_id})
        yield conn
        txn.rollback()


def _ids(conn: Connection, table: str) -> set[uuid.UUID]:
    return set(conn.execute(text(f"SELECT id FROM {table}")).scalars())


def _visible(conn: Connection, claim: Claim) -> dict[str, bool]:
    return {
        "claims": claim.id in _ids(conn, "claims"),
        "claim_documents": claim.document_id in _ids(conn, "claim_documents"),
        "extracted_fields": claim.field_id in _ids(conn, "extracted_fields"),
        "validation_results": claim.result_id in _ids(conn, "validation_results"),
    }


ALL = dict.fromkeys(RLS_TABLES, True)
NONE = dict.fromkeys(RLS_TABLES, False)


def test_customer_sees_only_their_own_claims_and_what_hangs_off_them(
    app: Engine, pair: Pair
) -> None:
    with acting_as(app, "customer", pair.a_sub) as conn:
        assert _visible(conn, pair.a) == ALL
        assert _visible(conn, pair.b) == NONE
        assert _ids(conn, "claims") == {pair.a.id}
        assert _ids(conn, "claim_documents") == {pair.a.document_id}
        assert _ids(conn, "extracted_fields") == {pair.a.field_id}
        assert _ids(conn, "validation_results") == {pair.a.result_id}


def test_customer_cannot_update_another_customers_claim(app: Engine, pair: Pair) -> None:
    with acting_as(app, "customer", pair.a_sub) as conn:
        updated = conn.execute(
            text("UPDATE claims SET status = 'failed' WHERE id = :id"), {"id": pair.b.id}
        )
        assert updated.rowcount == 0


def test_customer_cannot_insert_a_claim_for_someone_else(app: Engine, pair: Pair) -> None:
    with acting_as(app, "customer", pair.a_sub) as conn:
        policy_id = conn.execute(
            text("SELECT policy_id FROM claims WHERE id = :id"), {"id": pair.a.id}
        ).scalar_one()
        with pytest.raises(ProgrammingError, match="row-level security"):
            conn.execute(
                text(
                    "INSERT INTO claims (packet_id, policy_id, claim_type, incident_date,"
                    " created_by) VALUES (:k, :p, 'glass', '2026-01-01', :by)"
                ),
                {"k": f"PKT-{uuid.uuid4().hex[:10]}", "p": policy_id, "by": pair.b_sub},
            )


def test_customer_cannot_attach_a_document_to_another_customers_claim(
    app: Engine, pair: Pair
) -> None:
    with (
        acting_as(app, "customer", pair.a_sub) as conn,
        pytest.raises(ProgrammingError, match="row-level security"),
    ):
        conn.execute(
            text(
                "INSERT INTO claim_documents (claim_id, filename, storage_key)"
                " VALUES (:c, 'x.pdf', 'k')"
            ),
            {"c": pair.b.id},
        )


def test_adjuster_sees_every_claim(app: Engine, pair: Pair) -> None:
    with acting_as(app, "adjuster", "adj-1") as conn:
        assert _visible(conn, pair.a) == ALL
        assert _visible(conn, pair.b) == ALL


@pytest.mark.parametrize(
    ("role", "user_id"),
    [(None, None), (None, "cust-x"), ("", ""), ("customer", None), ("superuser", "adj-1")],
)
def test_without_a_known_role_nothing_is_visible(
    app: Engine, pair: Pair, role: str | None, user_id: str | None
) -> None:
    with acting_as(app, role, user_id) as conn:
        for table in RLS_TABLES:
            assert conn.execute(text(f"SELECT count(*) FROM {table}")).scalar_one() == 0, table


def test_only_the_worker_login_gets_the_system_identity(
    app: Engine, worker: Engine, pair: Pair
) -> None:
    with acting_as(app, "system", "worker") as conn:
        for table in RLS_TABLES:
            assert conn.execute(text(f"SELECT count(*) FROM {table}")).scalar_one() == 0, table
    with acting_as(worker, "system", "worker") as conn:
        assert _visible(conn, pair.a) == ALL
        assert _visible(conn, pair.b) == ALL


def test_identity_does_not_outlive_its_transaction(app: Engine, pair: Pair) -> None:
    with app.connect() as conn:
        with conn.begin():
            conn.execute(text("SELECT set_config('app.role', 'adjuster', true)"))
            assert conn.execute(text("SELECT count(*) FROM claims")).scalar_one() >= 2
        with conn.begin():
            assert conn.execute(text("SELECT count(*) FROM claims")).scalar_one() == 0


def test_worker_role_can_replace_derived_rows(worker: Engine, pair: Pair) -> None:
    with acting_as(worker, "system", "worker") as conn:
        fields = conn.execute(
            text("DELETE FROM extracted_fields WHERE claim_document_id = :d"),
            {"d": pair.b.document_id},
        )
        results = conn.execute(
            text("DELETE FROM validation_results WHERE claim_id = :c"), {"c": pair.b.id}
        )
        assert (fields.rowcount, results.rowcount) == (1, 1)
        conn.execute(
            text(
                "INSERT INTO validation_results (claim_id, rule_id, outcome)"
                " VALUES (:c, 'R1', 'missing')"
            ),
            {"c": pair.b.id},
        )
        claims = conn.execute(
            text("UPDATE claims SET status = 'processing' WHERE id = :c"), {"c": pair.b.id}
        )
        assert claims.rowcount == 1


def test_worker_role_without_identity_changes_nothing(worker: Engine, pair: Pair) -> None:
    with acting_as(worker, None, None) as conn:
        deleted = conn.execute(
            text("DELETE FROM validation_results WHERE claim_id = :c"), {"c": pair.a.id}
        )
        assert deleted.rowcount == 0


@pytest.mark.parametrize(
    "statement",
    [
        "UPDATE policy_versions SET document_url = 'x'",
        "DELETE FROM policy_versions",
        "UPDATE policies SET holder_id = 'x'",
        "INSERT INTO policies (policy_number, holder_id) VALUES ('P-x', 'h')",
        "UPDATE policy_clauses SET text = 'x'",
        "DELETE FROM claims",
        "DELETE FROM extracted_fields",
        "INSERT INTO validation_results (claim_id, rule_id, outcome)"
        " SELECT id, 'R9', 'found' FROM claims",
        "UPDATE extracted_fields SET value = 'x'",
        # The API inserts documents and jobs but never changes them; the worker does.
        "UPDATE claim_documents SET error = 'x'",
        "UPDATE jobs SET state = 'queued'",
        "DELETE FROM jobs",
        # The audit log is append-only for both services.
        "UPDATE audit_log SET actor = 'x'",
        "DELETE FROM audit_log",
    ],
)
def test_app_role_lacks_privileges_it_does_not_need(app: Engine, statement: str) -> None:
    with (
        acting_as(app, "adjuster", "adj-1") as conn,
        pytest.raises(ProgrammingError, match="permission denied"),
    ):
        conn.execute(text(statement))


@pytest.mark.parametrize(
    "statement",
    [
        "UPDATE policy_versions SET document_url = 'x'",
        "INSERT INTO claims (packet_id, policy_id, claim_type, incident_date, created_by)"
        " SELECT 'PKT-w', policy_id, claim_type, incident_date, created_by FROM claims LIMIT 1",
        "INSERT INTO claim_documents (claim_id, filename, storage_key)"
        " SELECT id, 'x.pdf', 'k' FROM claims LIMIT 1",
        "INSERT INTO jobs (claim_id) SELECT id FROM claims LIMIT 1",
        "DELETE FROM claims",
        "DELETE FROM jobs",
        "UPDATE audit_log SET actor = 'x'",
    ],
)
def test_worker_role_lacks_privileges_it_does_not_need(worker: Engine, statement: str) -> None:
    with (
        acting_as(worker, "system", "worker") as conn,
        pytest.raises(ProgrammingError, match="permission denied"),
    ):
        conn.execute(text(statement))


def test_rls_is_enabled_and_forced(owner: Engine) -> None:
    with owner.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT relname, relrowsecurity, relforcerowsecurity FROM pg_class"
                " WHERE relname = ANY(:t) AND relkind = 'r'"
            ),
            {"t": list(RLS_TABLES)},
        ).all()
    assert sorted(rows) == sorted((t, True, True) for t in RLS_TABLES)


def test_service_roles_cannot_bypass_rls(owner: Engine) -> None:
    with owner.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT rolname, rolbypassrls, rolsuper FROM pg_roles"
                " WHERE rolname IN ('claimcheck_app', 'claimcheck_worker')"
            )
        ).all()
    assert sorted(rows) == [("claimcheck_app", False, False), ("claimcheck_worker", False, False)]
