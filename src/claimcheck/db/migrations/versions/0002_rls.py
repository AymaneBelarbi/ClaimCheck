"""Row-level security on claim data, and least-privilege grants for the service roles.

A row of claims, claim_documents, extracted_fields or validation_results is visible only when
the transaction says who is asking: `app.role` is `adjuster` or `system` (the worker), or it is
`customer` and the claim was created for `app.user_id`. Services set both with
set_config(..., true) at the start of every transaction (see claimcheck.db.session). A
transaction that sets neither sees no rows, and an insert or update that would produce a row
the caller could not see is refused.

FORCE applies the policies to the table owner too. A superuser still bypasses them, which is
why the services never connect as one.

Revision ID: 0002
Revises: 0001
"""

from collections.abc import Sequence

from alembic import op

revision = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

ROLES = ("claimcheck_app", "claimcheck_worker")

_ROLE = "current_setting('app.role', true)"
_USER = "current_setting('app.user_id', true)"
_STAFF = f"{_ROLE} IN ('adjuster', 'system')"
_CUSTOMER = f"{_ROLE} = 'customer'"

# Who may see a row, per table. A customer reaches a child row through the claim it belongs to.
VISIBLE = {
    "claims": f"{_STAFF} OR ({_CUSTOMER} AND claims.created_by = {_USER})",
    "claim_documents": f"""{_STAFF} OR ({_CUSTOMER} AND EXISTS (
        SELECT 1 FROM claims c
        WHERE c.id = claim_documents.claim_id AND c.created_by = {_USER}))""",
    "extracted_fields": f"""{_STAFF} OR ({_CUSTOMER} AND EXISTS (
        SELECT 1 FROM claim_documents d JOIN claims c ON c.id = d.claim_id
        WHERE d.id = extracted_fields.claim_document_id AND c.created_by = {_USER}))""",
    "validation_results": f"""{_STAFF} OR ({_CUSTOMER} AND EXISTS (
        SELECT 1 FROM claims c
        WHERE c.id = validation_results.claim_id AND c.created_by = {_USER}))""",
}

POLICY_TABLES = "policies, policy_versions, policy_clauses"

# Exactly what each service does. Policy versions are never changed or deleted by a service;
# the seed and ingest commands run as the owner.
GRANTS = {
    "claimcheck_app": (
        f"SELECT ON {POLICY_TABLES}",
        "SELECT, INSERT, UPDATE ON claims, claim_documents, jobs",
        "SELECT ON extracted_fields, validation_results",
        "SELECT, INSERT ON audit_log",
    ),
    "claimcheck_worker": (
        f"SELECT ON {POLICY_TABLES}",
        "SELECT, UPDATE ON claims, claim_documents, jobs",
        "SELECT, INSERT, DELETE ON extracted_fields, validation_results",
        "SELECT, INSERT ON audit_log",
    ),
}
# audit_log.id is the only serial column.
SEQUENCE = "audit_log_id_seq"

ALL_TABLES = (
    f"{POLICY_TABLES}, claims, claim_documents, extracted_fields, validation_results, "
    "audit_log, jobs"
)


def upgrade() -> None:
    for role in ROLES:
        op.execute(f"REVOKE ALL ON {ALL_TABLES} FROM {role}")
        op.execute(f"REVOKE ALL ON ALL SEQUENCES IN SCHEMA public FROM {role}")
        for grant in GRANTS[role]:
            op.execute(f"GRANT {grant} TO {role}")
        op.execute(f"GRANT USAGE ON SEQUENCE {SEQUENCE} TO {role}")

    for table, visible in VISIBLE.items():
        op.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")
        op.execute(f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY")
        # One policy for every command: USING filters what is read, updated and deleted;
        # WITH CHECK refuses inserted or updated rows the caller would not be allowed to see.
        op.execute(
            f"CREATE POLICY {table}_visibility ON {table} FOR ALL"
            f" USING ({visible}) WITH CHECK ({visible})"
        )


def downgrade() -> None:
    for table in VISIBLE:
        op.execute(f"DROP POLICY IF EXISTS {table}_visibility ON {table}")
        op.execute(f"ALTER TABLE {table} NO FORCE ROW LEVEL SECURITY")
        op.execute(f"ALTER TABLE {table} DISABLE ROW LEVEL SECURITY")

    # Back to the grants of 0001.
    mutable = (
        f"{POLICY_TABLES}, claims, claim_documents, extracted_fields, validation_results, jobs"
    )
    for role in ROLES:
        op.execute(f"REVOKE ALL ON {ALL_TABLES} FROM {role}")
        op.execute(f"GRANT SELECT, INSERT, UPDATE ON {mutable} TO {role}")
        op.execute(f"GRANT SELECT, INSERT ON audit_log TO {role}")
        op.execute(f"GRANT USAGE ON ALL SEQUENCES IN SCHEMA public TO {role}")
