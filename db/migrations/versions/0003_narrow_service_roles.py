"""Bind the `system` identity to the worker's login, and drop grants the API never uses.

Migration 0002 let any transaction that set `app.role` to `system` see every claim. Only the
worker sets it, but the API's role could have set it too. The visibility policies now accept
`system` only when the session logged in as claimcheck_worker, so a bug or an injected setting
in the API cannot widen what it sees.

The API inserts claim documents and jobs and never changes them (enqueueing is an INSERT ...
ON CONFLICT DO NOTHING), so claimcheck_app loses UPDATE on both. The worker keeps it.

Revision ID: 0003
Revises: 0002
"""

from collections.abc import Sequence

from alembic import op

revision = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_ROLE = "current_setting('app.role', true)"
_USER = "current_setting('app.user_id', true)"
_CUSTOMER = f"{_ROLE} = 'customer'"
_STAFF_0002 = f"{_ROLE} IN ('adjuster', 'system')"
_STAFF = f"({_ROLE} = 'adjuster' OR ({_ROLE} = 'system' AND current_user = 'claimcheck_worker'))"


def _visible(staff: str) -> dict[str, str]:
    """Who may see a row, per table, as in 0002 with `staff` as the staff test."""
    return {
        "claims": f"{staff} OR ({_CUSTOMER} AND claims.created_by = {_USER})",
        "claim_documents": f"""{staff} OR ({_CUSTOMER} AND EXISTS (
            SELECT 1 FROM claims c
            WHERE c.id = claim_documents.claim_id AND c.created_by = {_USER}))""",
        "extracted_fields": f"""{staff} OR ({_CUSTOMER} AND EXISTS (
            SELECT 1 FROM claim_documents d JOIN claims c ON c.id = d.claim_id
            WHERE d.id = extracted_fields.claim_document_id AND c.created_by = {_USER}))""",
        "validation_results": f"""{staff} OR ({_CUSTOMER} AND EXISTS (
            SELECT 1 FROM claims c
            WHERE c.id = validation_results.claim_id AND c.created_by = {_USER}))""",
    }


def _replace_policies(staff: str) -> None:
    for table, visible in _visible(staff).items():
        op.execute(f"DROP POLICY {table}_visibility ON {table}")
        op.execute(
            f"CREATE POLICY {table}_visibility ON {table} FOR ALL"
            f" USING ({visible}) WITH CHECK ({visible})"
        )


def upgrade() -> None:
    _replace_policies(_STAFF)
    op.execute("REVOKE UPDATE ON claim_documents, jobs FROM claimcheck_app")


def downgrade() -> None:
    op.execute("GRANT UPDATE ON claim_documents, jobs TO claimcheck_app")
    _replace_policies(_STAFF_0002)
