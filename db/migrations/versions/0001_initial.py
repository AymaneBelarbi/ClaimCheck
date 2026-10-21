"""Initial schema: policies, versions, clauses, claims, documents, fields, results, audit, jobs.

Revision ID: 0001
Revises:
"""

import uuid
from collections.abc import Sequence
from datetime import datetime

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from pydantic import SecretStr
from sqlalchemy.dialects import postgresql

from core.settings import Settings

revision = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# Allowed values are spelled out here, not imported: a migration describes the schema as it
# was when it ran, even if core.types changes later (add a new migration for that).
CLAIM_TYPES = ("collision", "glass", "theft")
CLAIM_STATUSES = (
    "received",
    "processing",
    "needs_review",
    "approved_for_review",
    "rejected_incomplete",
    "failed",
)
DOC_TYPES = ("claim_form", "id_card", "invoice", "incident_report", "other")
VERDICTS = ("found", "missing", "conflicting", "low_confidence")
JOB_STATES = ("queued", "running", "done", "dead")

# Login role -> the setting holding its password (APP_DB_PASSWORD, WORKER_DB_PASSWORD).
ROLES = {
    "claimcheck_app": "app_db_password",
    "claimcheck_worker": "worker_db_password",
}


def _in(column: str, values: tuple[str, ...]) -> str:
    return f"{column} IN ({', '.join(repr(v) for v in values)})"


def _id() -> sa.Column[uuid.UUID]:
    return sa.Column("id", sa.Uuid(), primary_key=True, server_default=sa.text("gen_random_uuid()"))


def _timestamp(name: str) -> sa.Column[datetime]:
    return sa.Column(name, sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now())


def _ensure_roles() -> None:
    # Roles belong to the cluster, not to one database, so they are created if missing and
    # their passwords refreshed from the settings (the environment, then .env). downgrade()
    # leaves them in place because other databases on the same server (including test
    # databases) may still use them.
    bind = op.get_bind()
    # A fresh read, not the cached settings: the password is wanted as it is now.
    settings = Settings()
    for role, setting in ROLES.items():
        password: SecretStr = getattr(settings, setting)
        bind.execute(
            sa.text("SELECT set_config('claimcheck.pw', :pw, true)"),
            {"pw": password.get_secret_value()},
        )
        bind.execute(
            sa.text(
                f"""
                DO $$
                BEGIN
                    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = '{role}') THEN
                        EXECUTE format('CREATE ROLE {role} LOGIN PASSWORD %L',
                                       current_setting('claimcheck.pw'));
                    ELSE
                        EXECUTE format('ALTER ROLE {role} LOGIN PASSWORD %L',
                                       current_setting('claimcheck.pw'));
                    END IF;
                END
                $$
                """
            )
        )


def _grant() -> None:
    mutable = (
        "policies, policy_versions, policy_clauses, claims, claim_documents, "
        "extracted_fields, validation_results, jobs"
    )
    for role in ROLES:
        op.execute(f"GRANT SELECT, INSERT, UPDATE ON {mutable} TO {role}")
        # Audit rows can be added and read, never changed. The owner keeps full rights for
        # operational repair; only the roles the services connect as are restricted.
        op.execute(f"GRANT SELECT, INSERT ON audit_log TO {role}")
        op.execute(f"REVOKE UPDATE, DELETE, TRUNCATE ON audit_log FROM {role}")
        op.execute(f"GRANT USAGE ON ALL SEQUENCES IN SCHEMA public TO {role}")


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.execute("CREATE EXTENSION IF NOT EXISTS btree_gist")

    op.create_table(
        "policies",
        _id(),
        sa.Column("policy_number", sa.Text(), nullable=False, unique=True),
        sa.Column("holder_id", sa.Text(), nullable=False),
        sa.Column("product_line", sa.Text(), nullable=False, server_default="auto"),
        _timestamp("created_at"),
    )

    op.create_table(
        "policy_versions",
        _id(),
        sa.Column("policy_id", sa.Uuid(), sa.ForeignKey("policies.id"), nullable=False),
        sa.Column("version_no", sa.Integer(), nullable=False),
        sa.Column("effective_from", sa.Date(), nullable=False),
        sa.Column("effective_to", sa.Date()),
        sa.Column("document_url", sa.Text()),
        sa.UniqueConstraint("policy_id", "version_no"),
        sa.CheckConstraint(
            "effective_to IS NULL OR effective_to > effective_from",
            name="ck_policy_versions_range",
        ),
    )
    op.execute(
        """
        ALTER TABLE policy_versions ADD CONSTRAINT ex_policy_versions_no_overlap
        EXCLUDE USING gist (
            policy_id WITH =,
            daterange(effective_from, effective_to, '[)') WITH &&
        )
        """
    )

    op.create_table(
        "policy_clauses",
        _id(),
        sa.Column(
            "policy_version_id", sa.Uuid(), sa.ForeignKey("policy_versions.id"), nullable=False
        ),
        sa.Column("clause_ref", sa.Text(), nullable=False),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("coverage_type", sa.Text()),
        sa.Column("limit_amount", sa.Numeric(12, 2)),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("embedding", Vector(384), nullable=False),
        sa.UniqueConstraint("policy_version_id", "clause_ref"),
    )
    op.create_index("ix_policy_clauses_policy_version_id", "policy_clauses", ["policy_version_id"])
    op.create_index(
        "ix_policy_clauses_embedding",
        "policy_clauses",
        ["embedding"],
        postgresql_using="hnsw",
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )

    op.create_table(
        "claims",
        _id(),
        sa.Column("packet_id", sa.Text(), nullable=False, unique=True),
        sa.Column("policy_id", sa.Uuid(), sa.ForeignKey("policies.id"), nullable=False),
        sa.Column("claim_type", sa.Text(), nullable=False),
        sa.Column("incident_date", sa.Date(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False, server_default="received"),
        sa.Column("created_by", sa.Text(), nullable=False),
        _timestamp("created_at"),
        _timestamp("updated_at"),
        sa.CheckConstraint(_in("claim_type", CLAIM_TYPES), name="ck_claims_claim_type"),
        sa.CheckConstraint(_in("status", CLAIM_STATUSES), name="ck_claims_status"),
    )
    op.create_index("ix_claims_policy_id", "claims", ["policy_id"])

    op.create_table(
        "claim_documents",
        _id(),
        sa.Column("claim_id", sa.Uuid(), sa.ForeignKey("claims.id"), nullable=False),
        sa.Column("filename", sa.Text(), nullable=False),
        sa.Column("doc_type", sa.Text()),
        sa.Column("storage_key", sa.Text(), nullable=False),
        sa.Column("page_count", sa.Integer()),
        sa.Column("ocr_text", sa.Text()),
        sa.Column("ocr_mean_confidence", sa.Float()),
        sa.Column("classification_confidence", sa.Float()),
        sa.Column("classification_method", sa.Text()),
        sa.Column("low_confidence", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("error", sa.Text()),
        sa.CheckConstraint(
            "doc_type IS NULL OR " + _in("doc_type", DOC_TYPES), name="ck_claim_documents_doc_type"
        ),
    )
    op.create_index("ix_claim_documents_claim_id", "claim_documents", ["claim_id"])

    op.create_table(
        "extracted_fields",
        _id(),
        sa.Column(
            "claim_document_id", sa.Uuid(), sa.ForeignKey("claim_documents.id"), nullable=False
        ),
        sa.Column("field_name", sa.Text(), nullable=False),
        sa.Column("value", sa.Text()),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("verdict", sa.Text(), nullable=False),
        sa.Column("page", sa.Integer()),
        sa.Column("span_start", sa.Integer()),
        sa.Column("span_end", sa.Integer()),
        sa.Column("source_text", sa.Text()),
        sa.Column("details", postgresql.JSONB()),
        sa.UniqueConstraint("claim_document_id", "field_name"),
        sa.CheckConstraint(_in("verdict", VERDICTS), name="ck_extracted_fields_verdict"),
    )

    op.create_table(
        "validation_results",
        _id(),
        sa.Column("claim_id", sa.Uuid(), sa.ForeignKey("claims.id"), nullable=False),
        sa.Column("rule_id", sa.Text(), nullable=False),
        sa.Column("outcome", sa.Text(), nullable=False),
        sa.Column(
            "details", postgresql.JSONB(), nullable=False, server_default=sa.text("'{}'::jsonb")
        ),
        sa.CheckConstraint(_in("outcome", VERDICTS), name="ck_validation_results_outcome"),
    )
    op.create_index("ix_validation_results_claim_id", "validation_results", ["claim_id"])

    op.create_table(
        "audit_log",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("actor", sa.Text(), nullable=False),
        sa.Column("action", sa.Text(), nullable=False),
        sa.Column("entity", sa.Text(), nullable=False),
        sa.Column("entity_id", sa.Text(), nullable=False),
        sa.Column(
            "payload", postgresql.JSONB(), nullable=False, server_default=sa.text("'{}'::jsonb")
        ),
        _timestamp("at"),
    )

    op.create_table(
        "jobs",
        _id(),
        sa.Column("claim_id", sa.Uuid(), sa.ForeignKey("claims.id"), nullable=False, unique=True),
        sa.Column("state", sa.Text(), nullable=False, server_default="queued"),
        sa.Column("attempts", sa.Integer(), nullable=False, server_default=sa.text("0")),
        _timestamp("run_after"),
        sa.Column("last_error", sa.Text()),
        _timestamp("updated_at"),
        sa.CheckConstraint(_in("state", JOB_STATES), name="ck_jobs_state"),
    )
    op.create_index("ix_jobs_state_run_after", "jobs", ["state", "run_after"])

    _ensure_roles()
    _grant()


def downgrade() -> None:
    # Roles are cluster-wide and deliberately kept; see _ensure_roles.
    for table in (
        "jobs",
        "audit_log",
        "validation_results",
        "extracted_fields",
        "claim_documents",
        "claims",
        "policy_clauses",
        "policy_versions",
        "policies",
    ):
        op.drop_table(table)
    op.execute("DROP EXTENSION IF EXISTS btree_gist")
    op.execute("DROP EXTENSION IF EXISTS vector")
