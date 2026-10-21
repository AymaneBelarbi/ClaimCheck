"""SQLAlchemy models. Postgres is the source of truth; the clause embeddings are a derived index.

Enum-like columns are plain text with CHECK constraints built from `claimcheck.core.types`,
which keeps migrations free of Postgres enum types (those are awkward to alter).
"""

import uuid
from datetime import date, datetime
from decimal import Decimal
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    Text,
    UniqueConstraint,
    Uuid,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, ExcludeConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from claimcheck.core.types import ClaimStatus, ClaimType, DocType, Verdict

EMBEDDING_DIM = 384
JOB_STATES = ("queued", "running", "done", "dead")


def _one_of(column: str, values: tuple[str, ...] | type[Any]) -> str:
    members = values if isinstance(values, tuple) else tuple(str(v.value) for v in values)
    return f"{column} IN ({', '.join(repr(m) for m in members)})"


def _uuid_pk() -> Mapped[uuid.UUID]:
    return mapped_column(
        Uuid, primary_key=True, default=uuid.uuid4, server_default=text("gen_random_uuid()")
    )


def _created_at() -> Mapped[datetime]:
    return mapped_column(DateTime(timezone=True), server_default=func.now())


class Base(DeclarativeBase):
    pass


class Policy(Base):
    __tablename__ = "policies"

    id: Mapped[uuid.UUID] = _uuid_pk()
    policy_number: Mapped[str] = mapped_column(Text, unique=True)
    holder_id: Mapped[str] = mapped_column(Text)
    product_line: Mapped[str] = mapped_column(Text, default="auto", server_default="auto")
    created_at: Mapped[datetime] = _created_at()


class PolicyVersion(Base):
    __tablename__ = "policy_versions"
    __table_args__ = (
        UniqueConstraint("policy_id", "version_no"),
        CheckConstraint(
            "effective_to IS NULL OR effective_to > effective_from",
            name="ck_policy_versions_range",
        ),
        # A policy has at most one version in force on any day.
        ExcludeConstraint(
            ("policy_id", "="),
            (func.daterange(text("effective_from"), text("effective_to"), "[)"), "&&"),
            using="gist",
            name="ex_policy_versions_no_overlap",
        ),
    )

    id: Mapped[uuid.UUID] = _uuid_pk()
    policy_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("policies.id"))
    version_no: Mapped[int] = mapped_column(Integer)
    effective_from: Mapped[date] = mapped_column(Date)
    # NULL means the version is still current.
    effective_to: Mapped[date | None] = mapped_column(Date)
    document_url: Mapped[str | None] = mapped_column(Text)


class PolicyClause(Base):
    __tablename__ = "policy_clauses"
    __table_args__ = (
        UniqueConstraint("policy_version_id", "clause_ref"),
        Index(
            "ix_policy_clauses_embedding",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )

    id: Mapped[uuid.UUID] = _uuid_pk()
    policy_version_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("policy_versions.id"), index=True
    )
    clause_ref: Mapped[str] = mapped_column(Text)
    title: Mapped[str] = mapped_column(Text)
    coverage_type: Mapped[str | None] = mapped_column(Text)
    limit_amount: Mapped[Decimal | None] = mapped_column(Numeric(12, 2))
    text: Mapped[str] = mapped_column(Text)
    embedding: Mapped[Any] = mapped_column(Vector(EMBEDDING_DIM))


class Claim(Base):
    __tablename__ = "claims"
    __table_args__ = (
        CheckConstraint(_one_of("claim_type", ClaimType), name="ck_claims_claim_type"),
        CheckConstraint(_one_of("status", ClaimStatus), name="ck_claims_status"),
    )

    id: Mapped[uuid.UUID] = _uuid_pk()
    # Idempotency key: submitting the same packet twice yields the same claim.
    packet_id: Mapped[str] = mapped_column(Text, unique=True)
    policy_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("policies.id"), index=True)
    claim_type: Mapped[str] = mapped_column(Text)
    incident_date: Mapped[date] = mapped_column(Date)
    status: Mapped[str] = mapped_column(
        Text, default=ClaimStatus.RECEIVED.value, server_default=ClaimStatus.RECEIVED.value
    )
    created_by: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = _created_at()
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class ClaimDocument(Base):
    __tablename__ = "claim_documents"
    __table_args__ = (
        CheckConstraint(
            "doc_type IS NULL OR " + _one_of("doc_type", DocType),
            name="ck_claim_documents_doc_type",
        ),
    )

    id: Mapped[uuid.UUID] = _uuid_pk()
    claim_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("claims.id"), index=True)
    filename: Mapped[str] = mapped_column(Text)
    # NULL until the classifier has run.
    doc_type: Mapped[str | None] = mapped_column(Text)
    storage_key: Mapped[str] = mapped_column(Text)
    page_count: Mapped[int | None] = mapped_column(Integer)
    ocr_text: Mapped[str | None] = mapped_column(Text)
    ocr_mean_confidence: Mapped[float | None] = mapped_column(Float)
    classification_confidence: Mapped[float | None] = mapped_column(Float)
    classification_method: Mapped[str | None] = mapped_column(Text)
    low_confidence: Mapped[bool] = mapped_column(
        Boolean, default=False, server_default=text("false")
    )
    error: Mapped[str | None] = mapped_column(Text)


class ExtractedField(Base):
    __tablename__ = "extracted_fields"
    __table_args__ = (
        UniqueConstraint("claim_document_id", "field_name"),
        CheckConstraint(_one_of("verdict", Verdict), name="ck_extracted_fields_verdict"),
    )

    id: Mapped[uuid.UUID] = _uuid_pk()
    claim_document_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("claim_documents.id"))
    field_name: Mapped[str] = mapped_column(Text)
    value: Mapped[str | None] = mapped_column(Text)
    confidence: Mapped[float] = mapped_column(Float)
    verdict: Mapped[str] = mapped_column(Text)
    page: Mapped[int | None] = mapped_column(Integer)
    span_start: Mapped[int | None] = mapped_column(Integer)
    span_end: Mapped[int | None] = mapped_column(Integer)
    source_text: Mapped[str | None] = mapped_column(Text)
    details: Mapped[dict[str, Any] | None] = mapped_column(JSONB)


class ValidationResult(Base):
    __tablename__ = "validation_results"
    __table_args__ = (
        CheckConstraint(_one_of("outcome", Verdict), name="ck_validation_results_outcome"),
        Index("ix_validation_results_claim_id", "claim_id"),
    )

    id: Mapped[uuid.UUID] = _uuid_pk()
    claim_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("claims.id"))
    rule_id: Mapped[str] = mapped_column(Text)
    outcome: Mapped[str] = mapped_column(Text)
    # {"message": str, "evidence": [...]}
    details: Mapped[dict[str, Any]] = mapped_column(
        JSONB, default=dict, server_default=text("'{}'::jsonb")
    )


class AuditLog(Base):
    """Append-only: the migration withholds UPDATE and DELETE from the application roles."""

    __tablename__ = "audit_log"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    actor: Mapped[str] = mapped_column(Text)
    action: Mapped[str] = mapped_column(Text)
    entity: Mapped[str] = mapped_column(Text)
    entity_id: Mapped[str] = mapped_column(Text)
    payload: Mapped[dict[str, Any]] = mapped_column(
        JSONB, default=dict, server_default=text("'{}'::jsonb")
    )
    at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Job(Base):
    __tablename__ = "jobs"
    __table_args__ = (
        CheckConstraint(_one_of("state", JOB_STATES), name="ck_jobs_state"),
        Index("ix_jobs_state_run_after", "state", "run_after"),
    )

    id: Mapped[uuid.UUID] = _uuid_pk()
    claim_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("claims.id"), unique=True)
    state: Mapped[str] = mapped_column(Text, default="queued", server_default="queued")
    attempts: Mapped[int] = mapped_column(Integer, default=0, server_default=text("0"))
    run_after: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    last_error: Mapped[str | None] = mapped_column(Text)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
