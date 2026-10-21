"""Build a claim's review file from what the pipeline stored."""

import uuid
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from claimcheck.core.review import (
    ReviewClaim,
    ReviewDocument,
    ReviewField,
    ReviewFile,
    ReviewPolicyVersion,
    ReviewRule,
    summarize,
)
from claimcheck.core.types import ClaimStatus, ClaimType, DocType, Evidence, Verdict
from claimcheck.db.models import Claim, ClaimDocument, ExtractedField, ValidationResult
from claimcheck.db.policy_in_force import policy_in_force
from claimcheck.pipeline.validate import ALL_RULES
from claimcheck.worker.pipeline_runner import ClaimNotFound

_DESCRIPTIONS = {rule.rule_id: rule.description for rule in ALL_RULES}


def build_review_file(session: Session, claim_id: uuid.UUID) -> ReviewFile:
    claim = session.get(Claim, claim_id)
    if claim is None:
        raise ClaimNotFound(f"claim {claim_id} does not exist")
    version = policy_in_force(session, claim.policy_id, claim.incident_date)

    documents = list(
        session.scalars(
            select(ClaimDocument)
            .where(ClaimDocument.claim_id == claim_id)
            .order_by(ClaimDocument.filename, ClaimDocument.id)
        )
    )
    doc_types = {d.id: _doc_type(d.doc_type) for d in documents}
    order = {d.id: index for index, d in enumerate(documents)}
    stored_fields = sorted(
        session.scalars(
            select(ExtractedField).where(ExtractedField.claim_document_id.in_(list(order)))
        ),
        key=lambda f: (order[f.claim_document_id], f.field_name),
    )
    stored_rules = session.scalars(
        select(ValidationResult)
        .where(ValidationResult.claim_id == claim_id)
        .order_by(ValidationResult.rule_id)
    )

    fields = [
        ReviewField(
            document_id=f.claim_document_id,
            doc_type=doc_types[f.claim_document_id],
            field_name=f.field_name,
            value=f.value,
            confidence=f.confidence,
            verdict=Verdict(f.verdict),
            evidence=Evidence(
                str(f.claim_document_id),
                f.page,
                (f.span_start, f.span_end)
                if f.span_start is not None and f.span_end is not None
                else None,
                f.source_text,
            ),
        )
        for f in stored_fields
    ]
    rules = [
        ReviewRule(
            rule_id=r.rule_id,
            description=_DESCRIPTIONS.get(r.rule_id, ""),
            verdict=Verdict(r.outcome),
            message=str(r.details.get("message", "")),
            evidence=[_evidence(e) for e in r.details.get("evidence", [])],
        )
        for r in stored_rules
    ]
    return ReviewFile(
        claim=ReviewClaim(
            id=claim.id,
            packet_id=claim.packet_id,
            status=ClaimStatus(claim.status),
            claim_type=ClaimType(claim.claim_type),
            incident_date=claim.incident_date,
        ),
        policy_version=(
            ReviewPolicyVersion(
                version_no=version.version_no,
                effective_from=version.effective_from,
                effective_to=version.effective_to,
            )
            if version
            else None
        ),
        documents=[
            ReviewDocument(
                id=d.id,
                filename=d.filename,
                doc_type=doc_types[d.id],
                classification_confidence=d.classification_confidence,
                classification_method=d.classification_method,
                ocr_mean_confidence=d.ocr_mean_confidence,
                low_confidence=d.low_confidence,
                error=d.error,
            )
            for d in documents
        ],
        fields=fields,
        rules=rules,
        summary=summarize(ClaimStatus(claim.status), fields, rules),
    )


def _doc_type(value: str | None) -> DocType | None:
    return DocType(value) if value is not None else None


def _evidence(stored: dict[str, Any]) -> Evidence:
    span = stored.get("span")
    return Evidence(
        document_id=str(stored["document_id"]),
        page=stored.get("page"),
        span=(int(span[0]), int(span[1])) if span else None,
        text=stored.get("text"),
    )
