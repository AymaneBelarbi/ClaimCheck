"""A processed claim with one document, fields and rule results, for the tool and /ask tests."""

from datetime import date
from decimal import Decimal

from sqlalchemy.orm import Session

from claimcheck.core.types import ClaimStatus, DocType, Verdict
from claimcheck.db.models import Claim, ClaimDocument, ExtractedField, ValidationResult
from tests.worker.factories import add_claim, add_policy

CLAIMANT = "Jane Doe"


def add_reviewed_claim(
    session: Session,
    *,
    packet_id: str = "PKT-T1",
    incident_date: date = date(2024, 6, 1),
) -> Claim:
    """A claim in `needs_review` on a policy with two versions, the first in force on its date.

    Both versions carry clauses 4.1 (collision) and 4.2 (glass), so a search can only tell them
    apart by version number.
    """
    policy = add_policy(
        session,
        versions=((date(2024, 1, 1), date(2025, 1, 1)), (date(2025, 1, 1), None)),
        limits={"collision": Decimal("15000"), "glass": Decimal("1000")},
    )
    claim = add_claim(
        session,
        policy,
        packet_id=packet_id,
        incident_date=incident_date,
        status=ClaimStatus.NEEDS_REVIEW,
    )
    document = ClaimDocument(
        claim_id=claim.id,
        filename="claim_form.pdf",
        storage_key=f"{claim.id}/claim_form.pdf",
        doc_type=DocType.CLAIM_FORM.value,
        page_count=1,
        ocr_text=f"Claimant name: {CLAIMANT}\nPolicy number: {policy.policy_number}",
        ocr_mean_confidence=0.93,
        classification_confidence=0.88,
        classification_method="embedding",
    )
    session.add(document)
    session.flush()
    session.add(
        ExtractedField(
            claim_document_id=document.id,
            field_name="claimant_name",
            value=CLAIMANT,
            confidence=0.92,
            verdict=Verdict.FOUND.value,
            page=1,
            span_start=15,
            span_end=15 + len(CLAIMANT),
            source_text=CLAIMANT,
        )
    )
    session.add_all(
        [
            ValidationResult(
                claim_id=claim.id,
                rule_id="R1_NAME_MATCH",
                outcome=Verdict.FOUND.value,
                details={"message": "names match", "evidence": []},
            ),
            ValidationResult(
                claim_id=claim.id,
                rule_id="R6_REQUIRED_DOCUMENTS",
                outcome=Verdict.MISSING.value,
                details={"message": "missing: id_card, invoice, incident_report", "evidence": []},
            ),
        ]
    )
    session.flush()
    return claim
