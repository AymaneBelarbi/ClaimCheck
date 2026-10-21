import uuid
from datetime import date
from decimal import Decimal

import pytest
from sqlalchemy.orm import Session

from claimcheck.core.review import ReviewFile
from claimcheck.core.types import ClaimStatus, DocType, Evidence, Verdict
from claimcheck.db.models import Claim, ClaimDocument, ExtractedField, ValidationResult
from claimcheck.db.session import SessionFactory
from claimcheck.pipeline.validate import ALL_RULES
from claimcheck.worker.pipeline_runner import ClaimNotFound
from claimcheck.worker.review_file import build_review_file
from tests.worker.factories import add_claim, add_policy

pytestmark = pytest.mark.db


def _document(session: Session, claim: Claim, filename: str, doc_type: DocType) -> ClaimDocument:
    document = ClaimDocument(
        claim_id=claim.id,
        filename=filename,
        storage_key=f"{claim.id}/{filename}",
        doc_type=doc_type.value,
        page_count=1,
        ocr_text="text",
        ocr_mean_confidence=0.91,
        classification_confidence=0.8,
        classification_method="embedding",
    )
    session.add(document)
    session.flush()
    return document


def _field(
    session: Session, document: ClaimDocument, name: str, value: str | None, verdict: Verdict
) -> None:
    session.add(
        ExtractedField(
            claim_document_id=document.id,
            field_name=name,
            value=value,
            confidence=0.9 if verdict is Verdict.FOUND else 0.4,
            verdict=verdict.value,
            page=1,
            span_start=10,
            span_end=10 + len(value or ""),
            source_text=value,
        )
    )


def _rule(
    session: Session, claim: Claim, rule_id: str, verdict: Verdict, evidence: list[Evidence]
) -> None:
    session.add(
        ValidationResult(
            claim_id=claim.id,
            rule_id=rule_id,
            outcome=verdict.value,
            details={
                "message": f"{rule_id} said {verdict.value}",
                "evidence": [
                    {
                        "document_id": e.document_id,
                        "page": e.page,
                        "span": list(e.span) if e.span else None,
                        "text": e.text,
                    }
                    for e in evidence
                ],
            },
        )
    )


def _claim_without_id_card(session: Session) -> Claim:
    policy = add_policy(
        session,
        versions=((date(2024, 1, 1), date(2025, 1, 1)), (date(2025, 1, 1), None)),
        limits={"glass": Decimal("1500.00")},
    )
    claim = add_claim(
        session,
        policy,
        claim_type="glass",
        incident_date=date(2025, 3, 2),
        status=ClaimStatus.NEEDS_REVIEW,
    )
    form = _document(session, claim, "form.pdf", DocType.CLAIM_FORM)
    invoice = _document(session, claim, "invoice.pdf", DocType.INVOICE)
    _field(session, form, "claimant_name", "Ana Ruiz", Verdict.FOUND)
    _field(session, form, "incident_date", None, Verdict.LOW_CONFIDENCE)
    _field(session, invoice, "total", "820.00", Verdict.FOUND)
    name_evidence = Evidence(str(form.id), 1, (10, 18), "Ana Ruiz")
    _rule(session, claim, "R6_REQUIRED_DOCUMENTS", Verdict.MISSING, [])
    _rule(session, claim, "R1_NAME_MATCH", Verdict.MISSING, [name_evidence])
    _rule(session, claim, "R2_INCIDENT_IN_COVERAGE", Verdict.LOW_CONFIDENCE, [])
    _rule(session, claim, "R3_INVOICE_WITHIN_LIMIT", Verdict.FOUND, [])
    session.flush()
    return claim


def test_missing_id_card_blocks_the_decision(sessions: SessionFactory) -> None:
    with sessions() as session:
        claim = _claim_without_id_card(session)
        review = build_review_file(session, claim.id)

    assert review.summary.ready_for_decision is False
    assert review.summary.blocking == [
        "R1_NAME_MATCH",
        "R2_INCIDENT_IN_COVERAGE",
        "R6_REQUIRED_DOCUMENTS",
    ]
    assert review.summary.verdict_counts == {
        Verdict.FOUND: 1,
        Verdict.MISSING: 2,
        Verdict.CONFLICTING: 0,
        Verdict.LOW_CONFIDENCE: 1,
    }
    assert review.summary.field_verdict_counts == {
        Verdict.FOUND: 2,
        Verdict.MISSING: 0,
        Verdict.CONFLICTING: 0,
        Verdict.LOW_CONFIDENCE: 1,
    }


def test_review_file_carries_claim_policy_documents_fields_and_rules(
    sessions: SessionFactory,
) -> None:
    with sessions() as session:
        claim = _claim_without_id_card(session)
        review = build_review_file(session, claim.id)
        form = next(d for d in review.documents if d.filename == "form.pdf")

    assert review.claim.id == claim.id
    assert review.claim.status is ClaimStatus.NEEDS_REVIEW
    assert review.claim.incident_date == date(2025, 3, 2)
    assert review.policy_version is not None
    assert (review.policy_version.version_no, review.policy_version.effective_to) == (2, None)
    assert [d.filename for d in review.documents] == ["form.pdf", "invoice.pdf"]
    assert form.doc_type is DocType.CLAIM_FORM
    assert form.ocr_mean_confidence == pytest.approx(0.91)

    name = next(f for f in review.fields if f.field_name == "claimant_name")
    assert name.document_id == form.id
    assert name.doc_type is DocType.CLAIM_FORM
    assert name.evidence == Evidence(str(form.id), 1, (10, 18), "Ana Ruiz")

    rules = {r.rule_id: r for r in review.rules}
    descriptions = {rule.rule_id: rule.description for rule in ALL_RULES}
    assert rules["R1_NAME_MATCH"].description == descriptions["R1_NAME_MATCH"]
    assert rules["R1_NAME_MATCH"].message == "R1_NAME_MATCH said missing"
    assert rules["R1_NAME_MATCH"].evidence == [Evidence(str(form.id), 1, (10, 18), "Ana Ruiz")]
    assert [r.rule_id for r in review.rules] == sorted(rules)


def test_review_file_round_trips_as_json(sessions: SessionFactory) -> None:
    with sessions() as session:
        review = build_review_file(session, _claim_without_id_card(session).id)

    assert ReviewFile.model_validate_json(review.model_dump_json()) == review


def test_unknown_claim(sessions: SessionFactory) -> None:
    with sessions() as session, pytest.raises(ClaimNotFound):
        build_review_file(session, uuid.uuid4())


@pytest.mark.parametrize("status", [ClaimStatus.FAILED, ClaimStatus.PROCESSING])
def test_a_claim_without_rule_results_is_not_ready(
    sessions: SessionFactory, status: ClaimStatus
) -> None:
    with sessions() as session:
        policy = add_policy(session, versions=((date(2024, 1, 1), None),))
        claim = add_claim(session, policy, status=status)
        _document(session, claim, "form.pdf", DocType.CLAIM_FORM)
        review = build_review_file(session, claim.id)

    assert review.rules == []
    assert review.summary.blocking == []
    assert review.summary.ready_for_decision is False
