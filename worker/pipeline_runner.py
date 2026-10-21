"""Run one claim through OCR, classification, extraction and validation, and store the result.

Three phases, so no database transaction stays open while Tesseract or a model is working:

1. A short transaction moves the claim to `processing` and reads what the stages need.
2. The stages run with no transaction open.
3. One transaction replaces everything derived from an earlier run with the new rows and moves
   the claim to `needs_review`. Either all of it lands or none of it does, which is what makes
   running a claim again safe.
"""

import logging
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from typing import Any, Protocol

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from core.tracing import span as start_span
from core.types import ClaimStatus, ClaimType, DocType, Evidence, Verdict
from db.clauses import coverage_limit
from db.models import Claim, ClaimDocument, ExtractedField, ValidationResult
from db.policy_in_force import policy_in_force
from db.session import SessionFactory
from gateway.base import LLMGateway
from gateway.redaction import scope_gateway
from pipeline.classify import ClassificationResult, classify
from pipeline.extract import FieldResult, extract
from pipeline.ocr import run_ocr
from pipeline.types import OcrResult, UnsupportedDocumentError
from pipeline.validate import ClaimContext, RuleResult, run_rules
from pipeline.validate.base import DocFields, PolicyVersionInfo
from rag.embedder import Embedder
from worker.state_machine import transition
from worker.storage import Storage

logger = logging.getLogger(__name__)


class Classifier(Protocol):
    def __call__(
        self,
        ocr: OcrResult,
        embedder: Embedder,
        llm: LLMGateway | None,
        *,
        cache_key: str | None = None,
    ) -> ClassificationResult: ...


@dataclass
class PipelineDeps:
    storage: Storage
    embedder: Embedder
    llm: LLMGateway
    actor: str = "worker"
    classifier: Classifier = classify


class ClaimNotFound(LookupError):
    pass


@dataclass(frozen=True)
class _Document:
    id: uuid.UUID
    filename: str
    storage_key: str


@dataclass(frozen=True)
class _Inputs:
    packet_id: str
    claim_type: ClaimType
    incident_date: date
    documents: list[_Document]
    policy_version: PolicyVersionInfo | None
    coverage_limit: Decimal | None


@dataclass
class _Outcome:
    document: _Document
    doc_type: DocType = DocType.OTHER
    ocr: OcrResult | None = None
    classification: ClassificationResult | None = None
    error: str | None = None
    fields: list[FieldResult] = field(default_factory=list)

    @property
    def low_confidence(self) -> bool:
        return self.ocr is None or self.ocr.low_confidence


def process_claim(sessions: SessionFactory, claim_id: uuid.UUID, deps: PipelineDeps) -> None:
    """Process the claim and leave it in `needs_review`.

    A claim that is already past `processing` has been processed (a worker can die between
    storing the result and marking its job done) and is left as it is. Exceptions propagate
    with nothing stored, for the queue to retry.
    """
    # One trace per run of a claim: the stages and the model calls hang off this span.
    with start_span("claim.process", **{"claim.id": str(claim_id)}) as span:
        inputs = _start(sessions, claim_id, deps.actor)
        if inputs is None:
            return
        span.set_attribute("packet.id", inputs.packet_id)
        outcomes = _run_stages(inputs, deps)
        with start_span("stage.validate"):
            results = run_rules(_context(inputs, outcomes))
        _store(sessions, claim_id, outcomes, results, deps.actor)
    blocking = sum(1 for r in results if r.verdict is not Verdict.FOUND)
    logger.info(
        "claim %s (%s): %d documents, %d rules, %d not found",
        claim_id,
        inputs.packet_id,
        len(outcomes),
        len(results),
        blocking,
    )


def _start(sessions: SessionFactory, claim_id: uuid.UUID, actor: str) -> _Inputs | None:
    with sessions() as session:
        claim = session.get(Claim, claim_id, with_for_update=True)
        if claim is None:
            raise ClaimNotFound(f"claim {claim_id} does not exist")
        status = ClaimStatus(claim.status)
        if status in (ClaimStatus.RECEIVED, ClaimStatus.FAILED):
            transition(
                session, claim, ClaimStatus.PROCESSING, actor=actor, reason="pipeline started"
            )
        elif status is ClaimStatus.PROCESSING:
            # The previous attempt's worker died without recording a failure.
            logger.warning("claim %s was already processing; starting over", claim_id)
        else:
            logger.info("claim %s is %s already; nothing to do", claim_id, status.value)
            return None

        documents = [
            _Document(d.id, d.filename, d.storage_key)
            for d in session.scalars(
                select(ClaimDocument)
                .where(ClaimDocument.claim_id == claim_id)
                .order_by(ClaimDocument.filename, ClaimDocument.id)
            )
        ]
        version = policy_in_force(session, claim.policy_id, claim.incident_date)
        limit = coverage_limit(session, version.id, claim.claim_type) if version else None
        return _Inputs(
            packet_id=claim.packet_id,
            claim_type=ClaimType(claim.claim_type),
            incident_date=claim.incident_date,
            documents=documents,
            policy_version=(
                PolicyVersionInfo(version.version_no, version.effective_from, version.effective_to)
                if version
                else None
            ),
            coverage_limit=limit,
        )


def _run_stages(inputs: _Inputs, deps: PipelineDeps) -> list[_Outcome]:
    outcomes = [_Outcome(doc) for doc in inputs.documents]
    for outcome in outcomes:
        doc = outcome.document
        with start_span("stage.ocr", **{"document.id": str(doc.id)}):
            try:
                outcome.ocr = run_ocr(deps.storage.get(doc.storage_key), doc.filename)
            except UnsupportedDocumentError as exc:
                # Not a failure of the claim: the file goes to a person as `other`.
                outcome.error = str(exc)
                logger.info("claim document %s (%s) is unreadable: %s", doc.id, doc.filename, exc)

    # Every document's text, so a cloud backend's redactor knows the packet's names.
    gateway = scope_gateway(deps.llm, (o.ocr.text for o in outcomes if o.ocr is not None))
    for outcome in outcomes:
        if outcome.ocr is None:
            continue
        doc = outcome.document
        with start_span("stage.classify", **{"document.id": str(doc.id)}) as span:
            outcome.classification = deps.classifier(
                outcome.ocr,
                deps.embedder,
                gateway,
                cache_key=f"classify:{inputs.packet_id}:{doc.filename}",
            )
            outcome.doc_type = outcome.classification.doc_type
            span.set_attribute("doc_type", outcome.doc_type.value)
        if outcome.doc_type is DocType.OTHER:
            continue
        with start_span(
            "stage.extract", **{"document.id": str(doc.id), "doc_type": outcome.doc_type.value}
        ):
            outcome.fields = extract(
                outcome.doc_type,
                outcome.ocr,
                str(doc.id),
                gateway,
                cache_key=f"extract:{inputs.packet_id}:{doc.filename}",
            )
    return outcomes


def _context(inputs: _Inputs, outcomes: Sequence[_Outcome]) -> ClaimContext:
    return ClaimContext(
        claim_type=inputs.claim_type,
        incident_date=inputs.incident_date,
        documents=[
            DocFields(str(o.document.id), o.doc_type, {f.field_name: f for f in o.fields})
            for o in outcomes
        ],
        policy_version=inputs.policy_version,
        coverage_limit=inputs.coverage_limit,
    )


def _store(
    sessions: SessionFactory,
    claim_id: uuid.UUID,
    outcomes: Sequence[_Outcome],
    results: Sequence[RuleResult],
    actor: str,
) -> None:
    with sessions() as session:
        claim = session.get(Claim, claim_id, with_for_update=True)
        if claim is None:
            raise ClaimNotFound(f"claim {claim_id} does not exist")
        _clear(session, claim_id)
        for outcome in outcomes:
            _store_document(session, outcome)
        for result in results:
            session.add(
                ValidationResult(
                    claim_id=claim_id,
                    rule_id=result.rule_id,
                    outcome=result.verdict.value,
                    details={
                        "message": result.message,
                        "evidence": [evidence_json(e) for e in result.evidence],
                    },
                )
            )
        blocking = [r.rule_id for r in results if r.verdict is not Verdict.FOUND]
        reason = f"{len(results)} rules run; not found: {', '.join(blocking) or 'none'}"
        transition(session, claim, ClaimStatus.NEEDS_REVIEW, actor=actor, reason=reason)


def _clear(session: Session, claim_id: uuid.UUID) -> None:
    document_ids = select(ClaimDocument.id).where(ClaimDocument.claim_id == claim_id)
    session.execute(
        delete(ExtractedField).where(ExtractedField.claim_document_id.in_(document_ids))
    )
    session.execute(delete(ValidationResult).where(ValidationResult.claim_id == claim_id))


def _store_document(session: Session, outcome: _Outcome) -> None:
    document = session.get(ClaimDocument, outcome.document.id)
    if document is None:
        return
    ocr, classification = outcome.ocr, outcome.classification
    # Every derived column is written, so nothing from an earlier run survives.
    document.doc_type = outcome.doc_type.value
    document.ocr_text = ocr.text if ocr else None
    document.page_count = len(ocr.pages) if ocr else None
    document.ocr_mean_confidence = ocr.mean_confidence if ocr else None
    document.classification_confidence = classification.confidence if classification else None
    document.classification_method = classification.method if classification else None
    document.low_confidence = outcome.low_confidence
    document.error = outcome.error
    for result in outcome.fields:
        evidence = result.evidence
        span = evidence.span if evidence else None
        session.add(
            ExtractedField(
                claim_document_id=document.id,
                field_name=result.field_name,
                value=result.value,
                confidence=result.confidence,
                verdict=result.verdict.value,
                page=evidence.page if evidence else None,
                span_start=span[0] if span else None,
                span_end=span[1] if span else None,
                source_text=evidence.text if evidence else None,
                details={"reason": result.details} if result.details else None,
            )
        )


def evidence_json(evidence: Evidence) -> dict[str, Any]:
    return {
        "document_id": evidence.document_id,
        "page": evidence.page,
        "span": list(evidence.span) if evidence.span else None,
        "text": evidence.text,
    }
