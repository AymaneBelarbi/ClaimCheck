import io
import re
import threading
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from reportlab.pdfgen.canvas import Canvas
from sqlalchemy import func, select, update
from sqlalchemy.orm import Session

import pipeline.ocr
from core.types import ClaimStatus, DocType, Verdict
from db.models import (
    AuditLog,
    Claim,
    ClaimDocument,
    ExtractedField,
    Job,
    Policy,
    ValidationResult,
)
from db.session import SessionFactory
from evals.dataset.types import FIELD_NAMES, RULE_IDS
from evals.generate_dataset import generate
from gateway.base import LLMGateway
from gateway.redaction import RedactingGateway
from pipeline.classify import ClassificationResult
from pipeline.types import OcrResult
from rag.embedder import Embedder
from tests.fakes import HashingEmbedder
from tests.worker.factories import (
    add_claim,
    add_dataset_policy,
    add_document,
    add_packet,
    forced_classifier,
    gold_gateway,
    ground_truth,
)
from worker.main import run_once, serve
from worker.pipeline_runner import PipelineDeps, process_claim
from worker.queue import enqueue
from worker.review_file import build_review_file
from worker.storage import Storage

pytestmark = [pytest.mark.db, pytest.mark.ocr]

PACKET = "PKT-001"  # clean collision claim, four documents


def _later() -> datetime:
    """A worker clock safely past the database's now(), which stamps a new job's run_after."""
    return datetime.now(UTC) + timedelta(minutes=1)


def _much_later() -> datetime:
    """Past the first retry's 30 s backoff."""
    return datetime.now(UTC) + timedelta(minutes=5)


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("dataset")
    generate(out=root / "data", seed=20261020, questions_path=root / "questions.jsonl")
    return root / "data"


@pytest.fixture
def deps(data_dir: Path, tmp_path: Path) -> PipelineDeps:
    return PipelineDeps(
        storage=Storage(tmp_path / "storage"),
        embedder=HashingEmbedder(),
        llm=gold_gateway(data_dir),
        classifier=forced_classifier(data_dir),
    )


def _packet(
    sessions: SessionFactory,
    data_dir: Path,
    deps: PipelineDeps,
    skip: frozenset[str] = frozenset(),
    packet: str = PACKET,
) -> uuid.UUID:
    policy_number = ground_truth(data_dir, packet).policy_number
    with sessions() as session:
        policy = add_dataset_policy(session, data_dir, policy_number, HashingEmbedder())
        return add_packet(session, data_dir, packet, policy, deps.storage, skip=skip).id


def _count(session: Session, model: Any, *where: Any) -> int:
    return session.scalar(select(func.count()).select_from(model).where(*where)) or 0


def _derived(session: Session, claim_id: uuid.UUID) -> dict[str, Any]:
    docs = list(
        session.scalars(
            select(ClaimDocument)
            .where(ClaimDocument.claim_id == claim_id)
            .order_by(ClaimDocument.filename)
        )
    )
    fields = session.execute(
        select(ClaimDocument.filename, ExtractedField.field_name, ExtractedField.verdict)
        .join(ClaimDocument, ExtractedField.claim_document_id == ClaimDocument.id)
        .where(ClaimDocument.claim_id == claim_id)
        .order_by(ClaimDocument.filename, ExtractedField.field_name)
    ).all()
    rules = session.execute(
        select(ValidationResult.rule_id, ValidationResult.outcome)
        .where(ValidationResult.claim_id == claim_id)
        .order_by(ValidationResult.rule_id)
    ).all()
    return {
        "documents": [
            (d.filename, d.doc_type, d.page_count, d.low_confidence, d.error, d.ocr_text)
            for d in docs
        ],
        "fields": [tuple(row) for row in fields],
        "rules": [tuple(row) for row in rules],
    }


def _transitions(session: Session, claim_id: uuid.UUID) -> list[tuple[str, str]]:
    rows = session.scalars(
        select(AuditLog)
        .where(AuditLog.entity_id == str(claim_id), AuditLog.action == "claim.transition")
        .order_by(AuditLog.id)
    )
    return [(r.payload["from"], r.payload["to"]) for r in rows]


def test_clean_packet_reaches_needs_review_with_every_rule_found(
    sessions: SessionFactory, worker_sessions: SessionFactory, data_dir: Path, deps: PipelineDeps
) -> None:
    claim_id = _packet(sessions, data_dir, deps)

    process_claim(worker_sessions, claim_id, deps)

    truth = ground_truth(data_dir, PACKET)
    with sessions() as session:
        claim = session.get(Claim, claim_id)
        assert claim is not None and claim.status == ClaimStatus.NEEDS_REVIEW.value
        derived = _derived(session, claim_id)
        assert derived["rules"] == [(rule_id, "found") for rule_id in RULE_IDS]
        expected_types = {d.filename: d.doc_type.value for d in truth.documents}
        for filename, doc_type, pages, low, error, text in derived["documents"]:
            assert doc_type == expected_types[filename]
            assert pages == 1 and low is False and error is None and text
        expected_fields = sum(len(FIELD_NAMES[d.doc_type]) for d in truth.documents)
        assert len(derived["fields"]) == expected_fields
        assert _transitions(session, claim_id) == [
            ("received", "processing"),
            ("processing", "needs_review"),
        ]
    keys = {call.cache_key for call in deps.llm.calls}  # type: ignore[attr-defined]
    assert keys == {f"extract:{PACKET}:{d.filename}" for d in truth.documents}


def test_reprocessing_replaces_derived_rows(
    sessions: SessionFactory, worker_sessions: SessionFactory, data_dir: Path, deps: PipelineDeps
) -> None:
    claim_id = _packet(sessions, data_dir, deps)
    process_claim(worker_sessions, claim_id, deps)
    with sessions() as session:
        first = _derived(session, claim_id)
        # A packet that comes back for another run after a failure: the claim is failed but
        # still holds the rows of an earlier run.
        session.execute(
            update(Claim).where(Claim.id == claim_id).values(status=ClaimStatus.FAILED.value)
        )

    process_claim(worker_sessions, claim_id, deps)

    with sessions() as session:
        assert _derived(session, claim_id) == first
        claim = session.get(Claim, claim_id)
        assert claim is not None and claim.status == ClaimStatus.NEEDS_REVIEW.value


def test_claim_already_past_processing_is_left_alone(
    sessions: SessionFactory, worker_sessions: SessionFactory, data_dir: Path, deps: PipelineDeps
) -> None:
    claim_id = _packet(sessions, data_dir, deps)
    process_claim(worker_sessions, claim_id, deps)
    calls = len(deps.llm.calls)  # type: ignore[attr-defined]

    process_claim(worker_sessions, claim_id, deps)

    assert len(deps.llm.calls) == calls  # type: ignore[attr-defined]
    with sessions() as session:
        assert len(_transitions(session, claim_id)) == 2


def test_unreadable_files_go_to_review_and_the_job_completes(
    sessions: SessionFactory, worker_sessions: SessionFactory, data_dir: Path, deps: PipelineDeps
) -> None:
    """An empty file and a Word document in the packet do not fail the claim."""
    claim_id = _packet(sessions, data_dir, deps, skip=frozenset({"doc_2.png", "doc_3.pdf"}))
    with sessions() as session:
        claim = session.get(Claim, claim_id)
        assert claim is not None
        add_document(session, claim, deps.storage, "empty.pdf", b"")
        add_document(session, claim, deps.storage, "statement.docx", b"PK\x03\x04" + b"\0" * 64)
        enqueue(session, claim_id)

    assert run_once(worker_sessions, deps, clock=_later) is True

    with sessions() as session:
        job = session.scalars(select(Job).where(Job.claim_id == claim_id)).one()
        claim = session.get(Claim, claim_id)
        assert claim is not None and claim.status == ClaimStatus.NEEDS_REVIEW.value
        assert (job.state, job.attempts, job.last_error) == ("done", 1, None)
        docs = {
            d.filename: d
            for d in session.scalars(
                select(ClaimDocument).where(ClaimDocument.claim_id == claim_id)
            )
        }
        for name in ("empty.pdf", "statement.docx"):
            assert docs[name].doc_type == DocType.OTHER.value
            assert docs[name].low_confidence is True
            error = docs[name].error
            assert error is not None and name in error
            assert (
                _count(session, ExtractedField, ExtractedField.claim_document_id == docs[name].id)
                == 0
            )
        assert docs["doc_1.pdf"].doc_type == DocType.CLAIM_FORM.value
        assert docs["doc_4.pdf"].doc_type == DocType.INCIDENT_REPORT.value
        outcomes = dict(
            session.execute(
                select(ValidationResult.rule_id, ValidationResult.outcome).where(
                    ValidationResult.claim_id == claim_id
                )
            ).all()
        )
        assert outcomes["R6_REQUIRED_DOCUMENTS"] == Verdict.MISSING.value
        assert outcomes["R1_NAME_MATCH"] == Verdict.MISSING.value


def test_a_failed_run_is_retried_to_needs_review(
    sessions: SessionFactory, worker_sessions: SessionFactory, data_dir: Path, deps: PipelineDeps
) -> None:
    claim_id = _packet(sessions, data_dir, deps)
    with sessions() as session:
        enqueue(session, claim_id)
    real = deps.classifier
    failures = [RuntimeError("tesseract went away")]

    def flaky(
        ocr: OcrResult, embedder: Embedder, llm: LLMGateway | None, **kw: Any
    ) -> ClassificationResult:
        if failures:
            raise failures.pop()
        return real(ocr, embedder, llm, **kw)

    deps.classifier = flaky
    assert run_once(worker_sessions, deps, clock=_later) is True

    with sessions() as session:
        job = session.scalars(select(Job)).one()
        claim = session.get(Claim, claim_id)
        assert claim is not None and claim.status == ClaimStatus.FAILED.value
        assert (job.state, job.attempts) == ("queued", 1)
        assert job.last_error == "RuntimeError('tesseract went away')"
        assert _count(session, ValidationResult, ValidationResult.claim_id == claim_id) == 0

    assert run_once(worker_sessions, deps, clock=_later) is False  # backing off
    assert run_once(worker_sessions, deps, clock=_much_later) is True

    with sessions() as session:
        job = session.scalars(select(Job)).one()
        assert (job.state, job.attempts) == ("done", 2)
        assert _transitions(session, claim_id) == [
            ("received", "processing"),
            ("processing", "failed"),
            ("failed", "processing"),
            ("processing", "needs_review"),
        ]


def test_review_file_of_a_clean_packet_is_ready(
    sessions: SessionFactory, worker_sessions: SessionFactory, data_dir: Path, deps: PipelineDeps
) -> None:
    claim_id = _packet(sessions, data_dir, deps)
    process_claim(worker_sessions, claim_id, deps)

    with sessions() as session:
        review = build_review_file(session, claim_id)

    assert review.claim.packet_id == PACKET
    assert review.claim.status is ClaimStatus.NEEDS_REVIEW
    assert review.policy_version is not None and review.policy_version.version_no == 1
    assert review.summary.ready_for_decision is True
    assert review.summary.blocking == []
    assert review.summary.verdict_counts[Verdict.FOUND] == len(RULE_IDS)
    assert sum(review.summary.field_verdict_counts.values()) == len(review.fields)
    assert all(r.description for r in review.rules)
    by_id = {str(d.id): d for d in review.documents}
    for rule in review.rules:
        for evidence in rule.evidence:
            assert evidence.document_id in by_id


def test_stop_request_lets_the_current_job_finish(
    sessions: SessionFactory, worker_sessions: SessionFactory, data_dir: Path, deps: PipelineDeps
) -> None:
    claim_id = _packet(sessions, data_dir, deps)
    with sessions() as session:
        first = enqueue(session, claim_id)
        second = enqueue(session, add_claim(session, session.scalars(select(Policy)).one()).id)
        second.run_after = first.run_after + timedelta(seconds=1)
    stop = threading.Event()
    real = deps.classifier

    def stopping(
        ocr: OcrResult, embedder: Embedder, llm: LLMGateway | None, **kw: Any
    ) -> ClassificationResult:
        stop.set()  # what the SIGTERM handler does
        return real(ocr, embedder, llm, **kw)

    deps.classifier = stopping
    serve(worker_sessions, deps, stop, poll_interval=0.01, clock=_later)

    with sessions() as session:
        states = {j.claim_id: j.state for j in session.scalars(select(Job))}
        assert states == {claim_id: "done", second.claim_id: "queued"}


def test_a_cloud_backend_never_sees_the_packets_names_or_id_numbers(
    sessions: SessionFactory, worker_sessions: SessionFactory, data_dir: Path, deps: PipelineDeps
) -> None:
    """In this packet the licence prints the holder's name with no label the redactor knows,
    so only the packet scope (the claim form labels the same person) hides it in the licence's
    own request."""
    packet = "PKT-031"
    inner = gold_gateway(data_dir, is_cloud=True)
    # What gateway.factory does with any backend whose is_cloud is true.
    deps.llm = RedactingGateway(inner)
    claim_id = _packet(sessions, data_dir, deps, packet=packet)

    process_claim(worker_sessions, claim_id, deps)

    truth = ground_truth(data_dir, packet)
    by_type = {d.doc_type: d.fields for d in truth.documents}
    claimant = by_type[DocType.CLAIM_FORM]["claimant_name"]
    holder = by_type[DocType.ID_CARD]["full_name"]
    id_number = by_type[DocType.ID_CARD]["id_number"]
    assert claimant and holder and id_number
    secrets = {t.casefold() for t in f"{claimant} {holder}".split()} | {id_number.casefold()}
    sent = [f"{c.system}\n{c.prompt}" for c in inner.calls]
    assert len(sent) == len(truth.documents)
    for text in sent:
        words = set(re.findall(r"[\w-]+", text.casefold()))
        assert not secrets & words, secrets & words
    with sessions() as session:
        claim = session.get(Claim, claim_id)
        assert claim is not None and claim.status == ClaimStatus.NEEDS_REVIEW.value


def test_a_document_over_the_page_limit_goes_to_a_person(
    monkeypatch: pytest.MonkeyPatch,
    sessions: SessionFactory,
    worker_sessions: SessionFactory,
    data_dir: Path,
    deps: PipelineDeps,
) -> None:
    monkeypatch.setattr(pipeline.ocr, "MAX_PAGES", 1)
    claim_id = _packet(sessions, data_dir, deps, skip=frozenset({"doc_2.png", "doc_3.pdf"}))
    buffer = io.BytesIO()
    canvas = Canvas(buffer)
    for page in ("first", "second"):
        canvas.drawString(72, 700, f"Statement, {page} page")
        canvas.showPage()
    canvas.save()
    with sessions() as session:
        claim = session.get(Claim, claim_id)
        assert claim is not None
        add_document(session, claim, deps.storage, "long.pdf", buffer.getvalue())
        enqueue(session, claim_id)

    assert run_once(worker_sessions, deps, clock=_later) is True

    with sessions() as session:
        claim = session.get(Claim, claim_id)
        assert claim is not None and claim.status == ClaimStatus.NEEDS_REVIEW.value
        long = session.scalars(
            select(ClaimDocument).where(
                ClaimDocument.claim_id == claim_id, ClaimDocument.filename == "long.pdf"
            )
        ).one()
        assert (long.doc_type, long.low_confidence) == (DocType.OTHER.value, True)
        assert long.error is not None and "too many pages" in long.error
