"""Rows and doubles for the worker tests, built from the generated dataset where possible.

`add_dataset_policy` and `add_packet` are also what a manual end-to-end run uses to put one
packet into a database before the real seed command exists.
"""

import json
import uuid
from collections.abc import Callable
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any

from pydantic import BaseModel
from sqlalchemy.orm import Session

from claimcheck.core.types import ClaimStatus
from claimcheck.db.models import Claim, ClaimDocument, Policy, PolicyClause, PolicyVersion
from claimcheck.gateway.base import LLMGateway
from claimcheck.pipeline.classify import ClassificationResult
from claimcheck.pipeline.types import OcrResult
from claimcheck.rag.chunking import chunk_clauses
from claimcheck.rag.embedder import Embedder
from claimcheck.worker.storage import Storage
from evals.dataset.render import DATE_STYLES, format_date, format_money
from evals.dataset.types import GroundTruth, PolicyIndex
from tests.fakes import FakeCall, FakeGateway, HashingEmbedder


def ground_truth(data_dir: Path, packet_id: str) -> GroundTruth:
    path = data_dir / "packets" / packet_id / "ground_truth.json"
    return GroundTruth.model_validate_json(path.read_text(encoding="utf-8"))


def add_policy(
    session: Session,
    *,
    versions: tuple[tuple[date, date | None], ...] = ((date(2024, 1, 1), None),),
    limits: dict[str, Decimal] | None = None,
) -> Policy:
    """A policy whose every version carries one limited clause per entry in `limits`."""
    policy = Policy(policy_number=f"AUT-{uuid.uuid4().hex[:6]}", holder_id="cust-1")
    session.add(policy)
    session.flush()
    embedder = HashingEmbedder()
    for no, (start, end) in enumerate(versions, start=1):
        version = PolicyVersion(
            policy_id=policy.id, version_no=no, effective_from=start, effective_to=end
        )
        session.add(version)
        session.flush()
        for index, (coverage_type, limit) in enumerate(sorted((limits or {}).items()), start=1):
            body = f"We pay up to a limit of ${limit:,.2f} for {coverage_type} damage."
            session.add(
                PolicyClause(
                    policy_version_id=version.id,
                    clause_ref=f"4.{index}",
                    title=f"{coverage_type.title()} damage",
                    coverage_type=coverage_type,
                    limit_amount=limit,
                    text=body,
                    embedding=embedder.embed([body])[0],
                )
            )
    session.flush()
    return policy


def add_dataset_policy(
    session: Session, data_dir: Path, policy_number: str, embedder: Embedder
) -> Policy:
    """One policy from `policies/index.json`, with every version and its chunked clauses."""
    index = PolicyIndex.model_validate_json(
        (data_dir / "policies" / "index.json").read_text(encoding="utf-8")
    )
    entry = next(p for p in index.policies if p.policy_number == policy_number)
    policy = Policy(policy_number=entry.policy_number, holder_id=entry.holder_id)
    session.add(policy)
    session.flush()
    for v in entry.versions:
        version = PolicyVersion(
            policy_id=policy.id,
            version_no=v.version_no,
            effective_from=v.effective_from,
            effective_to=v.effective_to,
            document_url=v.file,
        )
        session.add(version)
        session.flush()
        clauses = chunk_clauses((data_dir / "policies" / v.file).read_text(encoding="utf-8"))
        vectors = embedder.embed([c.text for c in clauses])
        for clause, vector in zip(clauses, vectors, strict=True):
            session.add(
                PolicyClause(
                    policy_version_id=version.id,
                    clause_ref=clause.clause_ref,
                    title=clause.title,
                    coverage_type=clause.coverage_type,
                    limit_amount=clause.limit_amount,
                    text=clause.text,
                    embedding=vector,
                )
            )
    session.flush()
    return policy


def add_claim(
    session: Session,
    policy: Policy,
    *,
    packet_id: str | None = None,
    claim_type: str = "collision",
    incident_date: date = date(2024, 6, 1),
    status: ClaimStatus = ClaimStatus.RECEIVED,
) -> Claim:
    claim = Claim(
        packet_id=packet_id or f"PKT-{uuid.uuid4().hex[:8]}",
        policy_id=policy.id,
        claim_type=claim_type,
        incident_date=incident_date,
        status=status.value,
        created_by=policy.holder_id,
    )
    session.add(claim)
    session.flush()
    return claim


def add_document(
    session: Session, claim: Claim, storage: Storage, filename: str, data: bytes
) -> ClaimDocument:
    document = ClaimDocument(
        claim_id=claim.id, filename=filename, storage_key=storage.put(claim.id, filename, data)
    )
    session.add(document)
    session.flush()
    return document


def add_packet(
    session: Session,
    data_dir: Path,
    packet_id: str,
    policy: Policy,
    storage: Storage,
    *,
    skip: frozenset[str] = frozenset(),
) -> Claim:
    """The claim and its uploaded documents, as the upload endpoint would leave them."""
    truth = ground_truth(data_dir, packet_id)
    claim = add_claim(
        session,
        policy,
        packet_id=packet_id,
        claim_type=truth.claim_type.value,
        incident_date=truth.incident_date,
    )
    folder = data_dir / "packets" / packet_id
    for doc in truth.documents:
        if doc.filename not in skip:
            add_document(
                session, claim, storage, doc.filename, (folder / doc.filename).read_bytes()
            )
    return claim


def _packet_and_file(cache_key: str | None) -> tuple[str, str]:
    if cache_key is None:
        raise AssertionError("pipeline calls must carry a cache key")
    _, packet_id, filename = cache_key.split(":")[:3]
    return packet_id, filename


def forced_classifier(
    data_dir: Path,
) -> Callable[..., ClassificationResult]:
    """Classify by ground truth, so a test does not depend on the embedding model."""

    def classify(
        ocr: OcrResult,
        embedder: Embedder,
        llm: LLMGateway | None,
        *,
        cache_key: str | None = None,
    ) -> ClassificationResult:
        packet_id, filename = _packet_and_file(cache_key)
        truth = ground_truth(data_dir, packet_id)
        doc = next(d for d in truth.documents if d.filename == filename)
        return ClassificationResult(doc.doc_type, 0.99, "embedding")

    return classify


def gold_gateway(data_dir: Path, *, is_cloud: bool = False) -> FakeGateway:
    """An extraction model that reads every field right and quotes it as printed."""

    def answer(call: FakeCall) -> BaseModel:
        assert call.schema is not None and call.prompt is not None
        packet_id, filename = _packet_and_file(call.cache_key)
        truth = ground_truth(data_dir, packet_id)
        doc = next(d for d in truth.documents if d.filename == filename)
        fields = {
            name: _gold_field(name, doc.fields.get(name), call.prompt)
            for name in call.schema.model_fields
        }
        return call.schema.model_validate(fields)

    return FakeGateway(structured_handler=answer, is_cloud=is_cloud)


def _gold_field(name: str, gold: str | None, prompt: str) -> dict[str, Any]:
    if gold is None:
        return {"value": None, "confidence": 0.95, "source_text": None}
    if name == "line_items":
        items = json.loads(gold)
        rows = [
            {"description": i["description"], "amount": format_money(Decimal(i["amount"]))}
            for i in items
        ]
        return {"value": rows, "confidence": 0.95, "source_text": items[0]["description"]}
    printed = _as_printed(name, gold, prompt)
    return {"value": printed, "confidence": 0.95, "source_text": printed}


def _as_printed(name: str, gold: str, prompt: str) -> str:
    """The gold value in the form the document prints it, when that form is in the prompt."""
    candidates: list[str] = []
    if name.endswith("date") or name == "date_of_birth":
        day = date.fromisoformat(gold)
        candidates = [format_date(day, style) for style in DATE_STYLES]
    elif name == "total":
        candidates = [format_money(Decimal(gold))]
    for candidate in candidates:
        if candidate in prompt:
            return candidate
    return gold
