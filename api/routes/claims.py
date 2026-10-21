"""Claim intake and review endpoints.

Authorization is decided here, in code: a customer sees and creates only claims on policies
they hold, and asking for anyone else's claim is a 404 so its existence is not revealed.
Row-level security enforces the same rule again underneath (migration 0002): every request's
session carries the caller's role and subject, so a missed check here still returns no rows.
"""

import hashlib
import re
import uuid
from datetime import date
from typing import Annotated

import anyio
from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    Header,
    HTTPException,
    Query,
    Response,
    UploadFile,
)
from mcp import MCPError
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from agent.loop import AgentAnswer, answer_question
from api.auth import Principal, current_principal, require_role
from api.deps import get_embedder, get_llm, get_session, get_session_factory, get_storage
from api.limits import MAX_FILE_BYTES, MAX_FILES
from api.schemas import (
    AskIn,
    AskOut,
    ClaimCreated,
    ClaimDetail,
    ClaimSummary,
    Decision,
    DecisionIn,
    DecisionOut,
    DocumentOut,
)
from core.review import ReviewFile
from core.tracing import span as start_span
from core.types import ClaimStatus, ClaimType, DocType, Role
from db.models import AuditLog, Claim, ClaimDocument, Policy
from db.session import SessionFactory
from gateway.base import CassetteMissError, GatewayError, LLMGateway
from gateway.redaction import scope_gateway
from mcp_server.server import build_server, claim_ocr_texts, connect
from rag.embedder import Embedder
from worker.audit import record
from worker.pipeline_runner import ClaimNotFound
from worker.queue import enqueue
from worker.review_file import build_review_file
from worker.state_machine import InvalidTransition, transition
from worker.storage import Storage

router = APIRouter(prefix="/claims", tags=["claims"])

LIST_LIMIT = 100
# The whole agent run behind /ask: up to six model turns plus their tool calls.
ASK_DEADLINE_SECONDS = 120.0
# The unique index on claims.packet_id, which holds the Idempotency-Key.
PACKET_KEY_CONSTRAINT = "claims_packet_id_key"
_KEY = re.compile(r"[\x20-\x7e]{1,128}")
_READ_CHUNK = 1024 * 1024

CurrentPrincipal = Annotated[Principal, Depends(current_principal)]
AdjusterOnly = Annotated[Principal, Depends(require_role(Role.ADJUSTER))]
DbSession = Annotated[Session, Depends(get_session)]


def _not_found(what: str) -> HTTPException:
    return HTTPException(status_code=404, detail=f"{what} not found")


def find_by_key(session: Session, key: str) -> Claim | None:
    return session.scalar(select(Claim).where(Claim.packet_id == key))


def _idempotency_key(raw: str | None) -> str:
    if raw is None:
        raise HTTPException(status_code=400, detail="Idempotency-Key header is required")
    if not _KEY.fullmatch(raw) or not raw.strip():
        raise HTTPException(
            status_code=400, detail="Idempotency-Key must be 1 to 128 printable ASCII characters"
        )
    return raw


def _plain_name(raw: str | None) -> str:
    """The last path component of the client's filename, without control characters."""
    name = (raw or "").replace("\\", "/").rsplit("/", 1)[-1]
    name = "".join(ch for ch in name if ch.isprintable()).strip()
    return name[:255] or "upload"


def _read_capped(upload: UploadFile) -> bytes:
    """Read the file in chunks, giving up as soon as it passes the size limit."""
    chunks: list[bytes] = []
    total = 0
    while chunk := upload.file.read(_READ_CHUNK):
        total += len(chunk)
        if total > MAX_FILE_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"{upload.filename or 'file'} is larger than {MAX_FILE_BYTES // 2**20} MB",
            )
        chunks.append(chunk)
    return b"".join(chunks)


def _key_conflict() -> HTTPException:
    return HTTPException(
        status_code=409, detail="Idempotency-Key was already used for a different request"
    )


def _is_key_violation(exc: IntegrityError) -> bool:
    diag = getattr(exc.orig, "diag", None)
    return getattr(diag, "constraint_name", None) == PACKET_KEY_CONSTRAINT


def _replay(
    response: Response, session: Session, existing: Claim, policy_id: uuid.UUID, who: Principal
) -> ClaimCreated:
    """Answer a repeated Idempotency-Key, or refuse it if it was not this caller's request."""
    if existing.policy_id != policy_id:
        raise _key_conflict()
    if who.role is not Role.ADJUSTER:
        creator = session.scalar(
            select(AuditLog.actor).where(
                AuditLog.action == "claim.created", AuditLog.entity_id == str(existing.id)
            )
        )
        if creator != who.sub:
            raise _key_conflict()
    response.status_code = 200
    return ClaimCreated(claim_id=existing.id, status=ClaimStatus(existing.status))


@router.post("", status_code=202)
def create_claim(
    response: Response,
    principal: CurrentPrincipal,
    session: DbSession,
    storage: Annotated[Storage, Depends(get_storage)],
    policy_number: Annotated[str, Form()],
    claim_type: Annotated[ClaimType, Form()],
    incident_date: Annotated[date, Form()],
    files: Annotated[list[UploadFile] | None, File()] = None,
    # Some clients (PHP-style forms) name a repeated field with brackets.
    bracket_files: Annotated[list[UploadFile] | None, File(alias="files[]")] = None,
    idempotency_key: Annotated[str | None, Header()] = None,
) -> ClaimCreated:
    """Create a claim from an uploaded packet and queue it for processing (202).

    The Idempotency-Key becomes the claim's packet_id. Repeating a key answers 200 with the
    existing claim and creates nothing; the status in that reply is the claim's current one,
    which may be later than `received`. The same key on a different policy, or from a customer
    who did not create the claim, is a 409.
    """
    key = _idempotency_key(idempotency_key)
    uploads = [*(files or []), *(bracket_files or [])]
    if not uploads:
        raise HTTPException(status_code=422, detail="at least one file is required")
    if len(uploads) > MAX_FILES:
        raise HTTPException(status_code=413, detail=f"at most {MAX_FILES} files per claim")
    if incident_date > date.today():
        raise HTTPException(status_code=422, detail="incident_date is in the future")
    names = [_plain_name(u.filename) for u in uploads]
    if len(set(names)) != len(names):
        raise HTTPException(status_code=422, detail="file names must be unique within a claim")

    policy = session.scalar(select(Policy).where(Policy.policy_number == policy_number))
    if policy is None or (principal.role is Role.CUSTOMER and policy.holder_id != principal.sub):
        raise _not_found("policy")
    policy_id, holder_id = policy.id, policy.holder_id

    existing = find_by_key(session, key)
    if existing is not None:
        return _replay(response, session, existing, policy_id, principal)

    claim_id = uuid.uuid4()
    try:
        # Files go to disk first; if the transaction fails they are removed again.
        storage_keys = [
            storage.put(claim_id, name, _read_capped(upload))
            for upload, name in zip(uploads, names, strict=True)
        ]
        session.add(
            Claim(
                id=claim_id,
                packet_id=key,
                policy_id=policy_id,
                claim_type=claim_type.value,
                incident_date=incident_date,
                # The holder, not the uploader, so a claim an adjuster files is the customer's too.
                created_by=holder_id,
            )
        )
        session.flush()
        session.add_all(
            ClaimDocument(claim_id=claim_id, filename=name, storage_key=storage_key)
            for name, storage_key in zip(names, storage_keys, strict=True)
        )
        enqueue(session, claim_id)
        record(
            session,
            actor=principal.sub,
            action="claim.created",
            entity="claim",
            entity_id=str(claim_id),
            payload={"packet_id": key, "policy_number": policy_number, "documents": len(names)},
        )
        session.commit()
    except IntegrityError as exc:
        # A concurrent request with the same key got there first, or the key belongs to a
        # claim this caller cannot see: row-level security hides other customers' claims from
        # the lookup above, but not from the unique index.
        session.rollback()
        storage.discard(claim_id)
        existing = find_by_key(session, key)
        if existing is not None:
            return _replay(response, session, existing, policy_id, principal)
        if _is_key_violation(exc):
            raise _key_conflict() from None
        raise
    except BaseException:
        session.rollback()
        storage.discard(claim_id)
        raise
    return ClaimCreated(claim_id=claim_id, status=ClaimStatus.RECEIVED)


@router.get("")
def list_claims(
    principal: CurrentPrincipal,
    session: DbSession,
    status_filter: Annotated[ClaimStatus | None, Query(alias="status")] = None,
) -> list[ClaimSummary]:
    stmt = select(Claim).order_by(Claim.created_at.desc(), Claim.id).limit(LIST_LIMIT)
    if principal.role is Role.CUSTOMER:
        stmt = stmt.join(Policy, Policy.id == Claim.policy_id).where(
            Policy.holder_id == principal.sub
        )
    if status_filter is not None:
        stmt = stmt.where(Claim.status == status_filter.value)
    return [_summary(c) for c in session.scalars(stmt)]


@router.get("/{claim_id}")
def get_claim(claim_id: uuid.UUID, principal: CurrentPrincipal, session: DbSession) -> ClaimDetail:
    row = session.execute(
        select(Claim, Policy).join(Policy, Policy.id == Claim.policy_id).where(Claim.id == claim_id)
    ).one_or_none()
    if row is None or (principal.role is Role.CUSTOMER and row.Policy.holder_id != principal.sub):
        raise _not_found("claim")
    claim, policy = row.Claim, row.Policy
    documents = session.scalars(
        select(ClaimDocument)
        .where(ClaimDocument.claim_id == claim_id)
        .order_by(ClaimDocument.filename, ClaimDocument.id)
    )
    return ClaimDetail(
        **_summary(claim).model_dump(),
        policy_number=policy.policy_number,
        documents=[
            DocumentOut(
                id=d.id,
                filename=d.filename,
                doc_type=DocType(d.doc_type) if d.doc_type is not None else None,
                classification_confidence=d.classification_confidence,
                low_confidence=d.low_confidence,
                error=d.error,
            )
            for d in documents
        ],
    )


@router.get("/{claim_id}/review")
def get_review_file(claim_id: uuid.UUID, _: AdjusterOnly, session: DbSession) -> ReviewFile:
    try:
        return build_review_file(session, claim_id)
    except ClaimNotFound:
        raise _not_found("claim") from None


@router.post("/{claim_id}/decision")
def decide(
    claim_id: uuid.UUID, body: DecisionIn, principal: AdjusterOnly, session: DbSession
) -> DecisionOut:
    claim = session.get(Claim, claim_id)
    if claim is None:
        raise _not_found("claim")
    target = (
        ClaimStatus.APPROVED_FOR_REVIEW
        if body.decision is Decision.APPROVED_FOR_REVIEW
        else (ClaimStatus.REJECTED_INCOMPLETE)
    )
    try:
        transition(session, claim, target, actor=principal.sub, reason=body.reason)
    except InvalidTransition as exc:
        session.rollback()
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    session.commit()
    return DecisionOut(claim_id=claim_id, status=target)


@router.post("/{claim_id}/ask")
def ask(
    claim_id: uuid.UUID,
    body: AskIn,
    principal: AdjusterOnly,
    session: DbSession,
    sessions: Annotated[SessionFactory, Depends(get_session_factory)],
    gateway: Annotated[LLMGateway, Depends(get_llm)],
    embedder: Annotated[Embedder, Depends(get_embedder)],
) -> AskOut:
    """Answer an adjuster's question about the claim with the tool-calling agent.

    The question is audited before the agent starts, and every tool call the agent makes is
    audited by the tool server. The reply carries the answer, the clause refs it rests on and
    the names of the tools used; tool arguments and results stay in the audit log.
    """
    claim = session.get(Claim, claim_id)
    if claim is None:
        raise _not_found("claim")
    # The packet's text tells a cloud backend's redactor which names to remove from the
    # tool results the agent sends back to the model.
    texts = claim_ocr_texts(session, claim_id)
    record(
        session,
        actor=principal.sub,
        action="agent.question",
        entity="claim",
        entity_id=str(claim_id),
        payload={"question": body.question},
    )
    session.commit()

    server = build_server(sessions, embedder, principal.sub)
    scoped = scope_gateway(gateway, texts)
    digest = hashlib.sha256(body.question.encode("utf-8")).hexdigest()[:12]
    # The packet id (the upload's Idempotency-Key), not the claim id, so a recorded
    # conversation replays against a database where the same packet was filed again.
    cache_key = f"agent:{claim.packet_id}:{digest}"

    async def run() -> AgentAnswer:
        async with connect(server) as mcp:
            return await answer_question(
                claim_id=claim_id,
                question=body.question,
                gateway=scoped,
                mcp=mcp,
                cache_key=cache_key,
            )

    async def run_with_deadline() -> AgentAnswer:
        with anyio.fail_after(ASK_DEADLINE_SECONDS):
            return await run()

    try:
        # A sync endpoint runs in a worker thread, which has no event loop of its own. The
        # span stays open across it: the event loop copies the context, so each model call
        # the agent makes is a child of this span.
        with start_span("agent.ask", **{"claim.id": str(claim_id)}):
            answer = anyio.run(run_with_deadline)
    except GatewayError as exc:
        raise HTTPException(status_code=502, detail="the model backend did not answer") from exc
    except CassetteMissError as exc:
        # The default compose stack answers from recorded responses, which cover the demo
        # question only.
        raise HTTPException(
            status_code=503,
            detail="no recorded answer for this question under LLM_BACKEND=replay",
        ) from exc
    except MCPError as exc:
        raise HTTPException(status_code=502, detail="the tool server failed") from exc
    except TimeoutError as exc:
        raise HTTPException(
            status_code=504,
            detail=f"no answer within {ASK_DEADLINE_SECONDS:g} s",
        ) from exc
    # The tool server audits every call it receives; calls the model proposed after its budget
    # ran out never reach it, so they are recorded here.
    for call in answer.unexecuted_calls:
        record(
            session,
            actor=principal.sub,
            action="agent.tool_call",
            entity="claim",
            entity_id=str(claim_id),
            payload={"tool": call.name, "arguments": call.arguments, "outcome": "rejected: budget"},
        )
    session.commit()
    return AskOut(
        answer=answer.answer,
        citations=answer.citations,
        uncited=answer.uncited,
        tools_called=[call.name for call in answer.tool_calls],
    )


def _summary(claim: Claim) -> ClaimSummary:
    return ClaimSummary(
        claim_id=claim.id,
        packet_id=claim.packet_id,
        status=ClaimStatus(claim.status),
        claim_type=ClaimType(claim.claim_type),
        incident_date=claim.incident_date,
        created_at=claim.created_at,
    )
