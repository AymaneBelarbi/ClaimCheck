"""A job queue in Postgres: one job per claim, leased with SKIP LOCKED.

A job is `queued`, `running`, `done` or `dead`. Each lease is one attempt; a failed attempt
waits 30 s x 2^(attempt-1) before the next, and the third failure is final (`dead`) until an
operator requeues it. Every function takes the current time from its caller, which keeps
retry timing testable and makes the worker's clock the only clock.
"""

import logging
import uuid
from datetime import datetime, timedelta

from sqlalchemy import and_, or_, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from claimcheck.core.types import ClaimStatus
from claimcheck.db.models import Claim, Job
from claimcheck.worker.audit import record
from claimcheck.worker.state_machine import transition

logger = logging.getLogger(__name__)

MAX_ATTEMPTS = 3
BASE_DELAY = timedelta(seconds=30)
# A running job not touched for this long belongs to a worker that died mid-attempt.
LEASE_TIMEOUT = timedelta(minutes=15)
# Claim states the pipeline has finished with; the pipeline itself only ever reaches the first.
_PROCESSED = frozenset(
    {ClaimStatus.NEEDS_REVIEW, ClaimStatus.APPROVED_FOR_REVIEW, ClaimStatus.REJECTED_INCOMPLETE}
)


def backoff(attempt: int) -> timedelta:
    """Delay after failed attempt number `attempt` (1-based)."""
    return BASE_DELAY * int(2 ** (attempt - 1))


def enqueue(session: Session, claim_id: uuid.UUID) -> Job:
    """The claim's job, created if it does not exist yet; enqueueing twice is harmless."""
    session.execute(
        insert(Job).values(claim_id=claim_id).on_conflict_do_nothing(index_elements=["claim_id"])
    )
    return session.scalars(select(Job).where(Job.claim_id == claim_id)).one()


def lease_next(session: Session, now: datetime, *, actor: str = "worker") -> Job | None:
    """Claim the next due job for this worker, or None if nothing is due.

    Due means queued with run_after <= now, or running but abandoned (see LEASE_TIMEOUT). The
    row stays locked until the caller's transaction ends and SKIP LOCKED lets concurrent
    workers pass over it.

    An abandoned job is settled here instead of being run again when the outcome is already
    known: if its claim is past `processing`, the worker died after storing the result but
    before marking the job, so the job is `done`; otherwise, if it was on its last attempt, it
    is not run a fourth time and goes `dead`. Either way the search continues.
    """
    due = or_(
        and_(Job.state == "queued", Job.run_after <= now),
        and_(Job.state == "running", Job.updated_at < now - LEASE_TIMEOUT),
    )
    while True:
        job = session.scalars(
            select(Job)
            .where(due)
            .order_by(Job.run_after)
            .limit(1)
            .with_for_update(skip_locked=True)
        ).first()
        if job is None:
            return None
        if job.state == "running":
            logger.warning(
                "job %s claim %s: attempt %d was abandoned", job.id, job.claim_id, job.attempts
            )
            claim = session.get(Claim, job.claim_id)
            if claim is not None and ClaimStatus(claim.status) in _PROCESSED:
                logger.info("job %s claim %s: result already stored", job.id, job.claim_id)
                complete(session, job)
                continue
            if job.attempts >= MAX_ATTEMPTS:
                fail(
                    session,
                    job,
                    f"worker lost its lease during attempt {job.attempts}",
                    now,
                    actor=actor,
                )
                continue
        job.state = "running"
        job.attempts += 1
        job.updated_at = now
        session.flush()
        return job


def complete(session: Session, job: Job) -> None:
    job.state = "done"
    job.last_error = None
    session.flush()


def fail(session: Session, job: Job, error: str, now: datetime, *, actor: str = "worker") -> None:
    """Record a failed attempt: retry later, or give up after MAX_ATTEMPTS.

    Either way a claim still in `processing` moves to `failed`, so its status never claims
    work is under way when it is not; the next lease takes it back to `processing`. On the last
    attempt a claim that never left `received` (the attempt failed before the pipeline could
    start it) is failed too, through `processing`, so no dead job leaves its claim waiting.
    """
    claim = session.get(Claim, job.claim_id)
    reason = f"attempt {job.attempts} failed"
    final = job.attempts >= MAX_ATTEMPTS
    if claim is not None and final and claim.status == ClaimStatus.RECEIVED.value:
        transition(session, claim, ClaimStatus.PROCESSING, actor=actor, reason=reason)
    if claim is not None and claim.status == ClaimStatus.PROCESSING.value:
        transition(session, claim, ClaimStatus.FAILED, actor=actor, reason=reason)
    job.last_error = error
    job.updated_at = now
    payload: dict[str, object] = {
        "claim_id": str(job.claim_id),
        "attempt": job.attempts,
        "error": error,
    }
    if final:
        job.state = "dead"
        action = "job.dead"
    else:
        job.state = "queued"
        job.run_after = now + backoff(job.attempts)
        payload["retry_at"] = job.run_after.isoformat()
        action = "job.failed"
    record(
        session, actor=actor, action=action, entity="job", entity_id=str(job.id), payload=payload
    )
    session.flush()
