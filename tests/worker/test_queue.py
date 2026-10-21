import uuid
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from core.types import ClaimStatus
from db.models import AuditLog, Claim, Job
from db.session import SessionFactory
from tests.worker.factories import add_claim, add_policy
from worker.queue import (
    LEASE_TIMEOUT,
    MAX_ATTEMPTS,
    backoff,
    complete,
    enqueue,
    fail,
    lease_next,
)
from worker.reprocess import ReprocessError, requeue
from worker.state_machine import transition

pytestmark = pytest.mark.db


@pytest.fixture
def now() -> datetime:
    """The worker's clock for one test: comfortably after the database's now(), which stamps
    run_after on enqueue. Taken per test, so a slow run cannot leave it behind the database."""
    return datetime.now(UTC) + timedelta(minutes=1)


def _claim_id(session: Session) -> uuid.UUID:
    return add_claim(session, add_policy(session)).id


def _actions(session: Session, entity_id: str) -> list[str]:
    stmt = select(AuditLog.action).where(AuditLog.entity_id == entity_id).order_by(AuditLog.id)
    return list(session.scalars(stmt))


def _start(session: Session, claim_id: uuid.UUID) -> None:
    """What the pipeline does first: the claim moves to processing."""
    claim = session.get(Claim, claim_id)
    assert claim is not None
    transition(session, claim, ClaimStatus.PROCESSING, actor="worker")


def test_backoff_doubles_from_thirty_seconds() -> None:
    assert MAX_ATTEMPTS == 3
    assert [backoff(n) for n in (1, 2, 3)] == [
        timedelta(seconds=30),
        timedelta(seconds=60),
        timedelta(seconds=120),
    ]


def test_enqueue_is_idempotent_per_claim(
    sessions: SessionFactory, adjuster_sessions: SessionFactory
) -> None:
    with sessions() as session:
        claim_id = _claim_id(session)
    # The API enqueues, as the caller.
    with adjuster_sessions() as session:
        first = enqueue(session, claim_id)
    with adjuster_sessions() as session:
        second = enqueue(session, claim_id)
        assert second.id == first.id
        assert session.scalars(select(Job)).all() == [second]
        assert second.state == "queued" and second.attempts == 0


def test_lease_marks_running_and_counts_the_attempt(
    sessions: SessionFactory, worker_sessions: SessionFactory, now: datetime
) -> None:
    with sessions() as session:
        job_id = enqueue(session, _claim_id(session)).id
    with worker_sessions() as session:
        job = lease_next(session, now)
        assert job is not None and job.id == job_id
        assert (job.state, job.attempts, job.updated_at) == ("running", 1, now)
    with worker_sessions() as session:
        assert lease_next(session, now) is None


def test_lease_waits_for_run_after(
    sessions: SessionFactory, worker_sessions: SessionFactory, now: datetime
) -> None:
    with sessions() as session:
        job = enqueue(session, _claim_id(session))
        job.run_after = now + timedelta(seconds=10)
    with worker_sessions() as session:
        assert lease_next(session, now) is None
    with worker_sessions() as session:
        assert lease_next(session, now + timedelta(seconds=10)) is not None


def test_lease_takes_the_oldest_due_job_first(
    sessions: SessionFactory, worker_sessions: SessionFactory, now: datetime
) -> None:
    with sessions() as session:
        late = enqueue(session, _claim_id(session))
        early = enqueue(session, _claim_id(session))
        late.run_after = now - timedelta(seconds=1)
        early.run_after = now - timedelta(seconds=5)
    with worker_sessions() as session:
        job = lease_next(session, now)
        assert job is not None and job.id == early.id


def test_concurrent_leases_get_different_jobs(
    sessions: SessionFactory, worker_sessions: SessionFactory, now: datetime
) -> None:
    with sessions() as session:
        ids = {enqueue(session, _claim_id(session)).id for _ in range(2)}

    with worker_sessions() as first, worker_sessions() as second:
        a = lease_next(first, now)
        # `first` still holds its row lock; SKIP LOCKED hands `second` the other job.
        b = lease_next(second, now)
        assert a is not None and b is not None
        assert {a.id, b.id} == ids
        with worker_sessions() as third:
            assert lease_next(third, now) is None


def test_running_job_is_recovered_after_the_lease_timeout(
    sessions: SessionFactory, worker_sessions: SessionFactory, now: datetime
) -> None:
    with sessions() as session:
        job_id = enqueue(session, _claim_id(session)).id
    with worker_sessions() as session:
        assert lease_next(session, now) is not None
    with worker_sessions() as session:
        assert lease_next(session, now + LEASE_TIMEOUT) is None
    with worker_sessions() as session:
        later = now + LEASE_TIMEOUT + timedelta(seconds=1)
        job = lease_next(session, later)
        assert job is not None and job.id == job_id
        assert (job.state, job.attempts, job.updated_at) == ("running", 2, later)


def test_lost_final_attempt_is_dead_not_leased_again(
    sessions: SessionFactory, worker_sessions: SessionFactory, now: datetime
) -> None:
    with sessions() as session:
        claim_id = _claim_id(session)
        job = enqueue(session, claim_id)
        job.state, job.attempts, job.updated_at = "running", MAX_ATTEMPTS, now
        _start(session, claim_id)
        job_id = job.id
    with worker_sessions() as session:
        assert lease_next(session, now + LEASE_TIMEOUT + timedelta(seconds=1)) is None
    with sessions() as session:
        dead = session.get(Job, job_id)
        claim = session.get(Claim, claim_id)
        assert dead is not None and dead.state == "dead"
        assert dead.last_error is not None and "lease" in dead.last_error
        assert claim is not None and claim.status == ClaimStatus.FAILED.value
        assert _actions(session, str(job_id)) == ["job.dead"]


@pytest.mark.parametrize("attempts", [1, MAX_ATTEMPTS])
def test_abandoned_job_whose_claim_was_processed_is_done(
    sessions: SessionFactory, worker_sessions: SessionFactory, now: datetime, attempts: int
) -> None:
    """The worker stored the result and died before marking the job done."""
    with sessions() as session:
        claim_id = _claim_id(session)
        job = enqueue(session, claim_id)
        job.state, job.attempts, job.updated_at = "running", attempts, now
        _start(session, claim_id)
        claim = session.get(Claim, claim_id)
        assert claim is not None
        transition(session, claim, ClaimStatus.NEEDS_REVIEW, actor="worker")
        job_id = job.id
    with worker_sessions() as session:
        assert lease_next(session, now + LEASE_TIMEOUT + timedelta(seconds=1)) is None
    with sessions() as session:
        done = session.get(Job, job_id)
        claim = session.get(Claim, claim_id)
        assert done is not None and (done.state, done.attempts) == ("done", attempts)
        assert claim is not None and claim.status == ClaimStatus.NEEDS_REVIEW.value
        assert _actions(session, str(job_id)) == []


def test_fail_requeues_with_backoff_and_fails_the_claim(
    sessions: SessionFactory, worker_sessions: SessionFactory, now: datetime
) -> None:
    with sessions() as session:
        claim_id = _claim_id(session)
        enqueue(session, claim_id)
    with worker_sessions() as session:
        job = lease_next(session, now)
        assert job is not None
        _start(session, claim_id)
    with worker_sessions() as session:
        job = session.get(Job, job.id)
        assert job is not None
        fail(session, job, "RuntimeError('ocr crashed')", now)
    with sessions() as session:
        job = session.get(Job, job.id)
        claim = session.get(Claim, claim_id)
        assert job is not None and claim is not None
        assert job.state == "queued"
        assert job.run_after == now + timedelta(seconds=30)
        assert job.last_error == "RuntimeError('ocr crashed')"
        assert claim.status == ClaimStatus.FAILED.value
        assert _actions(session, str(job.id)) == ["job.failed"]
        failed = session.scalars(select(AuditLog).where(AuditLog.action == "job.failed")).one()
        assert failed.actor == "worker"
        assert failed.payload["attempt"] == 1
        assert failed.payload["error"] == "RuntimeError('ocr crashed')"


def test_three_failures_make_the_job_dead_and_the_claim_failed(
    sessions: SessionFactory, worker_sessions: SessionFactory, now: datetime
) -> None:
    with sessions() as session:
        claim_id = _claim_id(session)
        enqueue(session, claim_id)

    run_afters = []
    for _ in range(MAX_ATTEMPTS):
        with worker_sessions() as session:
            job = lease_next(session, now)
            assert job is not None
            _start(session, claim_id)
            fail(session, job, "boom", now)
            run_afters.append(job.run_after - now)
            now = job.run_after

    with worker_sessions() as session:
        job = session.scalars(select(Job)).one()
        claim = session.get(Claim, claim_id)
        assert claim is not None
        assert (job.state, job.attempts) == ("dead", MAX_ATTEMPTS)
        assert claim.status == ClaimStatus.FAILED.value
        assert run_afters[:2] == [timedelta(seconds=30), timedelta(seconds=60)]
        assert _actions(session, str(job.id)) == ["job.failed", "job.failed", "job.dead"]
        assert lease_next(session, now + timedelta(days=1)) is None


def test_fail_leaves_a_claim_that_is_not_processing_alone(
    sessions: SessionFactory, worker_sessions: SessionFactory, now: datetime
) -> None:
    with sessions() as session:
        claim_id = _claim_id(session)
        enqueue(session, claim_id)
    with worker_sessions() as session:
        job = lease_next(session, now)
        assert job is not None
        fail(session, job, "claim row missing a document", now)
        claim = session.get(Claim, claim_id)
        assert claim is not None and claim.status == ClaimStatus.RECEIVED.value


def test_a_claim_that_never_left_received_fails_with_its_last_attempt(
    sessions: SessionFactory, worker_sessions: SessionFactory, now: datetime
) -> None:
    """Every attempt failed before the pipeline could start the claim."""
    with sessions() as session:
        claim_id = _claim_id(session)
        enqueue(session, claim_id)
    for _ in range(MAX_ATTEMPTS):
        with worker_sessions() as session:
            job = lease_next(session, now)
            assert job is not None
            fail(session, job, "could not read the claim", now)
            now = job.run_after

    with sessions() as session:
        job = session.scalars(select(Job)).one()
        claim = session.get(Claim, claim_id)
        assert claim is not None
        assert job.state == "dead"
        assert claim.status == ClaimStatus.FAILED.value
        moves = session.scalars(
            select(AuditLog)
            .where(AuditLog.action == "claim.transition", AuditLog.entity_id == str(claim_id))
            .order_by(AuditLog.id)
        ).all()
        assert [(m.payload["from"], m.payload["to"]) for m in moves] == [
            ("received", "processing"),
            ("processing", "failed"),
        ]


def test_complete_marks_done(
    sessions: SessionFactory, worker_sessions: SessionFactory, now: datetime
) -> None:
    with sessions() as session:
        enqueue(session, _claim_id(session))
    with worker_sessions() as session:
        job = lease_next(session, now)
        assert job is not None
        complete(session, job)
    with worker_sessions() as session:
        assert session.scalars(select(Job.state)).one() == "done"
        assert lease_next(session, now + timedelta(days=1)) is None


def _dead_job(session: Session) -> tuple[uuid.UUID, uuid.UUID]:
    claim_id = _claim_id(session)
    job = enqueue(session, claim_id)
    job.state, job.attempts, job.last_error = "dead", MAX_ATTEMPTS, "boom"
    claim = session.get(Claim, claim_id)
    assert claim is not None
    transition(session, claim, ClaimStatus.PROCESSING, actor="worker")
    transition(session, claim, ClaimStatus.FAILED, actor="worker")
    return claim_id, job.id


def test_reprocess_requeues_a_dead_job(
    sessions: SessionFactory, worker_sessions: SessionFactory, now: datetime
) -> None:
    with sessions() as session:
        claim_id, job_id = _dead_job(session)
    with worker_sessions() as session:
        assert requeue(session, claim_id, now, actor="ops:alice").id == job_id
    with worker_sessions() as session:
        job = session.get(Job, job_id)
        assert job is not None
        assert (job.state, job.attempts, job.run_after) == ("queued", 0, now)
        requeued = session.scalars(select(AuditLog).where(AuditLog.action == "job.requeued")).one()
        assert requeued.actor == "ops:alice"
        assert requeued.payload == {
            "claim_id": str(claim_id),
            "previous_state": "dead",
            "previous_attempts": MAX_ATTEMPTS,
            "last_error": "boom",
        }
        leased = lease_next(session, now)
        assert leased is not None and leased.attempts == 1


def test_reprocess_refuses_a_claim_that_did_not_fail(
    sessions: SessionFactory, worker_sessions: SessionFactory, now: datetime
) -> None:
    with sessions() as session:
        claim_id = _claim_id(session)
        enqueue(session, claim_id)
    with worker_sessions() as session, pytest.raises(ReprocessError, match="received"):
        requeue(session, claim_id, now, actor="ops")


def test_reprocess_refuses_a_job_that_is_still_retrying(
    sessions: SessionFactory, worker_sessions: SessionFactory, now: datetime
) -> None:
    with sessions() as session:
        claim_id, job_id = _dead_job(session)
        job = session.get(Job, job_id)
        assert job is not None
        job.state = "queued"
    with worker_sessions() as session, pytest.raises(ReprocessError, match="queued"):
        requeue(session, claim_id, now, actor="ops")


def test_reprocess_refuses_an_unknown_claim(worker_sessions: SessionFactory, now: datetime) -> None:
    with worker_sessions() as session, pytest.raises(ReprocessError, match="no claim"):
        requeue(session, uuid.uuid4(), now, actor="ops")
