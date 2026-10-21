"""Put a failed claim's dead job back on the queue.

    python -m worker.reprocess <claim_id>

The job starts again from attempt 0. Only a claim in `failed` qualifies: anything else is
either still being retried or has been processed, and the state machine has no way back
from `needs_review`.
"""

import argparse
import uuid
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from core.types import ClaimStatus
from db.models import Claim, Job
from worker.audit import record
from worker.main import worker_sessions

REQUEUEABLE = frozenset({"dead", "done"})


class ReprocessError(Exception):
    pass


def requeue(session: Session, claim_id: uuid.UUID, now: datetime, *, actor: str) -> Job:
    claim = session.get(Claim, claim_id, with_for_update=True)
    if claim is None:
        raise ReprocessError(f"no claim {claim_id}")
    job = session.scalars(select(Job).where(Job.claim_id == claim_id).with_for_update()).first()
    if job is None:
        raise ReprocessError(f"claim {claim_id} has no job; it was never queued")
    if claim.status != ClaimStatus.FAILED.value:
        raise ReprocessError(
            f"claim {claim_id} is {claim.status}; only a failed claim can be reprocessed"
        )
    if job.state not in REQUEUEABLE:
        raise ReprocessError(
            f"job {job.id} is {job.state} (attempt {job.attempts}); it is still being retried"
        )
    record(
        session,
        actor=actor,
        action="job.requeued",
        entity="job",
        entity_id=str(job.id),
        payload={
            "claim_id": str(claim_id),
            "previous_state": job.state,
            "previous_attempts": job.attempts,
            "last_error": job.last_error,
        },
    )
    job.state = "queued"
    job.attempts = 0
    job.run_after = now
    job.updated_at = now
    session.flush()
    return job


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Requeue the dead job of a failed claim.")
    parser.add_argument("claim_id", type=uuid.UUID)
    parser.add_argument("--actor", default="operator", help="name recorded in the audit log")
    args = parser.parse_args(argv)
    sessions = worker_sessions()
    try:
        with sessions() as session:
            job = requeue(session, args.claim_id, datetime.now(UTC), actor=args.actor)
            job_id = job.id
    except ReprocessError as exc:
        raise SystemExit(f"cannot reprocess: {exc}") from exc
    print(f"job {job_id} for claim {args.claim_id} is queued again")


if __name__ == "__main__":
    main()
