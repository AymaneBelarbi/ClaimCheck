"""The worker process: lease a job, process its claim, record the outcome, repeat.

    python -m worker.main

SIGTERM or SIGINT lets the job in hand finish and then stops the loop, so stopping the
worker never abandons a claim halfway through.
"""

import logging
import signal
import threading
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from types import FrameType

from sqlalchemy import Engine
from sqlalchemy.orm import Session

from core.settings import Settings, get_settings
from core.tracing import setup_tracing
from db.models import Job
from db.session import SYSTEM, SessionFactory, engine_for, session_factory
from gateway.factory import get_gateway
from rag.embedder import FastEmbedEmbedder
from worker.pipeline_runner import PipelineDeps, process_claim
from worker.queue import complete, fail, lease_next
from worker.storage import Storage

logger = logging.getLogger("worker")

POLL_INTERVAL_SECONDS = 1.0


def utc_now() -> datetime:
    return datetime.now(UTC)


# The user id the worker's transactions carry. Row-level security keys on the `system` role;
# this only names who is asking.
WORKER_USER = "worker"


def worker_engine(settings: Settings | None = None) -> Engine:
    """The connection the worker uses for everything it reads and writes: the
    claimcheck_worker role, which may update claims and replace their derived rows but
    cannot create claims or change policies."""
    return engine_for((settings or get_settings()).worker_database_url)


def worker_sessions(settings: Settings | None = None) -> SessionFactory:
    """Transactions as the worker: every claim is visible to the `system` role."""
    return session_factory(worker_engine(settings), SYSTEM, WORKER_USER)


def run_once(
    sessions: SessionFactory, deps: PipelineDeps, clock: Callable[[], datetime] = utc_now
) -> bool:
    """Process one due job, if there is one. Returns whether a job was taken."""
    with sessions() as session:
        job = lease_next(session, clock(), actor=deps.actor)
        if job is None:
            return False
        job_id, claim_id, attempt = job.id, job.claim_id, job.attempts

    logger.info("job %s claim %s attempt %d: started", job_id, claim_id, attempt)
    try:
        process_claim(sessions, claim_id, deps)
    except Exception as exc:
        logger.exception("job %s claim %s attempt %d: failed", job_id, claim_id, attempt)
        with sessions() as session:
            fail(session, _job(session, job_id), repr(exc), clock(), actor=deps.actor)
        return True

    with sessions() as session:
        complete(session, _job(session, job_id))
    logger.info("job %s claim %s attempt %d: done", job_id, claim_id, attempt)
    return True


def serve(
    sessions: SessionFactory,
    deps: PipelineDeps,
    stop: threading.Event,
    poll_interval: float = POLL_INTERVAL_SECONDS,
    clock: Callable[[], datetime] = utc_now,
) -> None:
    """Run jobs until `stop` is set; the job in hand when it is set still finishes."""
    while not stop.is_set():
        try:
            worked = run_once(sessions, deps, clock)
        except Exception:
            # The database is unreachable or similar; the job, if any, is retried by lease.
            logger.exception("worker loop error")
            worked = False
        if not worked:
            stop.wait(poll_interval)
    logger.info("worker stopped")


def _job(session: Session, job_id: uuid.UUID) -> Job:
    job = session.get(Job, job_id, with_for_update=True)
    if job is None:
        raise LookupError(f"job {job_id} disappeared")
    return job


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    stop = threading.Event()

    def request_stop(signum: int, frame: FrameType | None) -> None:
        logger.info("%s received; finishing the current job", signal.Signals(signum).name)
        stop.set()

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    settings = get_settings()
    setup_tracing("claimcheck-worker", settings.otel_endpoint)
    deps = PipelineDeps(
        storage=Storage(Path(settings.storage_dir)),
        embedder=FastEmbedEmbedder(),
        llm=get_gateway(settings),
    )
    logger.info("worker started (LLM backend %s)", settings.llm_backend)
    serve(worker_sessions(), deps, stop)


if __name__ == "__main__":
    main()
