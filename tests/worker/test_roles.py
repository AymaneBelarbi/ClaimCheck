"""The worker process and the reprocess command log in as claimcheck_worker, as `system`."""

import pytest
from sqlalchemy import select, text
from sqlalchemy.engine import URL

from core.settings import get_settings
from core.types import ClaimStatus
from db.models import Job
from db.session import SessionFactory
from tests.db.conftest import as_role
from tests.worker.factories import add_claim, add_policy
from worker import reprocess
from worker.main import worker_engine, worker_sessions
from worker.queue import MAX_ATTEMPTS, enqueue
from worker.state_machine import transition

PROBE = text(
    "SELECT current_user, current_setting('app.role', true), current_setting('app.user_id', true)"
)


def test_worker_connects_as_the_worker_role() -> None:
    assert worker_engine().url.username == "claimcheck_worker"


@pytest.fixture
def worker_env(worker_db_url: URL, monkeypatch: pytest.MonkeyPatch) -> None:
    url = as_role(worker_db_url, "claimcheck_worker").render_as_string(hide_password=False)
    monkeypatch.setenv("WORKER_DATABASE_URL", url)
    get_settings.cache_clear()


@pytest.mark.db
@pytest.mark.usefixtures("worker_env")
def test_worker_sessions_carry_the_system_identity(sessions: SessionFactory) -> None:
    transaction = worker_sessions()
    for _ in range(2):  # every transaction, not only the first on a connection
        with transaction() as session:
            assert tuple(session.execute(PROBE).one()) == ("claimcheck_worker", "system", "worker")


@pytest.mark.db
@pytest.mark.usefixtures("worker_env")
def test_reprocess_command_requeues_as_the_worker(
    sessions: SessionFactory, capsys: pytest.CaptureFixture[str]
) -> None:
    with sessions() as session:
        claim = add_claim(session, add_policy(session))
        job = enqueue(session, claim.id)
        job.state, job.attempts = "dead", MAX_ATTEMPTS
        transition(session, claim, ClaimStatus.PROCESSING, actor="worker")
        transition(session, claim, ClaimStatus.FAILED, actor="worker")
        claim_id, job_id = claim.id, job.id

    reprocess.main([str(claim_id), "--actor", "ops:alice"])

    assert f"job {job_id}" in capsys.readouterr().out
    with sessions() as session:
        assert session.scalars(select(Job.state).where(Job.id == job_id)).one() == "queued"
