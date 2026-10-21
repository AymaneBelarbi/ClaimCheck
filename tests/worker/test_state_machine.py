import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from claimcheck.core.types import ClaimStatus
from claimcheck.db.models import AuditLog, Claim
from claimcheck.db.session import SessionFactory
from claimcheck.worker.audit import record
from claimcheck.worker.state_machine import ALLOWED, InvalidTransition, transition
from tests.worker.factories import add_claim, add_policy

pytestmark = pytest.mark.db

PAIRS = [(src, dst) for src, targets in ALLOWED.items() for dst in sorted(targets)]


def _audit_rows(session: Session, claim: Claim) -> list[AuditLog]:
    stmt = select(AuditLog).where(AuditLog.entity_id == str(claim.id)).order_by(AuditLog.id)
    return list(session.scalars(stmt))


def test_allowed_matches_the_spec() -> None:
    s = ClaimStatus
    assert {
        s.RECEIVED: frozenset({s.PROCESSING}),
        s.PROCESSING: frozenset({s.NEEDS_REVIEW, s.FAILED}),
        s.NEEDS_REVIEW: frozenset({s.APPROVED_FOR_REVIEW, s.REJECTED_INCOMPLETE}),
        s.APPROVED_FOR_REVIEW: frozenset(),
        s.REJECTED_INCOMPLETE: frozenset(),
        s.FAILED: frozenset({s.PROCESSING}),
    } == ALLOWED


@pytest.mark.parametrize(("src", "dst"), PAIRS, ids=[f"{a}->{b}" for a, b in PAIRS])
def test_every_allowed_transition_writes_one_audit_row(
    sessions: SessionFactory, src: ClaimStatus, dst: ClaimStatus
) -> None:
    with sessions() as session:
        claim = add_claim(session, add_policy(session), status=src)
        transition(session, claim, dst, actor="adjuster:adj-1", reason="test")

    with sessions() as session:
        stored = session.get(Claim, claim.id)
        assert stored is not None and stored.status == dst.value
        rows = _audit_rows(session, stored)
        assert len(rows) == 1
        assert rows[0].actor == "adjuster:adj-1"
        assert rows[0].action == "claim.transition"
        assert rows[0].entity == "claim"
        assert rows[0].payload == {"from": src.value, "to": dst.value, "reason": "test"}


def test_disallowed_transition_raises_and_writes_nothing(sessions: SessionFactory) -> None:
    with sessions() as session:
        claim = add_claim(session, add_policy(session))
        with pytest.raises(InvalidTransition, match="received -> needs_review"):
            transition(session, claim, ClaimStatus.NEEDS_REVIEW, actor="worker")
        assert claim.status == ClaimStatus.RECEIVED.value
        assert _audit_rows(session, claim) == []


def test_terminal_states_allow_nothing(sessions: SessionFactory) -> None:
    with sessions() as session:
        claim = add_claim(session, add_policy(session), status=ClaimStatus.APPROVED_FOR_REVIEW)
        with pytest.raises(InvalidTransition):
            transition(session, claim, ClaimStatus.PROCESSING, actor="worker")


def test_rollback_undoes_status_and_audit_together(sessions: SessionFactory) -> None:
    with sessions() as session:
        claim_id = add_claim(session, add_policy(session)).id

    class Boom(Exception):
        pass

    with pytest.raises(Boom), sessions() as session:
        claim = session.get(Claim, claim_id)
        assert claim is not None
        transition(session, claim, ClaimStatus.PROCESSING, actor="worker")
        raise Boom

    with sessions() as session:
        claim = session.get(Claim, claim_id)
        assert claim is not None and claim.status == ClaimStatus.RECEIVED.value
        assert _audit_rows(session, claim) == []


def test_transition_reads_the_committed_status_not_a_stale_object(
    sessions: SessionFactory,
) -> None:
    with sessions() as session:
        claim_id = add_claim(session, add_policy(session)).id

    with sessions() as first:
        stale = first.get(Claim, claim_id)
        assert stale is not None
        # Another transaction moves the claim on before this one acts.
        with sessions() as second:
            other = second.get(Claim, claim_id)
            assert other is not None
            transition(second, other, ClaimStatus.PROCESSING, actor="worker")
        with pytest.raises(InvalidTransition, match="processing -> processing"):
            transition(first, stale, ClaimStatus.PROCESSING, actor="worker")


def test_record_appends_an_audit_row(sessions: SessionFactory) -> None:
    with sessions() as session:
        record(
            session,
            actor="worker",
            action="job.failed",
            entity="job",
            entity_id="j-1",
            payload={"attempt": 1},
        )
    with sessions() as session:
        count = session.scalar(select(func.count()).select_from(AuditLog))
        row = session.scalars(select(AuditLog)).one()
        assert count == 1
        assert (row.action, row.entity, row.entity_id, row.payload) == (
            "job.failed",
            "job",
            "j-1",
            {"attempt": 1},
        )
        assert row.at is not None
