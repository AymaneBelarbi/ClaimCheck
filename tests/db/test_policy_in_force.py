import uuid
from collections.abc import Iterator
from datetime import date

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL
from sqlalchemy.orm import Session

from db.models import Policy, PolicyVersion
from db.policy_in_force import OverlappingVersionsError, policy_in_force

pytestmark = pytest.mark.db


@pytest.fixture
def session(migrated_url: URL) -> Iterator[Session]:
    """A session on the shared migrated database; everything a test writes is rolled back."""
    engine = create_engine(migrated_url)
    with Session(engine) as s:
        yield s
        s.rollback()
    engine.dispose()


def _policy(session: Session) -> Policy:
    policy = Policy(policy_number=f"P-{uuid.uuid4().hex[:8]}", holder_id="holder-1")
    session.add(policy)
    session.flush()
    return policy


def _version(
    session: Session, policy: Policy, no: int, start: date, end: date | None
) -> PolicyVersion:
    version = PolicyVersion(
        policy_id=policy.id, version_no=no, effective_from=start, effective_to=end
    )
    session.add(version)
    session.flush()
    return version


@pytest.fixture
def policy(session: Session) -> Policy:
    p = _policy(session)
    _version(session, p, 1, date(2025, 1, 1), date(2025, 7, 1))
    _version(session, p, 2, date(2025, 7, 1), date(2026, 1, 1))
    _version(session, p, 3, date(2026, 1, 1), None)
    return p


def _in_force(session: Session, policy: Policy, day: date) -> PolicyVersion | None:
    return policy_in_force(session, policy.id, day)


def test_day_inside_first_version(session: Session, policy: Policy) -> None:
    found = _in_force(session, policy, date(2025, 6, 30))
    assert found is not None
    assert found.version_no == 1


def test_effective_to_is_exclusive_and_effective_from_inclusive(
    session: Session, policy: Policy
) -> None:
    last_day_of_v1 = _in_force(session, policy, date(2025, 6, 30))
    boundary = _in_force(session, policy, date(2025, 7, 1))
    assert last_day_of_v1 is not None and last_day_of_v1.version_no == 1
    assert boundary is not None and boundary.version_no == 2


def test_first_day_of_first_version_is_covered(session: Session, policy: Policy) -> None:
    found = _in_force(session, policy, date(2025, 1, 1))
    assert found is not None
    assert found.version_no == 1


def test_open_ended_version_covers_the_far_future(session: Session, policy: Policy) -> None:
    found = _in_force(session, policy, date(2031, 1, 1))
    assert found is not None
    assert found.version_no == 3


def test_date_before_any_version_returns_none(session: Session, policy: Policy) -> None:
    assert _in_force(session, policy, date(2024, 12, 31)) is None


def test_unknown_policy_returns_none(session: Session, policy: Policy) -> None:
    assert policy_in_force(session, uuid.uuid4(), date(2025, 8, 1)) is None


def test_other_policies_versions_are_not_returned(session: Session, policy: Policy) -> None:
    other = _policy(session)
    _version(session, other, 1, date(2025, 1, 1), None)
    found = _in_force(session, policy, date(2025, 8, 1))
    assert found is not None
    assert found.policy_id == policy.id
    assert found.version_no == 2


def test_gap_between_versions_returns_none(session: Session) -> None:
    p = _policy(session)
    _version(session, p, 1, date(2025, 1, 1), date(2025, 3, 1))
    _version(session, p, 2, date(2025, 4, 1), None)
    assert policy_in_force(session, p.id, date(2025, 3, 15)) is None


def test_overlapping_rows_raise_even_without_the_constraint(session: Session) -> None:
    p = _policy(session)
    # DDL is transactional in Postgres, so the constraint comes back on rollback.
    session.execute(
        text("ALTER TABLE policy_versions DROP CONSTRAINT ex_policy_versions_no_overlap")
    )
    _version(session, p, 1, date(2025, 1, 1), date(2025, 9, 1))
    _version(session, p, 2, date(2025, 6, 1), None)
    with pytest.raises(OverlappingVersionsError):
        policy_in_force(session, p.id, date(2025, 7, 1))
    # Outside the overlap the answer is still unambiguous.
    found = policy_in_force(session, p.id, date(2025, 2, 1))
    assert found is not None
    assert found.version_no == 1
