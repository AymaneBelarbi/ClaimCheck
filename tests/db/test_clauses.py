from collections.abc import Iterator
from datetime import date
from decimal import Decimal

import pytest
from sqlalchemy import create_engine
from sqlalchemy.engine import URL
from sqlalchemy.orm import Session

from claimcheck.db.clauses import coverage_limit
from claimcheck.db.models import Policy, PolicyClause, PolicyVersion

pytestmark = pytest.mark.db


@pytest.fixture
def session(migrated_url: URL) -> Iterator[Session]:
    engine = create_engine(migrated_url)
    with Session(engine) as s:
        yield s
        s.rollback()
    engine.dispose()


def _version(session: Session, number: str) -> PolicyVersion:
    policy = Policy(policy_number=number, holder_id="cust-1")
    session.add(policy)
    session.flush()
    version = PolicyVersion(policy_id=policy.id, version_no=1, effective_from=date(2024, 1, 1))
    session.add(version)
    session.flush()
    return version


def _clause(
    session: Session,
    version: PolicyVersion,
    ref: str,
    coverage_type: str | None,
    limit: str | None,
) -> None:
    session.add(
        PolicyClause(
            policy_version_id=version.id,
            clause_ref=ref,
            title=f"Clause {ref}",
            coverage_type=coverage_type,
            limit_amount=Decimal(limit) if limit else None,
            text="...",
            embedding=[1.0] + [0.0] * 383,
        )
    )
    session.flush()


def test_limit_of_the_matching_clause(session: Session) -> None:
    version = _version(session, "AUT-900001")
    _clause(session, version, "4.1", "collision", "20000.00")
    _clause(session, version, "4.2", "glass", "1500.00")
    assert coverage_limit(session, version.id, "glass") == Decimal("1500.00")
    assert coverage_limit(session, version.id, "collision") == Decimal("20000.00")


def test_no_matching_clause(session: Session) -> None:
    version = _version(session, "AUT-900002")
    _clause(session, version, "4.1", "collision", "20000.00")
    _clause(session, version, "6.3", None, None)
    assert coverage_limit(session, version.id, "theft") is None


def test_only_the_given_version_counts(session: Session) -> None:
    first = _version(session, "AUT-900003")
    second = _version(session, "AUT-900004")
    _clause(session, first, "4.2", "glass", "1500.00")
    assert coverage_limit(session, second.id, "glass") is None


def test_first_limited_clause_by_ref_wins(session: Session) -> None:
    version = _version(session, "AUT-900005")
    _clause(session, version, "4.4", "glass", "900.00")
    _clause(session, version, "4.2", "glass", None)
    _clause(session, version, "4.3", "glass", "1500.00")
    assert coverage_limit(session, version.id, "glass") == Decimal("1500.00")
