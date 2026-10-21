import uuid
from datetime import date

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from claimcheck.db.models import Policy, PolicyClause, PolicyVersion
from claimcheck.rag.ingest import ingest_policy_version
from claimcheck.rag.retrieve import RetrievedClause, retrieve_clauses
from tests.fakes import HashingEmbedder

pytestmark = pytest.mark.db

V1 = """# Policy, version 1

## 4.1 Collision damage

We cover accidental damage caused by a collision, up to a limit of $15,000 per claim.

## 4.2 Glass breakage

We cover the windscreen and side windows after a crack or chip, up to a limit of $1,000 per claim.

## 5.1 Excess

You pay $500 on each collision claim.
"""

V2 = """# Policy, version 2

## 4.1 Collision damage

We cover accidental damage caused by a collision, up to a limit of $20,000 per claim.

## 4.2 Glass breakage

We cover windscreen cracks, sunroof panels and camera recalibration, up to a limit of
$1,500 per claim.

## 5.1 Excess

You pay $750 on each collision claim.
"""

EMBEDDER = HashingEmbedder()


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
    v1 = _version(session, p, 1, date(2025, 1, 1), date(2025, 7, 1))
    v2 = _version(session, p, 2, date(2025, 7, 1), None)
    ingest_policy_version(session, v1.id, V1, EMBEDDER)
    ingest_policy_version(session, v2.id, V2, EMBEDDER)
    return p


def _ask(
    session: Session, policy: Policy, day: date, query: str, k: int = 5
) -> list[RetrievedClause]:
    return retrieve_clauses(
        session, policy_id=policy.id, incident_date=day, query=query, embedder=EMBEDDER, k=k
    )


def test_a_date_in_v1_returns_only_v1_clauses(session: Session, policy: Policy) -> None:
    hits = _ask(session, policy, date(2025, 3, 1), "windscreen crack camera recalibration")

    assert hits
    assert {h.version_no for h in hits} == {1}


def test_a_date_in_v2_returns_only_v2_clauses(session: Session, policy: Policy) -> None:
    hits = _ask(session, policy, date(2025, 9, 1), "windscreen crack chip")

    assert hits
    assert {h.version_no for h in hits} == {2}


def test_the_wrong_era_clause_is_never_returned_even_when_it_matches_best(
    session: Session, policy: Policy
) -> None:
    # The query is v2's glass wording; asked about a v1 date, v1's glass clause must still win.
    hits = _ask(
        session,
        policy,
        date(2025, 6, 30),
        "We cover windscreen cracks, sunroof panels and camera recalibration",
    )

    assert hits[0].clause_ref == "4.2"
    assert hits[0].version_no == 1
    assert "$1,000" in hits[0].text


def test_the_boundary_day_belongs_to_the_new_version(session: Session, policy: Policy) -> None:
    hits = _ask(session, policy, date(2025, 7, 1), "glass")

    assert {h.version_no for h in hits} == {2}


def test_a_date_before_the_first_version_returns_nothing(session: Session, policy: Policy) -> None:
    assert _ask(session, policy, date(2024, 12, 31), "windscreen") == []


def test_an_unknown_policy_returns_nothing(session: Session, policy: Policy) -> None:
    hits = retrieve_clauses(
        session,
        policy_id=uuid.uuid4(),
        incident_date=date(2025, 3, 1),
        query="windscreen",
        embedder=EMBEDDER,
    )

    assert hits == []


def test_results_are_ranked_by_score_and_limited_to_k(session: Session, policy: Policy) -> None:
    hits = _ask(session, policy, date(2025, 3, 1), "windscreen side windows crack chip", k=2)

    assert len(hits) == 2
    assert hits[0].clause_ref == "4.2"
    assert hits[0].score >= hits[1].score
    assert 0.0 < hits[0].score <= 1.0


def test_default_k_is_five(session: Session) -> None:
    p = _policy(session)
    v = _version(session, p, 1, date(2025, 1, 1), None)
    sections = "\n\n".join(f"## 1.{n} Clause {n}\n\nText number {n}." for n in range(1, 9))
    ingest_policy_version(session, v.id, sections, EMBEDDER)

    hits = retrieve_clauses(
        session,
        policy_id=p.id,
        incident_date=date(2025, 2, 1),
        query="text number",
        embedder=EMBEDDER,
    )

    assert len(hits) == 5


def test_reingesting_a_version_does_not_duplicate_clauses(session: Session, policy: Policy) -> None:
    v1 = session.scalars(
        select(PolicyVersion).where(
            PolicyVersion.policy_id == policy.id, PolicyVersion.version_no == 1
        )
    ).one()

    count = ingest_policy_version(session, v1.id, V1, EMBEDDER)

    assert count == 3
    stored = session.scalar(
        select(func.count())
        .select_from(PolicyClause)
        .where(PolicyClause.policy_version_id == v1.id)
    )
    assert stored == 3


def test_reingesting_replaces_the_text_of_a_version(session: Session, policy: Policy) -> None:
    v1 = session.scalars(
        select(PolicyVersion).where(
            PolicyVersion.policy_id == policy.id, PolicyVersion.version_no == 1
        )
    ).one()

    ingest_policy_version(session, v1.id, V1.replace("$1,000", "$1,250"), EMBEDDER)

    glass = session.scalars(
        select(PolicyClause).where(
            PolicyClause.policy_version_id == v1.id, PolicyClause.clause_ref == "4.2"
        )
    ).one()
    assert "$1,250" in glass.text
    assert glass.limit_amount is not None
    assert int(glass.limit_amount) == 1250


def test_reingesting_one_version_leaves_the_others_alone(session: Session, policy: Policy) -> None:
    v2 = session.scalars(
        select(PolicyVersion).where(
            PolicyVersion.policy_id == policy.id, PolicyVersion.version_no == 2
        )
    ).one()
    v1 = session.scalars(
        select(PolicyVersion).where(
            PolicyVersion.policy_id == policy.id, PolicyVersion.version_no == 1
        )
    ).one()

    ingest_policy_version(session, v1.id, V1, EMBEDDER)

    stored = session.scalar(
        select(func.count())
        .select_from(PolicyClause)
        .where(PolicyClause.policy_version_id == v2.id)
    )
    assert stored == 3


def test_ingest_stores_structure_and_embeds_title_and_text(session: Session) -> None:
    seen: list[str] = []

    class Recording(HashingEmbedder):
        def embed(self, texts: list[str]) -> list[list[float]]:
            seen.extend(texts)
            return super().embed(texts)

    p = _policy(session)
    v = _version(session, p, 1, date(2025, 1, 1), None)

    ingest_policy_version(session, v.id, V1, Recording())

    assert seen[1] == (
        "Glass breakage\nWe cover the windscreen and side windows after a crack or chip, "
        "up to a limit of $1,000 per claim."
    )
    glass = session.scalars(
        select(PolicyClause).where(
            PolicyClause.policy_version_id == v.id, PolicyClause.clause_ref == "4.2"
        )
    ).one()
    assert glass.coverage_type == "glass"
    assert glass.title == "Glass breakage"
