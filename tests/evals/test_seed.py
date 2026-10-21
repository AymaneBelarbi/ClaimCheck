import json
from collections.abc import Iterator
from datetime import date
from pathlib import Path

import pytest
from sqlalchemy import create_engine, func, select
from sqlalchemy.engine import URL
from sqlalchemy.orm import Session

from claimcheck.db.models import Policy, PolicyClause, PolicyVersion
from evals.seed import SeedError, seed_database
from tests.fakes import HashingEmbedder

pytestmark = pytest.mark.db

V1 = """# Policy AUT-777001, version 1

## 4.1 Collision damage

We cover collision damage, up to a limit of $15,000 per claim.

## 4.2 Glass breakage

We cover the windscreen, up to a limit of $1,000 per claim.
"""

V2 = """# Policy AUT-777001, version 2

## 4.1 Collision damage

We cover collision damage, up to a limit of $20,000 per claim.

## 4.2 Glass breakage

We cover the windscreen and the sunroof, up to a limit of $1,500 per claim.

## 4.4 Hire car

We pay for a hire car for up to 14 days.
"""


class CountingEmbedder(HashingEmbedder):
    def __init__(self) -> None:
        super().__init__()
        self.texts = 0

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.texts += len(texts)
        return super().embed(texts)


@pytest.fixture
def session(migrated_url: URL) -> Iterator[Session]:
    engine = create_engine(migrated_url)
    with Session(engine) as s:
        yield s
        s.rollback()
    engine.dispose()


def _index(holder: str = "cust-9", v1_to: str | None = "2025-01-01") -> dict[str, object]:
    return {
        "policies": [
            {
                "policy_number": "AUT-777001",
                "holder_id": holder,
                "versions": [
                    {
                        "version_no": 1,
                        "effective_from": "2024-01-01",
                        "effective_to": v1_to,
                        "file": "AUT-777001/v1.md",
                    },
                    {
                        "version_no": 2,
                        "effective_from": "2025-01-01",
                        "effective_to": None,
                        "file": "AUT-777001/v2.md",
                    },
                ],
            }
        ]
    }


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    policies = tmp_path / "policies"
    (policies / "AUT-777001").mkdir(parents=True)
    (policies / "AUT-777001" / "v1.md").write_text(V1, encoding="utf-8")
    (policies / "AUT-777001" / "v2.md").write_text(V2, encoding="utf-8")
    (policies / "index.json").write_text(json.dumps(_index()), encoding="utf-8")
    return tmp_path


def _clause_count(session: Session) -> int:
    return session.scalar(select(func.count()).select_from(PolicyClause)) or 0


def test_seeds_policy_versions_and_clauses(session: Session, data_dir: Path) -> None:
    before = _clause_count(session)

    summary = seed_database(session, data_dir, CountingEmbedder())

    policy = session.scalars(select(Policy).where(Policy.policy_number == "AUT-777001")).one()
    assert policy.holder_id == "cust-9"
    assert policy.product_line == "auto"
    versions = session.scalars(
        select(PolicyVersion).where(PolicyVersion.policy_id == policy.id).order_by("version_no")
    ).all()
    assert [(v.version_no, v.effective_from, v.effective_to) for v in versions] == [
        (1, date(2024, 1, 1), date(2025, 1, 1)),
        (2, date(2025, 1, 1), None),
    ]
    assert [v.document_url for v in versions] == [
        "policies/AUT-777001/v1.md",
        "policies/AUT-777001/v2.md",
    ]
    assert _clause_count(session) - before == 5
    assert (summary.policies, summary.versions, summary.clauses) == (1, 2, 5)


def test_seeding_twice_changes_nothing_and_does_not_re_embed(
    session: Session, data_dir: Path
) -> None:
    embedder = CountingEmbedder()
    seed_database(session, data_dir, embedder)
    clauses, embedded = _clause_count(session), embedder.texts

    summary = seed_database(session, data_dir, embedder)

    assert _clause_count(session) == clauses
    assert embedder.texts == embedded
    assert summary.clauses == 0
    assert (
        session.scalar(
            select(func.count()).select_from(Policy).where(Policy.policy_number == "AUT-777001")
        )
        == 1
    )


def test_changed_markdown_is_re_ingested(session: Session, data_dir: Path) -> None:
    seed_database(session, data_dir, CountingEmbedder())
    (data_dir / "policies" / "AUT-777001" / "v1.md").write_text(
        V1.replace("$1,000", "$1,100"), encoding="utf-8"
    )

    summary = seed_database(session, data_dir, CountingEmbedder())

    assert summary.clauses == 2
    glass = session.scalars(
        select(PolicyClause).where(
            PolicyClause.clause_ref == "4.2", PolicyClause.text.like("%$1,100%")
        )
    ).all()
    assert len(glass) == 1


def test_reembed_rebuilds_every_version(session: Session, data_dir: Path) -> None:
    seed_database(session, data_dir, CountingEmbedder())
    clauses = _clause_count(session)
    embedder = CountingEmbedder()

    summary = seed_database(session, data_dir, embedder, reembed=True)

    assert embedder.texts == 5
    assert summary.clauses == 5
    assert _clause_count(session) == clauses


def test_holder_is_updated_on_rerun(session: Session, data_dir: Path) -> None:
    seed_database(session, data_dir, CountingEmbedder())
    (data_dir / "policies" / "index.json").write_text(
        json.dumps(_index(holder="cust-10")), encoding="utf-8"
    )

    seed_database(session, data_dir, CountingEmbedder())

    policy = session.scalars(select(Policy).where(Policy.policy_number == "AUT-777001")).one()
    assert policy.holder_id == "cust-10"


def test_changed_dates_of_an_existing_version_are_refused(session: Session, data_dir: Path) -> None:
    seed_database(session, data_dir, CountingEmbedder())
    (data_dir / "policies" / "index.json").write_text(
        json.dumps(_index(v1_to="2024-12-01")), encoding="utf-8"
    )

    with pytest.raises(SeedError, match="AUT-777001 v1"):
        seed_database(session, data_dir, CountingEmbedder())


def test_markdown_without_clauses_is_refused(session: Session, data_dir: Path) -> None:
    """Ingesting it would delete the version's stored clauses and put nothing back."""
    seed_database(session, data_dir, CountingEmbedder())
    clauses = _clause_count(session)
    (data_dir / "policies" / "AUT-777001" / "v1.md").write_text(
        "# Policy AUT-777001, version 1\n\nThe headings were lost.\n", encoding="utf-8"
    )

    with pytest.raises(SeedError, match=r"AUT-777001 v1 .*no clauses"):
        seed_database(session, data_dir, CountingEmbedder(), reembed=True)

    assert _clause_count(session) == clauses
