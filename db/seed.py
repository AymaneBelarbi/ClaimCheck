"""Load the generated policies into the database.

    python -m db.seed --data evals/data [--reembed]

Reads `<data>/policies/index.json` and the markdown it points to. Policies and versions are
upserted by policy number and version number; clauses are ingested for a version when its
stored clauses differ from the markdown, or for every version with --reembed (after changing
the embedding model or the text that gets embedded). Running it twice changes nothing.
"""

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from core.settings import get_settings
from db.models import Policy, PolicyClause, PolicyVersion
from db.session import engine_for
from evals.dataset.types import PolicyEntry, PolicyIndex, PolicyVersionEntry
from rag.chunking import chunk_clauses
from rag.embedder import Embedder, FastEmbedEmbedder
from rag.ingest import ingest_policy_version


class SeedError(Exception):
    """The data on disk cannot be applied to what the database already holds."""


@dataclass(frozen=True)
class SeedSummary:
    policies: int
    versions: int
    # Clauses written by this run; 0 when everything was already up to date.
    clauses: int


def seed_database(
    session: Session, data_dir: Path, embedder: Embedder, *, reembed: bool = False
) -> SeedSummary:
    """Upsert everything in `data_dir`/policies. The caller commits."""
    index = PolicyIndex.model_validate_json(
        (data_dir / "policies" / "index.json").read_text(encoding="utf-8")
    )
    versions = clauses = 0
    for entry in index.policies:
        policy = _upsert_policy(session, entry)
        for version_entry in entry.versions:
            version = _upsert_version(session, policy, version_entry)
            markdown = (data_dir / "policies" / version_entry.file).read_text(encoding="utf-8")
            if not chunk_clauses(markdown):
                # Ingesting it would delete the stored clauses and insert none, which empties a
                # version that must never lose its wording (ADR-002).
                raise SeedError(
                    f"{entry.policy_number} v{version_entry.version_no} ({version_entry.file}) "
                    "has no clauses ('## <ref> <title>' sections); nothing was changed"
                )
            versions += 1
            if reembed or _clauses_differ(session, version, markdown):
                clauses += ingest_policy_version(session, version.id, markdown, embedder)
    return SeedSummary(policies=len(index.policies), versions=versions, clauses=clauses)


def _upsert_policy(session: Session, entry: PolicyEntry) -> Policy:
    policy = session.scalars(
        select(Policy).where(Policy.policy_number == entry.policy_number)
    ).one_or_none()
    if policy is None:
        policy = Policy(policy_number=entry.policy_number, holder_id=entry.holder_id)
        session.add(policy)
    else:
        policy.holder_id = entry.holder_id
    session.flush()
    return policy


def _upsert_version(session: Session, policy: Policy, entry: PolicyVersionEntry) -> PolicyVersion:
    document_url = f"policies/{entry.file}"
    version = session.scalars(
        select(PolicyVersion).where(
            PolicyVersion.policy_id == policy.id, PolicyVersion.version_no == entry.version_no
        )
    ).one_or_none()
    if version is None:
        version = PolicyVersion(
            policy_id=policy.id,
            version_no=entry.version_no,
            effective_from=entry.effective_from,
            effective_to=entry.effective_to,
            document_url=document_url,
        )
        session.add(version)
    elif (version.effective_from, version.effective_to) != (
        entry.effective_from,
        entry.effective_to,
    ):
        # Versions are never edited once published (ADR-002); a different range means the
        # dataset was regenerated and the development database should be recreated.
        raise SeedError(
            f"{policy.policy_number} v{entry.version_no} is stored with different dates "
            f"({version.effective_from} to {version.effective_to}) than the data "
            f"({entry.effective_from} to {entry.effective_to}); reset the database"
        )
    else:
        version.document_url = document_url
    session.flush()
    return version


def _clauses_differ(session: Session, version: PolicyVersion, markdown: str) -> bool:
    stored = {
        (c.clause_ref, c.title, c.text)
        for c in session.scalars(
            select(PolicyClause).where(PolicyClause.policy_version_id == version.id)
        )
    }
    return stored != {(c.clause_ref, c.title, c.text) for c in chunk_clauses(markdown)}


def main(argv: list[str] | None = None, *, embedder: Embedder | None = None) -> int:
    parser = argparse.ArgumentParser(description="Load the generated policies into the database.")
    parser.add_argument(
        "--data", type=Path, required=True, help="dataset directory, e.g. evals/data"
    )
    parser.add_argument("--reembed", action="store_true", help="re-embed every version's clauses")
    parser.add_argument("--database-url", default=None, help="default: DATABASE_URL from settings")
    args = parser.parse_args(argv)

    engine = engine_for(args.database_url or get_settings().database_url)
    try:
        with Session(engine) as session, session.begin():
            summary = seed_database(
                session, args.data, embedder or FastEmbedEmbedder(), reembed=args.reembed
            )
    except (SeedError, FileNotFoundError) as exc:
        print(f"seed failed: {exc}", file=sys.stderr)
        return 1
    print(
        f"{summary.policies} policies, {summary.versions} versions, "
        f"{summary.clauses} clauses written"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
