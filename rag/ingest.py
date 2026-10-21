"""Write a policy version's clauses, with their embeddings, into policy_clauses."""

import uuid

from sqlalchemy import delete
from sqlalchemy.orm import Session

from db.models import PolicyClause
from rag.chunking import chunk_clauses
from rag.embedder import Embedder


def ingest_policy_version(
    session: Session, version_id: uuid.UUID, markdown: str, embedder: Embedder
) -> int:
    """Replace the clauses of one version with those in `markdown`; returns how many were stored.

    The delete and the inserts run in the caller's transaction, so a failure part-way leaves the
    previous clauses in place. Only this version's rows are touched. The title is embedded with
    the text because many clauses say little without it ("We do not pay for ..." under "Hire
    cars and replacement transport").
    """
    clauses = chunk_clauses(markdown)
    vectors = embedder.embed([f"{c.title}\n{c.text}" for c in clauses])
    session.execute(delete(PolicyClause).where(PolicyClause.policy_version_id == version_id))
    session.add_all(
        PolicyClause(
            policy_version_id=version_id,
            clause_ref=clause.clause_ref,
            title=clause.title,
            coverage_type=clause.coverage_type,
            limit_amount=clause.limit_amount,
            text=clause.text,
            embedding=vector,
        )
        for clause, vector in zip(clauses, vectors, strict=True)
    )
    session.flush()
    return len(clauses)
