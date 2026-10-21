"""Clause retrieval that can only see the policy version in force on the incident date."""

import uuid
from dataclasses import dataclass
from datetime import date

from sqlalchemy import select
from sqlalchemy.orm import Session

from db.models import PolicyClause
from db.policy_in_force import policy_in_force
from rag.embedder import Embedder


@dataclass(frozen=True)
class RetrievedClause:
    clause_ref: str
    title: str
    text: str
    version_no: int
    # Cosine similarity, 1 - cosine distance: 1.0 is the same direction.
    score: float


def retrieve_clauses(
    session: Session,
    *,
    policy_id: uuid.UUID,
    incident_date: date,
    query: str,
    embedder: Embedder,
    k: int = 5,
) -> list[RetrievedClause]:
    """The k clauses closest to `query` among those of the version in force on `incident_date`.

    The version is resolved first and the SQL filters on it before ordering by distance, so a
    clause from another version cannot take a place in the top k. No version in force (unknown
    policy, date before the first version, gap) gives an empty list, never the nearest version.
    """
    version = policy_in_force(session, policy_id, incident_date)
    if version is None:
        return []
    query_vector = embedder.embed([query])[0]
    distance = PolicyClause.embedding.cosine_distance(query_vector)
    stmt = (
        select(PolicyClause, distance.label("distance"))
        .where(PolicyClause.policy_version_id == version.id)
        .order_by(distance, PolicyClause.clause_ref)
        .limit(k)
    )
    return [
        RetrievedClause(
            clause_ref=clause.clause_ref,
            title=clause.title,
            text=clause.text,
            version_no=version.version_no,
            score=1.0 - float(dist),
        )
        for clause, dist in session.execute(stmt)
    ]
