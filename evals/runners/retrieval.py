"""Retrieval stage: does the clause that answers a question come back in the top 5?

Reads `clause_questions.jsonl` (next to the data directory, i.e. `evals/clause_questions.jsonl`
for the default `evals/data`) and runs each question through `retrieve_clauses` against a
seeded database, as the agent's clause tool will. A question is answered when the expected
clause of the expected version is among the 5 results.

The question file has no dev/holdout field, so the split is made here from the question id:
the first 8 bytes of sha256(id) modulo 10, below 7 is dev, otherwise holdout. That is a 70/30
split that does not move when questions are added, and it matches the proportion of the
packets (35/15). On the current 49 questions it gives 35 dev and 14 holdout.

Metrics:
    retrieval.recall_at_5          share of questions whose expected clause is in the top 5
    retrieval.recall_at_1          share whose expected clause comes first (not gated: with 15
                                   clauses a version and k = 5, recall@5 says little alone)
    retrieval.mrr                  mean of 1/rank of the expected clause (0 when not in the top 5)
    retrieval.wrong_version_hits   retrieved clauses whose version is not the expected one;
                                   must be 0, the version filter makes it so

`details.control` reruns the same questions with the version filter removed: the k nearest
clauses among all versions of the policy. Nothing in the product searches that way; it is here
to measure what the filter prevents. `control_wrong_version_top1` counts questions whose first
clause comes from another version, `control_wrong_version_hits` the returned clauses that do.
"""

import hashlib
import uuid
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from claimcheck.db.models import Policy, PolicyClause, PolicyVersion
from claimcheck.db.session import engine_for
from claimcheck.rag.embedder import Embedder
from claimcheck.rag.retrieve import RetrievedClause, retrieve_clauses
from evals.context import EvalContext, StageUnavailable
from evals.dataset.types import ClauseQuestion
from evals.report import StageResult

K = 5
DEV_BUCKETS = 7


def split_of(question_id: str) -> str:
    bucket = int.from_bytes(hashlib.sha256(question_id.encode()).digest()[:8], "big") % 10
    return "dev" if bucket < DEV_BUCKETS else "holdout"


def run(split: str, ctx: EvalContext) -> StageResult:
    path = ctx.data_dir.parent / "clause_questions.jsonl"
    questions = [
        q
        for q in (
            ClauseQuestion.model_validate_json(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
        if split_of(q.id) == split
    ]

    per_question: list[dict[str, Any]] = []
    misses: list[dict[str, Any]] = []
    wrong_version = 0
    control_top1 = control_hits = control_returned = control_answered = 0
    with Session(engine_for(ctx.settings.database_url)) as session:
        policies = _policy_ids(session, {q.policy_number for q in questions})
        for q in questions:
            hits = retrieve_clauses(
                session,
                policy_id=policies[q.policy_number],
                incident_date=q.incident_date,
                query=q.question,
                embedder=ctx.embedder,
                k=K,
            )
            wrong_version += sum(h.version_no != q.expected_version_no for h in hits)
            rank = next(
                (
                    i
                    for i, h in enumerate(hits, start=1)
                    if h.clause_ref == q.expected_clause_ref
                    and h.version_no == q.expected_version_no
                ),
                None,
            )
            per_question.append({"id": q.id, "rank": rank})

            unfiltered = _unfiltered(session, policies[q.policy_number], q.question, ctx.embedder)
            control_returned += len(unfiltered)
            control_hits += sum(h.version_no != q.expected_version_no for h in unfiltered)
            control_top1 += bool(unfiltered) and unfiltered[0].version_no != q.expected_version_no
            control_answered += any(
                h.clause_ref == q.expected_clause_ref and h.version_no == q.expected_version_no
                for h in unfiltered
            )
            if rank is None:
                misses.append(
                    {
                        "id": q.id,
                        "policy_number": q.policy_number,
                        "incident_date": q.incident_date.isoformat(),
                        "question": q.question,
                        "expected_clause_ref": q.expected_clause_ref,
                        "expected_version_no": q.expected_version_no,
                        "retrieved": [
                            {
                                "clause_ref": h.clause_ref,
                                "version_no": h.version_no,
                                "score": round(h.score, 4),
                            }
                            for h in hits
                        ],
                    }
                )

    total = len(questions)
    answered = sum(1 for r in per_question if r["rank"] is not None)
    reciprocal = sum(1 / r["rank"] for r in per_question if r["rank"] is not None)
    first = sum(1 for r in per_question if r["rank"] == 1)
    metrics = {
        "retrieval.recall_at_5": answered / total if total else 0.0,
        "retrieval.recall_at_1": first / total if total else 0.0,
        "retrieval.mrr": reciprocal / total if total else 0.0,
        "retrieval.wrong_version_hits": float(wrong_version),
    }
    details: dict[str, Any] = {
        "split": split,
        "questions": total,
        "per_question": per_question,
        "misses": misses,
        "control": {
            "description": "same questions, version filter removed (all versions of the policy)",
            "control_wrong_version_top1": {"count": control_top1, "total": total},
            "control_wrong_version_hits": {"count": control_hits, "returned": control_returned},
            "control_recall_at_5": control_answered / total if total else 0.0,
        },
    }
    return StageResult(metrics=metrics, details=details)


def _policy_ids(session: Session, numbers: set[str]) -> dict[str, uuid.UUID]:
    if not session.scalar(select(func.count()).select_from(Policy)):
        raise StageUnavailable("no policies in the database; run make seed first")
    found = {
        p.policy_number: p.id
        for p in session.scalars(select(Policy).where(Policy.policy_number.in_(numbers)))
    }
    missing = sorted(numbers - found.keys())
    if missing:
        raise StageUnavailable(
            f"policies {', '.join(missing)} are not in the database; run make seed first"
        )
    return found


def _unfiltered(
    session: Session, policy_id: uuid.UUID, query: str, embedder: Embedder
) -> list[RetrievedClause]:
    """The k nearest clauses across every version of the policy: the search the version filter
    exists to prevent, run only to measure it."""
    distance = PolicyClause.embedding.cosine_distance(embedder.embed([query])[0])
    stmt = (
        select(PolicyClause, PolicyVersion.version_no, distance.label("distance"))
        .join(PolicyVersion, PolicyVersion.id == PolicyClause.policy_version_id)
        .where(PolicyVersion.policy_id == policy_id)
        .order_by(distance, PolicyClause.clause_ref, PolicyVersion.version_no)
        .limit(K)
    )
    return [
        RetrievedClause(clause.clause_ref, clause.title, clause.text, version_no, 1.0 - float(dist))
        for clause, version_no, dist in session.execute(stmt)
    ]
