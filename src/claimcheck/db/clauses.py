"""Clause lookups that are always scoped to one policy version."""

import uuid
from decimal import Decimal

from sqlalchemy import select
from sqlalchemy.orm import Session

from claimcheck.db.models import PolicyClause


def coverage_limit(session: Session, version_id: uuid.UUID, coverage_type: str) -> Decimal | None:
    """The limit of the version's clause for `coverage_type`, or None if it has none.

    Clauses that only mention the coverage carry no limit and are skipped, as the evals do. If
    a version ever had two limited clauses of one type, the lowest clause_ref decides.
    """
    stmt = (
        select(PolicyClause.limit_amount)
        .where(
            PolicyClause.policy_version_id == version_id,
            PolicyClause.coverage_type == coverage_type,
            PolicyClause.limit_amount.is_not(None),
        )
        .order_by(PolicyClause.clause_ref)
        .limit(1)
    )
    return session.scalar(stmt)
