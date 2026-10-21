"""Append-only audit trail. Rows are added in the caller's transaction, never updated."""

from typing import Any

from sqlalchemy.orm import Session

from claimcheck.db.models import AuditLog


def record(
    session: Session,
    *,
    actor: str,
    action: str,
    entity: str,
    entity_id: str,
    payload: dict[str, Any],
) -> None:
    """Add one audit row; it commits or rolls back with whatever change it describes."""
    session.add(
        AuditLog(actor=actor, action=action, entity=entity, entity_id=entity_id, payload=payload)
    )
