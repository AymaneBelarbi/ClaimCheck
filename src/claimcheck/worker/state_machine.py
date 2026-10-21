"""The claim state machine. `transition` is the only code that changes a claim's status."""

from sqlalchemy.orm import Session

from claimcheck.core.types import ClaimStatus
from claimcheck.db.models import Claim
from claimcheck.worker.audit import record

_S = ClaimStatus

ALLOWED: dict[ClaimStatus, frozenset[ClaimStatus]] = {
    _S.RECEIVED: frozenset({_S.PROCESSING}),
    _S.PROCESSING: frozenset({_S.NEEDS_REVIEW, _S.FAILED}),
    _S.NEEDS_REVIEW: frozenset({_S.APPROVED_FOR_REVIEW, _S.REJECTED_INCOMPLETE}),
    _S.APPROVED_FOR_REVIEW: frozenset(),
    _S.REJECTED_INCOMPLETE: frozenset(),
    _S.FAILED: frozenset({_S.PROCESSING}),
}


class InvalidTransition(Exception):
    """The claim's current status does not allow the requested one."""


def transition(
    session: Session,
    claim: Claim,
    to: ClaimStatus,
    *,
    actor: str,
    reason: str | None = None,
) -> None:
    """Move `claim` to `to` and audit it, both in the caller's transaction.

    The row is locked and re-read first, so the check runs against the committed status rather
    than whatever an older object in memory says, and two callers cannot both move the claim
    out of the same state.
    """
    session.refresh(claim, with_for_update=True)
    current = ClaimStatus(claim.status)
    if to not in ALLOWED[current]:
        raise InvalidTransition(f"claim {claim.id}: {current.value} -> {to.value} is not allowed")
    claim.status = to.value
    record(
        session,
        actor=actor,
        action="claim.transition",
        entity="claim",
        entity_id=str(claim.id),
        payload={"from": current.value, "to": to.value, "reason": reason},
    )
    session.flush()
