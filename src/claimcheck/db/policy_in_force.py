"""Which policy version applied on a given day.

Every clause lookup goes through `policy_in_force`, so a claim is only ever checked against
the wording that was in force when the incident happened.
"""

import uuid
from datetime import date

from sqlalchemy import Date, func, literal_column, select
from sqlalchemy.orm import Session

from claimcheck.db.models import PolicyVersion

# Open-ended versions (effective_to IS NULL) compare as "never ends".
_INFINITY = literal_column("'infinity'::date", Date)


class OverlappingVersionsError(Exception):
    """More than one version of a policy matched a single date.

    The exclusion constraint on policy_versions should make this impossible; the check exists so
    a damaged table fails loudly instead of returning an arbitrary version.
    """


def policy_in_force(
    session: Session, policy_id: uuid.UUID, incident_date: date
) -> PolicyVersion | None:
    """Return the version with effective_from <= incident_date < effective_to, or None.

    The start is inclusive and the end exclusive, so a version that ends on 2025-07-01 and its
    successor that starts on 2025-07-01 never both apply.
    """
    stmt = (
        select(PolicyVersion)
        .where(
            PolicyVersion.policy_id == policy_id,
            PolicyVersion.effective_from <= incident_date,
            incident_date < func.coalesce(PolicyVersion.effective_to, _INFINITY),
        )
        .order_by(PolicyVersion.version_no)
        # Two rows are enough to know the answer is ambiguous.
        .limit(2)
    )
    matches = session.scalars(stmt).all()
    if len(matches) > 1:
        raise OverlappingVersionsError(
            f"policy {policy_id} has versions "
            f"{', '.join(str(v.version_no) for v in matches)} in force on {incident_date}"
        )
    return matches[0] if matches else None
