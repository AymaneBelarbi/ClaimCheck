"""The API's database sessions: the claimcheck_app role, with the caller's identity on them."""

import uuid
from datetime import date

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Engine, text

from claimcheck.api.auth import Principal
from claimcheck.api.deps import api_engine, get_session
from claimcheck.core.types import Role
from claimcheck.db.models import Claim, Policy
from claimcheck.db.session import SessionFactory
from tests.api.conftest import Headers

PROBE = text(
    "SELECT current_user, current_setting('app.role', true), current_setting('app.user_id', true)"
)


def test_api_connects_as_the_app_role() -> None:
    assert api_engine().url.username == "claimcheck_app"


@pytest.mark.db
def test_request_session_keeps_the_callers_identity_after_a_rollback(
    app_role_engine: Engine, sessions: SessionFactory
) -> None:
    requests = get_session(Principal(sub="cust-1", role=Role.CUSTOMER), app_role_engine)
    session = next(requests)
    try:
        assert tuple(session.execute(PROBE).one()) == ("claimcheck_app", "customer", "cust-1")
        session.rollback()
        assert tuple(session.execute(PROBE).one()) == ("claimcheck_app", "customer", "cust-1")
        session.commit()
        assert tuple(session.execute(PROBE).one()) == ("claimcheck_app", "customer", "cust-1")
    finally:
        next(requests, None)  # finish the dependency, which closes the session


def _mismatched_claim(sessions: SessionFactory) -> uuid.UUID:
    """A claim on cust-1's policy recorded as created for cust-2.

    The API never writes such a row (created_by is the policy holder); it stands in for a bug
    in the code's own check, which looks at the policy holder, so that only row-level security
    is left to keep the claim from cust-1.
    """
    with sessions() as session:
        policy = Policy(policy_number=f"AUT-{uuid.uuid4().hex[:6]}", holder_id="cust-1")
        session.add(policy)
        session.flush()
        claim = Claim(
            packet_id=f"PKT-{uuid.uuid4().hex[:8]}",
            policy_id=policy.id,
            claim_type="glass",
            incident_date=date(2024, 6, 1),
            created_by="cust-2",
        )
        session.add(claim)
        session.flush()
        return claim.id


@pytest.mark.db
def test_row_security_hides_a_claim_the_code_check_would_let_through(
    client: TestClient, sessions: SessionFactory, cust1: Headers, cust2: Headers, adjuster: Headers
) -> None:
    claim_id = _mismatched_claim(sessions)

    assert client.get(f"/claims/{claim_id}", headers=cust1).status_code == 404
    assert client.get("/claims", headers=cust1).json() == []
    assert client.get(f"/claims/{claim_id}", headers=adjuster).status_code == 200


@pytest.mark.db
def test_health_needs_no_token(client: TestClient) -> None:
    assert client.get("/health").json() == {"status": "ok", "db": True}
