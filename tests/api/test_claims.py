import uuid
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from api import limits
from api.routes import claims as claim_routes
from core.review import ReviewFile
from core.types import ClaimStatus
from db.models import AuditLog, Claim, ClaimDocument, Job, Policy, ValidationResult
from db.session import SessionFactory
from tests.api.conftest import Headers
from tests.worker.factories import add_claim
from worker.review_file import build_review_file
from worker.storage import Storage

pytestmark = pytest.mark.db

PDF = b"%PDF-1.4 test packet"


def _files(*names: str) -> list[tuple[str, tuple[str, bytes, str]]]:
    return [("files", (name, PDF + name.encode(), "application/pdf")) for name in names]


def upload(
    client: TestClient,
    headers: Headers,
    policy_number: str,
    *,
    key: str | None = "key-1",
    files: list[tuple[str, tuple[str, bytes, str]]] | None = None,
    data: dict[str, str] | None = None,
) -> Any:
    form = {
        "policy_number": policy_number,
        "claim_type": "collision",
        "incident_date": "2024-06-01",
    }
    form.update(data or {})
    sent = dict(headers)
    if key is not None:
        sent["Idempotency-Key"] = key
    return client.post(
        "/claims",
        headers=sent,
        data=form,
        files=files if files is not None else _files("form.pdf", "invoice.pdf"),
    )


def count(sessions: SessionFactory, model: type) -> int:
    with sessions() as session:
        return session.scalar(select(func.count()).select_from(model)) or 0


def stored_files(storage: Storage) -> list[Path]:
    return [p for p in storage.root.rglob("*") if p.is_file()] if storage.root.exists() else []


# ---- upload -------------------------------------------------------------------------------


def test_upload_creates_received_claim_documents_job_and_audit(
    client: TestClient,
    sessions: SessionFactory,
    storage: Storage,
    cust1: Headers,
    policy1: str,
) -> None:
    response = upload(client, cust1, policy1, key="PKT-77")

    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "received"
    claim_id = uuid.UUID(body["claim_id"])

    with sessions() as session:
        claim = session.get(Claim, claim_id)
        assert claim is not None
        assert claim.packet_id == "PKT-77"
        assert claim.status == "received"
        assert claim.claim_type == "collision"
        assert claim.incident_date == date(2024, 6, 1)
        assert claim.created_by == "cust-1"

        documents = list(session.scalars(select(ClaimDocument).order_by(ClaimDocument.filename)))
        assert [d.filename for d in documents] == ["form.pdf", "invoice.pdf"]
        assert all(d.claim_id == claim_id and d.doc_type is None for d in documents)
        assert storage.get(documents[0].storage_key) == PDF + b"form.pdf"

        job = session.scalars(select(Job)).one()
        assert (job.claim_id, job.state, job.attempts) == (claim_id, "queued", 0)

        audit = session.scalars(select(AuditLog).where(AuditLog.action == "claim.created")).one()
        assert audit.actor == "cust-1"
        assert audit.entity_id == str(claim_id)


def test_same_idempotency_key_returns_the_same_claim_and_no_second_job(
    client: TestClient,
    sessions: SessionFactory,
    storage: Storage,
    cust1: Headers,
    policy1: str,
) -> None:
    """A repeated key answers 200 with the first claim and creates no second claim or job."""
    first = upload(client, cust1, policy1, key="same-key")
    files_after_first = stored_files(storage)
    second = upload(client, cust1, policy1, key="same-key")

    assert first.status_code == 202
    assert second.status_code == 200
    assert second.json() == first.json()
    assert count(sessions, Claim) == 1
    assert count(sessions, Job) == 1
    assert count(sessions, ClaimDocument) == 2
    assert stored_files(storage) == files_after_first
    with sessions() as session:
        created = select(func.count()).where(AuditLog.action == "claim.created")
        assert session.scalar(created) == 1


def test_replay_ignores_a_different_payload(
    client: TestClient, sessions: SessionFactory, cust1: Headers, policy1: str
) -> None:
    first = upload(client, cust1, policy1, key="k")
    second = upload(client, cust1, policy1, key="k", files=_files("other.pdf"))
    assert second.status_code == 200
    assert second.json()["claim_id"] == first.json()["claim_id"]
    assert count(sessions, ClaimDocument) == 2


def test_replay_reports_the_current_status(
    client: TestClient, sessions: SessionFactory, cust1: Headers, policy1: str
) -> None:
    first = upload(client, cust1, policy1, key="k")
    with sessions() as session:
        claim = session.get(Claim, uuid.UUID(first.json()["claim_id"]))
        assert claim is not None
        claim.status = ClaimStatus.NEEDS_REVIEW.value
    assert upload(client, cust1, policy1, key="k").json()["status"] == "needs_review"


def test_adjuster_can_replay_a_customers_key(
    client: TestClient, adjuster: Headers, cust1: Headers, policy1: str
) -> None:
    first = upload(client, cust1, policy1, key="k")
    second = upload(client, adjuster, policy1, key="k")
    assert second.status_code == 200
    assert second.json()["claim_id"] == first.json()["claim_id"]


def test_key_reused_for_another_policy_is_409(
    client: TestClient,
    sessions: SessionFactory,
    adjuster: Headers,
    policy1: str,
    policy2: str,
) -> None:
    assert upload(client, adjuster, policy1, key="k").status_code == 202
    assert upload(client, adjuster, policy2, key="k").status_code == 409
    assert count(sessions, Claim) == 1


def test_customer_cannot_reuse_another_customers_key(
    client: TestClient,
    sessions: SessionFactory,
    cust1: Headers,
    cust2: Headers,
    policy1: str,
    policy2: str,
) -> None:
    first = upload(client, cust1, policy1, key="k")
    response = upload(client, cust2, policy2, key="k")
    assert response.status_code == 409
    assert str(first.json()["claim_id"]) not in response.text
    assert count(sessions, Claim) == 1


def test_customer_cannot_replay_a_key_an_adjuster_used(
    client: TestClient, adjuster: Headers, cust1: Headers, policy1: str
) -> None:
    assert upload(client, adjuster, policy1, key="k").status_code == 202
    assert upload(client, cust1, policy1, key="k").status_code == 409


def test_lost_race_on_the_unique_key_resolves_to_the_winner(
    client: TestClient,
    sessions: SessionFactory,
    storage: Storage,
    cust1: Headers,
    policy1: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two requests pass the key lookup together; the second insert hits the unique index."""
    first = upload(client, cust1, policy1, key="race")
    files_after_first = stored_files(storage)

    real = claim_routes.find_by_key
    calls = iter([None])  # the first lookup (before the insert) sees nothing

    def stale(session: Session, key: str) -> Claim | None:
        for miss in calls:
            return miss
        return real(session, key)

    monkeypatch.setattr(claim_routes, "find_by_key", stale)
    second = upload(client, cust1, policy1, key="race", files=_files("loser.pdf"))

    assert second.status_code == 200
    assert second.json()["claim_id"] == first.json()["claim_id"]
    assert count(sessions, Claim) == 1
    assert count(sessions, Job) == 1
    assert stored_files(storage) == files_after_first  # the loser's files were removed


def test_lost_race_against_another_customers_claim_is_409(
    client: TestClient,
    cust1: Headers,
    cust2: Headers,
    policy1: str,
    policy2: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert upload(client, cust1, policy1, key="race").status_code == 202
    real = claim_routes.find_by_key
    calls = iter([None])

    def stale(session: Session, key: str) -> Claim | None:
        for miss in calls:
            return miss
        return real(session, key)

    monkeypatch.setattr(claim_routes, "find_by_key", stale)
    assert upload(client, cust2, policy2, key="race").status_code == 409


def test_missing_or_invalid_key_is_400(client: TestClient, cust1: Headers, policy1: str) -> None:
    assert upload(client, cust1, policy1, key=None).status_code == 400
    assert upload(client, cust1, policy1, key="").status_code == 400
    assert upload(client, cust1, policy1, key="k" * 129).status_code == 400
    assert upload(client, cust1, policy1, key="bad\tkey").status_code == 400
    assert upload(client, cust1, policy1, key="k" * 128).status_code == 202


def test_unknown_claim_type_is_422(client: TestClient, cust1: Headers, policy1: str) -> None:
    response = upload(client, cust1, policy1, data={"claim_type": "fire"})
    assert response.status_code == 422


@pytest.mark.parametrize("value", ["tomorrow", "2024-13-40", ""])
def test_unparseable_incident_date_is_422(
    client: TestClient, cust1: Headers, policy1: str, value: str
) -> None:
    assert upload(client, cust1, policy1, data={"incident_date": value}).status_code == 422


def test_future_incident_date_is_422(
    client: TestClient, sessions: SessionFactory, cust1: Headers, policy1: str
) -> None:
    tomorrow = (date.today() + timedelta(days=1)).isoformat()
    today = date.today().isoformat()
    assert upload(client, cust1, policy1, data={"incident_date": tomorrow}).status_code == 422
    assert count(sessions, Claim) == 0
    assert (
        upload(client, cust1, policy1, key="today", data={"incident_date": today}).status_code
        == 202
    )


def test_no_files_is_422(client: TestClient, cust1: Headers, policy1: str) -> None:
    assert upload(client, cust1, policy1, files=[]).status_code == 422


def test_files_field_may_be_named_with_brackets(
    client: TestClient, cust1: Headers, policy1: str
) -> None:
    files = [("files[]", ("form.pdf", PDF, "application/pdf"))]
    assert upload(client, cust1, policy1, files=files).status_code == 202


def test_more_than_ten_files_is_413(
    client: TestClient, sessions: SessionFactory, storage: Storage, cust1: Headers, policy1: str
) -> None:
    names = [f"doc{i}.pdf" for i in range(limits.MAX_FILES + 1)]
    assert upload(client, cust1, policy1, files=_files(*names)).status_code == 413
    assert count(sessions, Claim) == 0
    assert stored_files(storage) == []

    assert upload(client, cust1, policy1, key="ten", files=_files(*names[:10])).status_code == 202


def test_file_over_20_mb_is_413_and_leaves_nothing_behind(
    client: TestClient, sessions: SessionFactory, storage: Storage, cust1: Headers, policy1: str
) -> None:
    big = [("files", ("big.pdf", b"x" * (limits.MAX_FILE_BYTES + 1), "application/pdf"))]
    files = _files("small.pdf") + big
    assert upload(client, cust1, policy1, files=files).status_code == 413
    assert count(sessions, Claim) == 0
    assert count(sessions, ClaimDocument) == 0
    assert count(sessions, Job) == 0
    assert stored_files(storage) == []  # small.pdf was stored first, then removed


def test_file_of_exactly_20_mb_is_accepted(
    client: TestClient, cust1: Headers, policy1: str
) -> None:
    exact = [("files", ("exact.pdf", b"x" * limits.MAX_FILE_BYTES, "application/pdf"))]
    assert upload(client, cust1, policy1, files=exact).status_code == 202


def test_unknown_policy_is_404(client: TestClient, adjuster: Headers) -> None:
    assert upload(client, adjuster, "AUT-NOPE").status_code == 404


def test_customer_cannot_file_on_someone_elses_policy(
    client: TestClient,
    sessions: SessionFactory,
    storage: Storage,
    cust1: Headers,
    policy2: str,
) -> None:
    response = upload(client, cust1, policy2)
    assert response.status_code == 404
    assert response.json() == upload(client, cust1, "AUT-NOPE", key="k2").json()
    assert count(sessions, Claim) == 0
    assert stored_files(storage) == []


def test_adjuster_files_for_the_policy_holder(
    client: TestClient, sessions: SessionFactory, adjuster: Headers, cust1: Headers, policy1: str
) -> None:
    created = upload(client, adjuster, policy1)
    assert created.status_code == 202
    claim_id = created.json()["claim_id"]

    with sessions() as session:
        claim = session.get(Claim, uuid.UUID(claim_id))
        assert claim is not None
        assert claim.created_by == "cust-1"
        audit = session.scalars(select(AuditLog).where(AuditLog.action == "claim.created")).one()
        assert audit.actor == "adj-1"

    assert client.get(f"/claims/{claim_id}", headers=cust1).status_code == 200


def test_unauthenticated_upload_is_401(client: TestClient, policy1: str) -> None:
    assert upload(client, {}, policy1).status_code == 401


def test_stored_filename_is_a_plain_name(
    client: TestClient, sessions: SessionFactory, cust1: Headers, policy1: str
) -> None:
    files = _files("../../etc/passwd.pdf", "C:\\scans\\id.png")
    assert upload(client, cust1, policy1, files=files).status_code == 202
    with sessions() as session:
        names = sorted(session.scalars(select(ClaimDocument.filename)))
    assert names == ["id.png", "passwd.pdf"]


def test_duplicate_filenames_are_422(client: TestClient, cust1: Headers, policy1: str) -> None:
    files = [
        ("files", ("a.pdf", b"one", "application/pdf")),
        ("files", ("sub/a.pdf", b"two", "application/pdf")),
    ]
    assert upload(client, cust1, policy1, files=files).status_code == 422


def test_failed_transaction_removes_the_stored_files(
    client: TestClient,
    sessions: SessionFactory,
    storage: Storage,
    cust1: Headers,
    policy1: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom(session: Session, claim_id: uuid.UUID) -> Job:
        raise RuntimeError("queue down")

    monkeypatch.setattr(claim_routes, "enqueue", boom)
    with pytest.raises(RuntimeError, match="queue down"):
        upload(client, cust1, policy1)
    assert count(sessions, Claim) == 0
    assert count(sessions, ClaimDocument) == 0
    assert stored_files(storage) == []


# ---- reading ------------------------------------------------------------------------------


def _policy(sessions: SessionFactory, holder_id: str) -> uuid.UUID:
    with sessions() as session:
        policy = Policy(policy_number=f"AUT-{uuid.uuid4().hex[:6]}", holder_id=holder_id)
        session.add(policy)
        session.flush()
        return policy.id


def _claim(
    sessions: SessionFactory,
    holder_id: str,
    status: ClaimStatus = ClaimStatus.RECEIVED,
    **kwargs: Any,
) -> uuid.UUID:
    with sessions() as session:
        policy = Policy(policy_number=f"AUT-{uuid.uuid4().hex[:6]}", holder_id=holder_id)
        session.add(policy)
        session.flush()
        return add_claim(session, policy, status=status, **kwargs).id


def test_list_is_scoped_to_the_caller(
    client: TestClient, sessions: SessionFactory, adjuster: Headers, cust1: Headers
) -> None:
    mine = _claim(sessions, "cust-1")
    theirs = _claim(sessions, "cust-2")

    as_customer = client.get("/claims", headers=cust1)
    assert [c["claim_id"] for c in as_customer.json()] == [str(mine)]

    as_adjuster = client.get("/claims", headers=adjuster)
    assert {c["claim_id"] for c in as_adjuster.json()} == {str(mine), str(theirs)}


def test_list_item_shape(client: TestClient, sessions: SessionFactory, adjuster: Headers) -> None:
    _claim(
        sessions, "cust-1", packet_id="PKT-1", claim_type="glass", incident_date=date(2024, 3, 2)
    )
    [item] = client.get("/claims", headers=adjuster).json()
    assert set(item) == {
        "claim_id",
        "packet_id",
        "status",
        "claim_type",
        "incident_date",
        "created_at",
    }
    assert (item["packet_id"], item["status"], item["claim_type"]) == (
        "PKT-1",
        "received",
        "glass",
    )
    assert item["incident_date"] == "2024-03-02"


def test_list_filters_by_status_newest_first(
    client: TestClient, sessions: SessionFactory, adjuster: Headers
) -> None:
    older = _claim(sessions, "cust-1", ClaimStatus.NEEDS_REVIEW)
    _claim(sessions, "cust-1", ClaimStatus.RECEIVED)
    newer = _claim(sessions, "cust-1", ClaimStatus.NEEDS_REVIEW)

    response = client.get("/claims", params={"status": "needs_review"}, headers=adjuster)
    assert [c["claim_id"] for c in response.json()] == [str(newer), str(older)]
    assert client.get("/claims", params={"status": "bogus"}, headers=adjuster).status_code == 422


def test_list_is_capped_at_100(
    client: TestClient, sessions: SessionFactory, adjuster: Headers
) -> None:
    with sessions() as session:
        policy = Policy(policy_number="AUT-many", holder_id="cust-1")
        session.add(policy)
        session.flush()
        for _ in range(101):
            add_claim(session, policy)
    assert len(client.get("/claims", headers=adjuster).json()) == 100


def test_customer_gets_404_for_someone_elses_claim(
    client: TestClient, sessions: SessionFactory, cust1: Headers
) -> None:
    other = _claim(sessions, "cust-2")
    response = client.get(f"/claims/{other}", headers=cust1)
    assert response.status_code == 404
    assert response.json() == client.get(f"/claims/{uuid.uuid4()}", headers=cust1).json()


def test_get_claim_returns_status_and_documents(
    client: TestClient, adjuster: Headers, cust1: Headers, policy1: str
) -> None:
    claim_id = upload(client, cust1, policy1, key="PKT-5").json()["claim_id"]
    for headers in (cust1, adjuster):
        body = client.get(f"/claims/{claim_id}", headers=headers).json()
        assert body["claim_id"] == claim_id
        assert body["packet_id"] == "PKT-5"
        assert body["status"] == "received"
        assert body["policy_number"] == policy1
        assert [d["filename"] for d in body["documents"]] == ["form.pdf", "invoice.pdf"]
        assert "storage_key" not in body["documents"][0]


def test_get_unknown_claim(client: TestClient, adjuster: Headers) -> None:
    assert client.get(f"/claims/{uuid.uuid4()}", headers=adjuster).status_code == 404
    assert client.get("/claims/not-a-uuid", headers=adjuster).status_code == 422


def test_customer_cannot_read_the_review_file(
    client: TestClient, sessions: SessionFactory, cust1: Headers
) -> None:
    own = _claim(sessions, "cust-1")
    assert client.get(f"/claims/{own}/review", headers=cust1).status_code == 403


def test_adjuster_reads_the_review_file(
    client: TestClient, sessions: SessionFactory, adjuster: Headers
) -> None:
    claim_id = _claim(sessions, "cust-1", status=ClaimStatus.NEEDS_REVIEW, packet_id="PKT-9")
    with sessions() as session:
        session.add(ClaimDocument(claim_id=claim_id, filename="form.pdf", storage_key="k"))
        session.add(
            ValidationResult(
                claim_id=claim_id,
                rule_id="R1",
                outcome="missing",
                details={"message": "no ID card", "evidence": []},
            )
        )
        expected = build_review_file(session, claim_id)

    response = client.get(f"/claims/{claim_id}/review", headers=adjuster)

    assert response.status_code == 200
    body = response.json()
    assert body["claim"]["packet_id"] == "PKT-9"
    assert [d["filename"] for d in body["documents"]] == ["form.pdf"]
    assert body["rules"][0]["message"] == "no ID card"
    assert body["summary"]["blocking"] == ["R1"]
    assert body["summary"]["ready_for_decision"] is False
    assert ReviewFile.model_validate(body) == expected
    assert client.get(f"/claims/{uuid.uuid4()}/review", headers=adjuster).status_code == 404


def test_review_requires_a_token(client: TestClient) -> None:
    assert client.get(f"/claims/{uuid.uuid4()}/review").status_code == 401


# ---- decisions ----------------------------------------------------------------------------


def _decide(client: TestClient, headers: Headers, claim_id: uuid.UUID, **body: Any) -> Any:
    payload = {"decision": "approved_for_review", "reason": "documents match"} | body
    return client.post(f"/claims/{claim_id}/decision", headers=headers, json=payload)


def test_decision_from_received_is_409(
    client: TestClient, sessions: SessionFactory, adjuster: Headers
) -> None:
    claim_id = _claim(sessions, "cust-1", ClaimStatus.RECEIVED)
    assert _decide(client, adjuster, claim_id).status_code == 409
    with sessions() as session:
        claim = session.get(Claim, claim_id)
        assert claim is not None
        assert claim.status == "received"


def test_decision_from_needs_review_is_recorded_with_the_adjuster(
    client: TestClient, sessions: SessionFactory, adjuster: Headers
) -> None:
    claim_id = _claim(sessions, "cust-1", ClaimStatus.NEEDS_REVIEW)

    response = _decide(client, adjuster, claim_id, reason="all rules found")

    assert response.status_code == 200
    assert response.json() == {"claim_id": str(claim_id), "status": "approved_for_review"}
    with sessions() as session:
        claim = session.get(Claim, claim_id)
        assert claim is not None
        assert claim.status == "approved_for_review"
        audit = session.scalars(
            select(AuditLog).where(
                AuditLog.entity_id == str(claim_id), AuditLog.action == "claim.transition"
            )
        ).one()
        assert audit.actor == "adj-1"
        assert audit.payload == {
            "from": "needs_review",
            "to": "approved_for_review",
            "reason": "all rules found",
        }


def test_rejection_and_second_decision(
    client: TestClient, sessions: SessionFactory, adjuster: Headers
) -> None:
    claim_id = _claim(sessions, "cust-1", ClaimStatus.NEEDS_REVIEW)
    rejected = _decide(client, adjuster, claim_id, decision="rejected_incomplete", reason="no ID")
    assert rejected.status_code == 200
    assert rejected.json()["status"] == "rejected_incomplete"
    assert _decide(client, adjuster, claim_id).status_code == 409


def test_decision_validation(
    client: TestClient, sessions: SessionFactory, adjuster: Headers
) -> None:
    claim_id = _claim(sessions, "cust-1", ClaimStatus.NEEDS_REVIEW)
    assert _decide(client, adjuster, claim_id, decision="paid").status_code == 422
    assert _decide(client, adjuster, claim_id, decision="processing").status_code == 422
    assert _decide(client, adjuster, claim_id, reason="").status_code == 422
    no_reason = client.post(
        f"/claims/{claim_id}/decision", headers=adjuster, json={"decision": "approved_for_review"}
    )
    assert no_reason.status_code == 422
    assert _decide(client, adjuster, uuid.uuid4()).status_code == 404


def test_customer_cannot_decide(
    client: TestClient, sessions: SessionFactory, cust1: Headers
) -> None:
    claim_id = _claim(sessions, "cust-1", ClaimStatus.NEEDS_REVIEW)
    assert _decide(client, cust1, claim_id).status_code == 403
    with sessions() as session:
        claim = session.get(Claim, claim_id)
        assert claim is not None
        assert claim.status == "needs_review"


def test_decision_requires_a_token(client: TestClient) -> None:
    response = client.post(
        f"/claims/{uuid.uuid4()}/decision",
        json={"decision": "approved_for_review", "reason": "x"},
    )
    assert response.status_code == 401
