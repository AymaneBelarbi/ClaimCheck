from core.types import ClaimStatus, ClaimType, DocType, Evidence, Role, Verdict


def test_verdict_values() -> None:
    assert {v.value for v in Verdict} == {"found", "missing", "conflicting", "low_confidence"}


def test_doc_types() -> None:
    assert {d.value for d in DocType} == {
        "claim_form",
        "id_card",
        "invoice",
        "incident_report",
        "other",
    }


def test_claim_types() -> None:
    assert {c.value for c in ClaimType} == {"collision", "glass", "theft"}


def test_claim_statuses() -> None:
    assert {s.value for s in ClaimStatus} == {
        "received",
        "processing",
        "needs_review",
        "approved_for_review",
        "rejected_incomplete",
        "failed",
    }


def test_roles() -> None:
    assert {r.value for r in Role} == {"adjuster", "customer"}


def test_enums_compare_as_strings() -> None:
    assert str(Verdict.FOUND) == "found"
    assert f"{ClaimStatus.NEEDS_REVIEW}" == "needs_review"


def test_evidence_is_frozen_and_hashable() -> None:
    ev = Evidence(document_id="doc-1", page=2, span=(10, 24), text="Total 1,234.56")
    assert hash(ev) == hash(Evidence("doc-1", 2, (10, 24), "Total 1,234.56"))
    try:
        ev.page = 3  # type: ignore[misc]
    except AttributeError:
        pass
    else:
        raise AssertionError("Evidence must be immutable")
