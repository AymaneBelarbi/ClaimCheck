from collections.abc import Callable

from claimcheck.core.types import ClaimType, DocType, Verdict
from claimcheck.pipeline.validate.base import ClaimContext
from claimcheck.pipeline.validate.incident_date_consistent import IncidentDateConsistent
from tests.pipeline.validate.factories import INCIDENT, doc, passing_documents, without

Make = Callable[..., ClaimContext]
DAY = INCIDENT.isoformat()


def _evaluate(ctx: ClaimContext) -> tuple[Verdict, str, list[str]]:
    result = IncidentDateConsistent().evaluate(ctx)
    assert result.rule_id == "R7_INCIDENT_DATE_CONSISTENT"
    return result.verdict, result.message, [e.document_id for e in result.evidence]


def test_form_and_report_agree_with_the_claim(make_ctx: Make) -> None:
    verdict, _, evidence = _evaluate(make_ctx())

    assert verdict is Verdict.FOUND
    assert sorted(evidence) == ["form", "report"]


def test_without_an_incident_report_only_the_form_is_compared(make_ctx: Make) -> None:
    ctx = make_ctx(
        claim_type=ClaimType.GLASS,
        documents=without(passing_documents(), DocType.INCIDENT_REPORT),
    )

    verdict, _, evidence = _evaluate(ctx)

    assert verdict is Verdict.FOUND
    assert evidence == ["form"]


def test_a_form_dated_differently_conflicts(make_ctx: Make) -> None:
    documents = [
        *without(passing_documents(), DocType.CLAIM_FORM),
        doc(DocType.CLAIM_FORM, "form", claimant_name="Jose Alvarez", incident_date="2026-04-02"),
    ]

    verdict, message, evidence = _evaluate(make_ctx(documents=documents))

    assert verdict is Verdict.CONFLICTING
    assert "2026-04-02" in message and DAY in message
    assert sorted(evidence) == ["form", "report"]


def test_a_report_dated_differently_conflicts(make_ctx: Make) -> None:
    documents = [
        *without(passing_documents(), DocType.INCIDENT_REPORT),
        doc(DocType.INCIDENT_REPORT, "report", incident_date="2026-03-09"),
    ]

    verdict, message, _ = _evaluate(make_ctx(documents=documents))

    assert verdict is Verdict.CONFLICTING
    assert "incident_report" in message and "2026-03-09" in message


def test_no_claim_form_is_missing(make_ctx: Make) -> None:
    ctx = make_ctx(documents=without(passing_documents(), DocType.CLAIM_FORM))

    verdict, message, _ = _evaluate(ctx)

    assert verdict is Verdict.MISSING
    assert "claim_form.incident_date" in message


def test_a_form_without_the_date_is_missing(make_ctx: Make) -> None:
    documents = [
        *without(passing_documents(), DocType.CLAIM_FORM),
        doc(DocType.CLAIM_FORM, "form", incident_date=(None, Verdict.MISSING)),
    ]

    assert _evaluate(make_ctx(documents=documents))[0] is Verdict.MISSING


def test_a_low_confidence_date_is_low_confidence(make_ctx: Make) -> None:
    documents = [
        *without(passing_documents(), DocType.CLAIM_FORM),
        doc(DocType.CLAIM_FORM, "form", incident_date=(DAY, Verdict.LOW_CONFIDENCE)),
    ]

    assert _evaluate(make_ctx(documents=documents))[0] is Verdict.LOW_CONFIDENCE


def test_low_confidence_wins_over_a_disagreement(make_ctx: Make) -> None:
    documents = [
        doc(DocType.CLAIM_FORM, "form", incident_date=("2026-04-02", Verdict.LOW_CONFIDENCE)),
        doc(DocType.INCIDENT_REPORT, "report", incident_date="2026-03-01"),
    ]

    assert _evaluate(make_ctx(documents=documents))[0] is Verdict.LOW_CONFIDENCE


def test_an_unparseable_date_is_low_confidence(make_ctx: Make) -> None:
    documents = [
        *without(passing_documents(), DocType.CLAIM_FORM),
        doc(DocType.CLAIM_FORM, "form", incident_date="10th of March"),
    ]

    assert _evaluate(make_ctx(documents=documents))[0] is Verdict.LOW_CONFIDENCE


def test_applies_to_every_claim_type() -> None:
    assert IncidentDateConsistent().applies_to == frozenset(ClaimType)
