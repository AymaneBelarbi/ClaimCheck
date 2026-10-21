from collections.abc import Callable

import pytest

from claimcheck.core.types import ClaimType, DocType, Verdict
from claimcheck.pipeline.validate.base import ClaimContext
from claimcheck.pipeline.validate.invoice_after_incident import InvoiceAfterIncident
from tests.pipeline.validate.factories import doc, passing_documents, without

Make = Callable[..., ClaimContext]


@pytest.mark.parametrize(
    ("invoice_date", "verdict"),
    [
        ("2026-03-11", Verdict.FOUND),
        ("2026-03-10", Verdict.FOUND),
        ("2026-03-09", Verdict.CONFLICTING),
    ],
)
def test_invoice_date_against_incident(make_ctx: Make, invoice_date: str, verdict: Verdict) -> None:
    ctx = make_ctx(documents=[doc(DocType.INVOICE, "inv", invoice_date=invoice_date)])

    result = InvoiceAfterIncident().evaluate(ctx)

    assert result.rule_id == "R4_INVOICE_AFTER_INCIDENT"
    assert result.verdict is verdict


def test_missing_invoice(make_ctx: Make) -> None:
    ctx = make_ctx(documents=without(passing_documents(), DocType.INVOICE))

    assert InvoiceAfterIncident().evaluate(ctx).verdict is Verdict.MISSING


def test_low_confidence_date(make_ctx: Make) -> None:
    ctx = make_ctx(
        documents=[doc(DocType.INVOICE, "inv", invoice_date=(None, Verdict.LOW_CONFIDENCE))]
    )

    assert InvoiceAfterIncident().evaluate(ctx).verdict is Verdict.LOW_CONFIDENCE


def test_two_invoices_with_different_dates_conflict(make_ctx: Make) -> None:
    ctx = make_ctx(
        documents=[
            doc(DocType.INVOICE, "a", invoice_date="2026-03-11"),
            doc(DocType.INVOICE, "b", invoice_date="2026-03-12"),
        ]
    )

    result = InvoiceAfterIncident().evaluate(ctx)

    assert result.verdict is Verdict.CONFLICTING
    assert len(result.evidence) == 2


def test_does_not_apply_to_theft() -> None:
    assert InvoiceAfterIncident().applies_to == {ClaimType.COLLISION, ClaimType.GLASS}
