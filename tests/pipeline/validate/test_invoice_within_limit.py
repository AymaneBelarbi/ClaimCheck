from collections.abc import Callable
from decimal import Decimal

import pytest

from claimcheck.core.types import ClaimType, DocType, Verdict
from claimcheck.pipeline.validate.base import ClaimContext, DocFields
from claimcheck.pipeline.validate.invoice_within_limit import InvoiceWithinLimit
from tests.pipeline.validate.factories import doc, passing_documents, without

Make = Callable[..., ClaimContext]


def _invoice(total: str, verdict: Verdict = Verdict.FOUND, name: str = "inv") -> DocFields:
    return doc(DocType.INVOICE, name, total=(total, verdict))


@pytest.mark.parametrize(
    ("total", "verdict"), [("1500.00", Verdict.FOUND), ("1500.01", Verdict.CONFLICTING)]
)
def test_total_against_limit(make_ctx: Make, total: str, verdict: Verdict) -> None:
    ctx = make_ctx(
        claim_type=ClaimType.GLASS,
        documents=[_invoice(total)],
        coverage_limit=Decimal("1500"),
    )

    result = InvoiceWithinLimit().evaluate(ctx)

    assert result.rule_id == "R3_INVOICE_WITHIN_LIMIT"
    assert result.verdict is verdict
    assert "1,500" in result.message


def test_no_limit_is_missing(make_ctx: Make) -> None:
    ctx = make_ctx(documents=[_invoice("100.00")], coverage_limit=None)

    result = InvoiceWithinLimit().evaluate(ctx)

    assert result.verdict is Verdict.MISSING
    assert "limit" in result.message


def test_missing_invoice(make_ctx: Make) -> None:
    ctx = make_ctx(documents=without(passing_documents(), DocType.INVOICE))

    assert InvoiceWithinLimit().evaluate(ctx).verdict is Verdict.MISSING


def test_low_confidence_total(make_ctx: Make) -> None:
    ctx = make_ctx(documents=[_invoice("100.00", Verdict.LOW_CONFIDENCE)])

    assert InvoiceWithinLimit().evaluate(ctx).verdict is Verdict.LOW_CONFIDENCE


def test_two_invoices_with_different_totals_conflict(make_ctx: Make) -> None:
    ctx = make_ctx(documents=[_invoice("900.00", name="a"), _invoice("1900.00", name="b")])

    result = InvoiceWithinLimit().evaluate(ctx)

    assert result.verdict is Verdict.CONFLICTING
    assert len(result.evidence) == 2
    assert {e.document_id for e in result.evidence} == {"a", "b"}


def test_two_invoices_with_the_same_total_are_fine(make_ctx: Make) -> None:
    ctx = make_ctx(documents=[_invoice("900.00", name="a"), _invoice("900.00", name="b")])

    assert InvoiceWithinLimit().evaluate(ctx).verdict is Verdict.FOUND


def test_does_not_apply_to_theft() -> None:
    assert InvoiceWithinLimit().applies_to == {ClaimType.COLLISION, ClaimType.GLASS}


def test_invoice_without_the_field_is_missing(make_ctx: Make) -> None:
    ctx = make_ctx(documents=[_invoice("900.00", name="a"), doc(DocType.INVOICE, "b")])

    assert InvoiceWithinLimit().evaluate(ctx).verdict is Verdict.MISSING


def test_total_and_limit_are_printed_the_same_way(make_ctx: Make) -> None:
    documents = [
        *without(passing_documents(), DocType.INVOICE),
        doc(DocType.INVOICE, "inv", total="1873.72", invoice_date="2026-03-12"),
    ]

    result = InvoiceWithinLimit().evaluate(
        make_ctx(documents=documents, coverage_limit=Decimal("1500"))
    )

    assert result.message == "Invoice total 1,873.72 exceeds the limit of 1,500.00 by 373.72"
