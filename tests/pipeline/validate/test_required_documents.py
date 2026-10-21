from collections.abc import Callable

import pytest

from core.types import ClaimType, DocType, Verdict
from pipeline.validate.base import ClaimContext, DocFields
from pipeline.validate.required_documents import RequiredDocuments
from tests.pipeline.validate.factories import passing_documents, without

Make = Callable[..., ClaimContext]


def _extra(doc_type: DocType) -> DocFields:
    return DocFields("extra", doc_type, {})


def test_collision_with_everything_found(make_ctx: Make) -> None:
    result = RequiredDocuments().evaluate(make_ctx())

    assert result.rule_id == "R6_REQUIRED_DOCUMENTS"
    assert result.verdict is Verdict.FOUND


def test_collision_without_incident_report_missing(make_ctx: Make) -> None:
    ctx = make_ctx(documents=without(passing_documents(), DocType.INCIDENT_REPORT))

    result = RequiredDocuments().evaluate(ctx)

    assert result.verdict is Verdict.MISSING
    assert "incident_report" in result.message
    assert "invoice" not in result.message


def test_theft_does_not_need_an_invoice(make_ctx: Make) -> None:
    ctx = make_ctx(
        claim_type=ClaimType.THEFT, documents=without(passing_documents(), DocType.INVOICE)
    )

    assert RequiredDocuments().evaluate(ctx).verdict is Verdict.FOUND


def test_glass_does_not_need_an_incident_report(make_ctx: Make) -> None:
    ctx = make_ctx(
        claim_type=ClaimType.GLASS,
        documents=without(passing_documents(), DocType.INCIDENT_REPORT),
    )

    assert RequiredDocuments().evaluate(ctx).verdict is Verdict.FOUND


def test_all_absent_types_are_listed(make_ctx: Make) -> None:
    ctx = make_ctx(documents=[])

    result = RequiredDocuments().evaluate(ctx)

    assert result.verdict is Verdict.MISSING
    for name in ("claim_form", "id_card", "invoice", "incident_report"):
        assert name in result.message


@pytest.mark.parametrize("claim_type", list(ClaimType))
def test_other_documents_never_count(make_ctx: Make, claim_type: ClaimType) -> None:
    ctx = make_ctx(claim_type=claim_type, documents=[_extra(DocType.OTHER)])

    assert RequiredDocuments().evaluate(ctx).verdict is Verdict.MISSING
