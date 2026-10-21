from collections.abc import Callable

import pytest

from core.types import DocType, Verdict
from pipeline.validate.base import ClaimContext
from pipeline.validate.id_valid_on_incident import IdValidOnIncident
from tests.pipeline.validate.factories import doc, passing_documents, without

Make = Callable[..., ClaimContext]


@pytest.mark.parametrize(
    ("expiry", "verdict"),
    [
        ("2027-01-01", Verdict.FOUND),
        ("2026-03-10", Verdict.FOUND),
        ("2026-03-09", Verdict.CONFLICTING),
    ],
)
def test_expiry_against_incident(make_ctx: Make, expiry: str, verdict: Verdict) -> None:
    ctx = make_ctx(documents=[doc(DocType.ID_CARD, "id", expiry_date=expiry)])

    result = IdValidOnIncident().evaluate(ctx)

    assert result.rule_id == "R5_ID_VALID_ON_INCIDENT"
    assert result.verdict is verdict


def test_missing_id_card(make_ctx: Make) -> None:
    ctx = make_ctx(documents=without(passing_documents(), DocType.ID_CARD))

    assert IdValidOnIncident().evaluate(ctx).verdict is Verdict.MISSING


def test_missing_expiry_field(make_ctx: Make) -> None:
    ctx = make_ctx(documents=[doc(DocType.ID_CARD, "id", expiry_date=(None, Verdict.MISSING))])

    assert IdValidOnIncident().evaluate(ctx).verdict is Verdict.MISSING


def test_low_confidence_expiry(make_ctx: Make) -> None:
    ctx = make_ctx(
        documents=[doc(DocType.ID_CARD, "id", expiry_date=("2027-01-01", Verdict.LOW_CONFIDENCE))]
    )

    assert IdValidOnIncident().evaluate(ctx).verdict is Verdict.LOW_CONFIDENCE


def test_two_id_cards_with_different_expiry_conflict(make_ctx: Make) -> None:
    ctx = make_ctx(
        documents=[
            doc(DocType.ID_CARD, "a", expiry_date="2027-01-01"),
            doc(DocType.ID_CARD, "b", expiry_date="2028-01-01"),
        ]
    )

    assert IdValidOnIncident().evaluate(ctx).verdict is Verdict.CONFLICTING
