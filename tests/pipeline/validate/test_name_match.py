from collections.abc import Callable

import pytest

from claimcheck.core.types import DocType, Verdict
from claimcheck.pipeline.validate.base import ClaimContext, DocFields
from claimcheck.pipeline.validate.name_match import NameMatch
from tests.pipeline.validate.factories import doc, passing_documents, without

Make = Callable[..., ClaimContext]


def _names(claimant: str, holder: str) -> list[DocFields]:
    return [
        doc(DocType.CLAIM_FORM, "form", claimant_name=claimant),
        doc(DocType.ID_CARD, "id", full_name=holder),
    ]


@pytest.mark.parametrize(
    ("claimant", "holder"),
    [
        ("José Álvarez", "ALVAREZ JOSE"),
        ("Jose Alvarez", "Jose Alvarez"),
        ("JOSE ALVAREZ", "jose alvarez"),
    ],
)
def test_same_person_found(make_ctx: Make, claimant: str, holder: str) -> None:
    result = NameMatch().evaluate(make_ctx(documents=_names(claimant, holder)))

    assert result.rule_id == "R1_NAME_MATCH"
    assert result.verdict is Verdict.FOUND


def test_different_names_conflicting_with_score_in_message(make_ctx: Make) -> None:
    result = NameMatch().evaluate(make_ctx(documents=_names("Joseph Alvares", "Jose Alvarez")))

    assert result.verdict is Verdict.CONFLICTING
    assert "Jose Alvarez" in result.message
    assert "Joseph Alvares" in result.message
    assert "< 90" in result.message
    assert {e.document_id for e in result.evidence} == {"form", "id"}


def test_missing_id_card(make_ctx: Make) -> None:
    ctx = make_ctx(documents=without(passing_documents(), DocType.ID_CARD))

    assert NameMatch().evaluate(ctx).verdict is Verdict.MISSING


def test_missing_name_value(make_ctx: Make) -> None:
    docs = [
        doc(DocType.CLAIM_FORM, "form", claimant_name=(None, Verdict.MISSING)),
        doc(DocType.ID_CARD, "id", full_name="Jose Alvarez"),
    ]

    assert NameMatch().evaluate(make_ctx(documents=docs)).verdict is Verdict.MISSING


def test_low_confidence_input(make_ctx: Make) -> None:
    docs = [
        doc(DocType.CLAIM_FORM, "form", claimant_name="Jose Alvarez"),
        doc(DocType.ID_CARD, "id", full_name=("Jose Alvarez", Verdict.LOW_CONFIDENCE)),
    ]

    result = NameMatch().evaluate(make_ctx(documents=docs))

    assert result.verdict is Verdict.LOW_CONFIDENCE
    assert [e.document_id for e in result.evidence if e.document_id == "id"]


def test_value_none_with_low_confidence_verdict_stays_low_confidence(make_ctx: Make) -> None:
    docs = [
        doc(DocType.CLAIM_FORM, "form", claimant_name="Jose Alvarez"),
        doc(DocType.ID_CARD, "id", full_name=(None, Verdict.LOW_CONFIDENCE)),
    ]

    assert NameMatch().evaluate(make_ctx(documents=docs)).verdict is Verdict.LOW_CONFIDENCE


def test_missing_beats_low_confidence(make_ctx: Make) -> None:
    docs = [doc(DocType.CLAIM_FORM, "form", claimant_name=("Jose Alvarez", Verdict.LOW_CONFIDENCE))]

    assert NameMatch().evaluate(make_ctx(documents=docs)).verdict is Verdict.MISSING


def test_two_id_cards_with_different_names_conflict(make_ctx: Make) -> None:
    docs = [
        doc(DocType.CLAIM_FORM, "form", claimant_name="Jose Alvarez"),
        doc(DocType.ID_CARD, "id-a", full_name="Jose Alvarez"),
        doc(DocType.ID_CARD, "id-b", full_name="Maria Lopez"),
    ]

    result = NameMatch().evaluate(make_ctx(documents=docs))

    assert result.verdict is Verdict.CONFLICTING
    assert {"id-a", "id-b"} <= {e.document_id for e in result.evidence}


def test_input_already_marked_conflicting(make_ctx: Make) -> None:
    docs = [
        doc(DocType.CLAIM_FORM, "form", claimant_name="Jose Alvarez"),
        doc(DocType.ID_CARD, "id", full_name=("Jose Alvarez", Verdict.CONFLICTING)),
    ]

    assert NameMatch().evaluate(make_ctx(documents=docs)).verdict is Verdict.CONFLICTING


def test_applies_to_every_claim_type() -> None:
    assert len(NameMatch().applies_to) == 3


def test_two_id_cards_spelling_one_name_differently_agree(make_ctx: Make) -> None:
    docs = [
        doc(DocType.CLAIM_FORM, "form", claimant_name="José Álvarez"),
        doc(DocType.ID_CARD, "id-a", full_name="José Álvarez"),
        doc(DocType.ID_CARD, "id-b", full_name="ALVAREZ JOSE"),
    ]

    result = NameMatch().evaluate(make_ctx(documents=docs))

    assert result.verdict is Verdict.FOUND


def test_two_claim_forms_spelling_one_name_differently_agree(make_ctx: Make) -> None:
    docs = [
        doc(DocType.CLAIM_FORM, "form-a", claimant_name="Jose Alvarez"),
        doc(DocType.CLAIM_FORM, "form-b", claimant_name="ALVAREZ JOSE"),
        doc(DocType.ID_CARD, "id", full_name="José Álvarez"),
    ]

    assert NameMatch().evaluate(make_ctx(documents=docs)).verdict is Verdict.FOUND
