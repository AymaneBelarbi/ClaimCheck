import json
import re
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from core.types import DocType, Verdict
from evals.dataset.types import FIELD_NAMES
from gateway.base import GatewayError, SchemaError
from gateway.replay import ReplayGateway
from pipeline.extract import MAX_CHARS, FieldResult, extract
from pipeline.schemas import (
    SCHEMAS,
    ClaimFormOut,
    IdCardOut,
    IncidentReportOut,
    InvoiceOut,
    LineItem,
    LLMField,
)
from pipeline.types import OcrPage, OcrResult, OcrWord
from tests.fakes import FakeCall, FakeGateway

CLAIM_TEXT = (
    "Northgate Mutual\n"
    "Motor claim form\n"
    "Claimant name Desiderio Becerra\n"
    "Policy number AUT-104233\n"
    "Vehicle registration FCV-6316\n"
    "Date of incident 14/03/2026\n"
    "Type of incident Collision\n"
    "What happened\n"
    "Another driver pulled out of a side road and hit the front left of my\n"
    "car. Nobody was hurt."
)


def _ocr(text: str, confidence: float = 0.95, low: dict[str, float] | None = None) -> OcrResult:
    """OCR result whose words are the whitespace-separated tokens of `text`."""
    low = low or {}
    words: list[OcrWord] = []
    for m in re.finditer(r"[^\s\f]+", text):
        page = text.count("\f", 0, m.start()) + 1
        conf = low.get(m.group(0), confidence)
        words.append(OcrWord(m.group(0), conf, page, m.start(), m.end()))
    pages = [OcrPage(i, t) for i, t in enumerate(text.split("\f"), start=1)]
    return OcrResult(
        pages=pages, words=words, text=text, mean_confidence=confidence, low_confidence=False
    )


def _f(value: Any, source_text: str | None = None, confidence: float = 0.95) -> dict[str, Any]:
    if source_text is None and isinstance(value, str):
        source_text = value
    return {"value": value, "confidence": confidence, "source_text": source_text}


def _claim(**overrides: dict[str, Any]) -> ClaimFormOut:
    fields: dict[str, dict[str, Any]] = {
        "claimant_name": _f("Desiderio Becerra"),
        "policy_number": _f("AUT-104233"),
        "incident_date": _f("14/03/2026"),
        "incident_type": _f("Collision"),
        "vehicle_plate": _f("FCV-6316"),
        "description": _f(
            "Another driver pulled out of a side road and hit the front left of my car. "
            "Nobody was hurt."
        ),
    }
    fields.update(overrides)
    return ClaimFormOut.model_validate(fields)


def _run(
    output: BaseModel | list[BaseModel | Exception],
    text: str = CLAIM_TEXT,
    doc_type: DocType = DocType.CLAIM_FORM,
    **ocr_kwargs: Any,
) -> tuple[dict[str, FieldResult], FakeGateway]:
    llm = FakeGateway(output if isinstance(output, list) else [output])
    results = extract(doc_type, _ocr(text, **ocr_kwargs), "doc-1", llm, cache_key="k")
    return {r.field_name: r for r in results}, llm


def test_found_field_has_evidence_span() -> None:
    results, _ = _run(_claim())

    name = results["claimant_name"]
    assert name.verdict is Verdict.FOUND
    assert name.value == "Desiderio Becerra"
    assert name.details is None
    assert name.evidence is not None
    assert name.evidence.document_id == "doc-1"
    assert name.evidence.page == 1
    assert name.evidence.span is not None
    start, end = name.evidence.span
    assert CLAIM_TEXT[start:end] == "Desiderio Becerra"
    assert name.evidence.text == "Desiderio Becerra"
    assert name.confidence == pytest.approx(0.95)


def test_confidence_is_min_of_llm_and_ocr_span() -> None:
    results, _ = _run(_claim(), low={"Becerra": 0.75})

    # words "Desiderio" 0.95 and "Becerra" 0.75 -> span mean 0.85 < llm 0.95
    assert results["claimant_name"].confidence == pytest.approx(0.85)
    assert results["claimant_name"].verdict is Verdict.FOUND


def test_poor_ocr_under_the_quote_is_low_confidence() -> None:
    results, _ = _run(_claim(), low={"Desiderio": 0.4, "Becerra": 0.5})

    name = results["claimant_name"]
    assert name.verdict is Verdict.LOW_CONFIDENCE
    assert name.confidence == pytest.approx(0.45)
    # the value is still shown to the reviewer, flagged
    assert name.value == "Desiderio Becerra"


def test_null_value_is_missing() -> None:
    results, _ = _run(_claim(vehicle_plate={"value": None, "confidence": 0.9, "source_text": None}))

    plate = results["vehicle_plate"]
    assert plate.verdict is Verdict.MISSING
    assert plate.value is None
    assert plate.evidence is not None
    assert plate.evidence.document_id == "doc-1"
    assert plate.evidence.span is None


@pytest.mark.parametrize("empty", ["", "   ", "null", "N/A"])
def test_placeholder_value_is_missing(empty: str) -> None:
    results, _ = _run(_claim(vehicle_plate=_f(empty, source_text=None)))

    assert results["vehicle_plate"].verdict is Verdict.MISSING
    assert results["vehicle_plate"].value is None


def test_low_llm_confidence_is_low_confidence() -> None:
    results, _ = _run(_claim(policy_number=_f("AUT-104233", confidence=0.6)))

    policy = results["policy_number"]
    assert policy.verdict is Verdict.LOW_CONFIDENCE
    assert policy.confidence == pytest.approx(0.6)
    assert policy.value == "AUT-104233"


def test_confidence_at_threshold_is_found() -> None:
    results, _ = _run(_claim(policy_number=_f("AUT-104233", confidence=0.70)))

    assert results["policy_number"].verdict is Verdict.FOUND


def test_quote_not_in_text_halves_confidence() -> None:
    results, _ = _run(_claim(policy_number=_f("AUT-999999", source_text="Policy AUT-999999")))

    policy = results["policy_number"]
    assert policy.confidence == pytest.approx(0.95 * 0.5)
    assert policy.verdict is Verdict.LOW_CONFIDENCE
    assert policy.details == "source_not_found"
    assert policy.evidence is not None
    assert policy.evidence.span is None
    assert policy.evidence.page is None
    assert policy.evidence.text == "Policy AUT-999999"


def test_value_without_quote_halves_confidence() -> None:
    results, _ = _run(
        _claim(policy_number={"value": "AUT-104233", "confidence": 0.9, "source_text": None})
    )

    policy = results["policy_number"]
    assert policy.confidence == pytest.approx(0.45)
    assert policy.verdict is Verdict.LOW_CONFIDENCE
    assert policy.evidence is not None
    assert policy.evidence.text is None


def test_quote_across_a_line_break_is_located() -> None:
    results, _ = _run(_claim())

    description = results["description"]
    assert description.verdict is Verdict.FOUND
    assert description.evidence is not None
    assert description.evidence.span is not None
    start, end = description.evidence.span
    assert CLAIM_TEXT[start:end].startswith("Another driver")
    assert CLAIM_TEXT[start:end].endswith("Nobody was hurt.")


def test_quote_with_an_ocr_slip_in_its_label_is_located_fuzzily() -> None:
    text = CLAIM_TEXT.replace("Claimant name", "Claimant narne")
    results, _ = _run(
        _claim(
            claimant_name=_f("Desiderio Becerra", source_text="Claimant name Desiderio Becerra")
        ),
        text=text,
    )

    name = results["claimant_name"]
    assert name.verdict is Verdict.FOUND
    assert name.evidence is not None
    assert name.evidence.span is not None
    start, end = name.evidence.span
    assert "Desiderio Becerra" in text[start:end]


def test_value_that_differs_from_the_document_is_not_found() -> None:
    # The fuzzy match finds the quote, but the value itself is not what the document says.
    text = CLAIM_TEXT.replace("Desiderio Becerra", "Desiderio Becerru")
    results, _ = _run(_claim(), text=text)

    name = results["claimant_name"]
    assert name.verdict is Verdict.LOW_CONFIDENCE
    assert name.details == "source_not_found"
    assert name.confidence == pytest.approx(0.95 * 0.5)


def test_value_absent_from_a_real_quote_is_not_found() -> None:
    # probe A: the quote is genuine, the value is invented
    results, _ = _run(
        _claim(policy_number=_f("AUT-999999", source_text="Policy number AUT-104233"))
    )

    policy = results["policy_number"]
    assert policy.verdict is Verdict.LOW_CONFIDENCE
    assert policy.details == "source_not_found"
    assert policy.confidence == pytest.approx(0.95 * 0.5)


def test_one_character_off_identifier_is_not_found() -> None:
    # probe B: value and quote agree with each other, and fuzzily with the document, but the
    # document prints AUT-104233.
    results, _ = _run(
        _claim(policy_number=_f("AUT-104238", source_text="Policy number AUT-104238"))
    )

    policy = results["policy_number"]
    assert policy.verdict is Verdict.LOW_CONFIDENCE
    assert policy.details == "source_not_found"


def test_date_quoted_by_its_label_alone_is_not_found() -> None:
    # probe C: the quote holds no date, so nothing in the document backs the value
    results, _ = _run(_claim(incident_date=_f("2026-03-15", source_text="Date of incident")))

    date = results["incident_date"]
    assert date.verdict is Verdict.LOW_CONFIDENCE
    assert date.details == "source_not_found"
    assert date.confidence == pytest.approx(0.95 * 0.5)


def test_date_quote_with_another_date_in_the_document_is_not_found() -> None:
    # The quote is located and shows 14/03/2026; the value says the 15th.
    results, _ = _run(
        _claim(incident_date=_f("2026-03-15", source_text="Date of incident 14/03/2026"))
    )

    date = results["incident_date"]
    assert date.verdict is Verdict.LOW_CONFIDENCE
    assert date.value is None


ID_TEXT = "DRIVER LICENCE\nDL NO.\nD30736241\nNAME\nBECERRA DESIDERIO"


def _id_card(**overrides: dict[str, Any]) -> IdCardOut:
    fields: dict[str, dict[str, Any]] = {
        "full_name": _f("BECERRA DESIDERIO"),
        "id_number": _f("D30736241", source_text="DL NO.\nD30736241"),
        "date_of_birth": {"value": None, "confidence": 0.9, "source_text": None},
        "expiry_date": {"value": None, "confidence": 0.9, "source_text": None},
    }
    fields.update(overrides)
    return IdCardOut.model_validate(fields)


def test_ocr_confidence_is_taken_under_the_value_not_its_label() -> None:
    # Tesseract scores small labels at 0 even when it reads them right.
    results, _ = _run(
        _id_card(), text=ID_TEXT, doc_type=DocType.ID_CARD, low={"DL": 0.0, "NO.": 0.0}
    )

    number = results["id_number"]
    assert number.verdict is Verdict.FOUND
    assert number.confidence == pytest.approx(0.95)
    assert number.evidence is not None
    assert number.evidence.text == "DL NO.\nD30736241"


def test_quote_stitched_from_label_and_value_falls_back_to_the_value() -> None:
    # Box layouts put labels on one line and values on the next.
    text = "POLICY NO. INCIDENT TYPE\nAUT-104233 COLLISION"
    results, _ = _run(
        _claim(policy_number=_f("AUT-104233", source_text="POLICY NO.\nAUT-104233")), text=text
    )

    policy = results["policy_number"]
    assert policy.verdict is Verdict.FOUND
    assert policy.confidence == pytest.approx(0.95)
    assert policy.details is None
    assert policy.evidence is not None
    assert policy.evidence.span is not None
    start, end = policy.evidence.span
    assert text[start:end] == "AUT-104233"


def test_value_contradicted_where_it_is_quoted_is_not_found_elsewhere() -> None:
    # The claimant box reads "Lourencgo"; the model wrote "Lourengo", which the OCR also
    # produced (wrongly) on the signature line. The cited place disagrees, so no support.
    text = (
        "POLICYHOLDER / CLAIMANT\nLuisa Lourencgo\nDATE OF LOSS\n30/03/2025\n"
        "I confirm the above is true. Signed Luisa Lourengo, 2 April 2025."
    )
    results, _ = _run(
        _claim(
            claimant_name=_f(
                "Luisa Lourengo", source_text="POLICYHOLDER / CLAIMANT\nLuisa Lourengo"
            )
        ),
        text=text,
    )

    name = results["claimant_name"]
    assert name.verdict is Verdict.LOW_CONFIDENCE
    assert name.details == "source_not_found"


def test_value_outside_its_quote_gets_no_fallback() -> None:
    # The value is in the document, but the quote the model gave does not contain it.
    results, _ = _run(_claim(policy_number=_f("AUT-104233", source_text="Policy holder AUT")))

    policy = results["policy_number"]
    assert policy.verdict is Verdict.LOW_CONFIDENCE
    assert policy.details == "source_not_found"


def test_page_comes_from_form_feeds() -> None:
    text = "Northgate Mutual\nMotor claim form\fPolicy number AUT-104233"
    results, _ = _run(_claim(), text=text)

    policy = results["policy_number"]
    assert policy.evidence is not None
    assert policy.evidence.page == 2
    assert policy.evidence.span is not None
    start, end = policy.evidence.span
    assert text[start:end] == "AUT-104233"


def test_ambiguous_date_is_low_confidence() -> None:
    text = CLAIM_TEXT.replace("14/03/2026", "03/04/2026")
    results, _ = _run(_claim(incident_date=_f("03/04/2026")), text=text)

    date = results["incident_date"]
    assert date.verdict is Verdict.LOW_CONFIDENCE
    assert date.value is None
    assert date.details == "ambiguous_date"


def test_month_first_numeric_date_is_refused() -> None:
    text = CLAIM_TEXT.replace("14/03/2026", "03/14/2026")
    results, _ = _run(_claim(incident_date=_f("03/14/2026")), text=text)

    assert results["incident_date"].value is None
    assert results["incident_date"].details == "ambiguous_date"


def test_iso_value_read_from_an_ambiguous_date_is_refused() -> None:
    # The model reformatted 03/04/2026 itself; picking day or month first is a guess.
    text = CLAIM_TEXT.replace("14/03/2026", "03/04/2026")
    results, _ = _run(_claim(incident_date=_f("2026-04-03", source_text="03/04/2026")), text=text)

    assert results["incident_date"].value is None
    assert results["incident_date"].verdict is Verdict.LOW_CONFIDENCE
    assert results["incident_date"].details == "ambiguous_date"


def test_iso_value_that_contradicts_its_quote_is_refused() -> None:
    text = CLAIM_TEXT.replace("14/03/2026", "8 February 2024")
    results, _ = _run(
        _claim(incident_date=_f("2024-08-02", source_text="8 February 2024")), text=text
    )

    assert results["incident_date"].value is None
    assert results["incident_date"].details == "date_mismatch"


def test_unparseable_date_is_low_confidence() -> None:
    text = CLAIM_TEXT.replace("14/03/2026", "next Tuesday")
    results, _ = _run(_claim(incident_date=_f("next Tuesday")), text=text)

    date = results["incident_date"]
    assert date.verdict is Verdict.LOW_CONFIDENCE
    assert date.value is None
    assert date.details == "unparseable_date"


def test_impossible_date_is_unparseable() -> None:
    text = CLAIM_TEXT.replace("14/03/2026", "31/02/2026")
    results, _ = _run(_claim(incident_date=_f("31/02/2026")), text=text)

    assert results["incident_date"].details == "unparseable_date"


def test_unambiguous_date_normalized() -> None:
    results, _ = _run(_claim())

    date = results["incident_date"]
    assert date.value == "2026-03-14"
    assert date.verdict is Verdict.FOUND


@pytest.mark.parametrize(
    "printed",
    ["2026-03-14", "14 March 2026", "March 14, 2026", "14 Mar 2026", "14-03-2026"],
)
def test_accepted_date_formats(printed: str) -> None:
    text = CLAIM_TEXT.replace("14/03/2026", printed)
    results, _ = _run(_claim(incident_date=_f(printed)), text=text)

    assert results["incident_date"].value == "2026-03-14"
    assert results["incident_date"].verdict is Verdict.FOUND


def test_iso_value_matching_a_long_date_quote_is_found() -> None:
    text = CLAIM_TEXT.replace("14/03/2026", "14 March 2026")
    results, _ = _run(
        _claim(incident_date=_f("2026-03-14", source_text="14 March 2026")), text=text
    )

    assert results["incident_date"].value == "2026-03-14"
    assert results["incident_date"].verdict is Verdict.FOUND


def test_incident_type_lowercased() -> None:
    results, _ = _run(_claim())

    assert results["incident_type"].value == "collision"
    assert results["incident_type"].verdict is Verdict.FOUND


def test_unknown_incident_type_is_low_confidence() -> None:
    text = CLAIM_TEXT.replace("Collision", "Vandalism")
    results, _ = _run(_claim(incident_type=_f("Vandalism")), text=text)

    kind = results["incident_type"]
    assert kind.verdict is Verdict.LOW_CONFIDENCE
    assert kind.value is None
    assert kind.details == "unknown_incident_type"


INVOICE_TEXT = (
    "GARRETT & SONS PANELWORKS\n"
    "Tax invoice\n"
    "Issued: 22 February 2024\n"
    "Wheel alignment\n"
    "Labour (12.0 hours)\n"
    "$1,693.45\n"
    "$1,136.65\n"
    "Amount due $2,830.10"
)


def _invoice(**overrides: dict[str, Any]) -> InvoiceOut:
    fields: dict[str, dict[str, Any]] = {
        "vendor": _f("GARRETT & SONS PANELWORKS"),
        "invoice_date": _f("22 February 2024"),
        "line_items": _f(
            [
                {"description": "Wheel alignment", "amount": "$1,693.45"},
                {"description": "Labour (12.0 hours)", "amount": "1136.65"},
            ],
            source_text="Wheel alignment",
        ),
        "total": _f("$2,830.10"),
    }
    fields.update(overrides)
    return InvoiceOut.model_validate(fields)


def test_amounts_and_line_items_normalized() -> None:
    results, _ = _run(_invoice(), text=INVOICE_TEXT, doc_type=DocType.INVOICE)

    assert results["total"].value == "2830.10"
    assert results["total"].verdict is Verdict.FOUND
    assert results["invoice_date"].value == "2024-02-22"
    items = results["line_items"]
    assert items.verdict is Verdict.FOUND
    assert items.value == json.dumps(
        [
            {"amount": "1693.45", "description": "Wheel alignment"},
            {"amount": "1136.65", "description": "Labour (12.0 hours)"},
        ],
        sort_keys=True,
    )


@pytest.mark.parametrize("printed", ["2.830,10", "12,50", "about $2,800", "2830.105"])
def test_unparseable_amount_is_low_confidence(printed: str) -> None:
    results, _ = _run(
        _invoice(total=_f(printed, source_text="Amount due $2,830.10")),
        text=INVOICE_TEXT,
        doc_type=DocType.INVOICE,
    )

    assert results["total"].verdict is Verdict.LOW_CONFIDENCE
    assert results["total"].value is None
    assert results["total"].details == "unparseable_amount"


def test_one_bad_line_amount_refuses_the_whole_list() -> None:
    items = [
        {"description": "Wheel alignment", "amount": "$1,693.45"},
        {"description": "Labour (12.0 hours)", "amount": "1.136,65"},
    ]
    results, _ = _run(
        _invoice(line_items=_f(items, source_text="Wheel alignment")),
        text=INVOICE_TEXT,
        doc_type=DocType.INVOICE,
    )

    assert results["line_items"].value is None
    assert results["line_items"].details == "unparseable_amount"


def test_empty_line_items_is_missing() -> None:
    results, _ = _run(
        _invoice(line_items={"value": [], "confidence": 0.9, "source_text": None}),
        text=INVOICE_TEXT,
        doc_type=DocType.INVOICE,
    )

    assert results["line_items"].verdict is Verdict.MISSING


def test_line_items_evidence_covers_every_row() -> None:
    # The quote is stitched from two columns; each row is checked on its own instead.
    items = [
        {"description": "Wheel alignment", "amount": "$1,693.45"},
        {"description": "Labour (12.0 hours)", "amount": "$1,136.65"},
    ]
    results, _ = _run(
        _invoice(line_items=_f(items, source_text="Wheel alignment\n$1,693.45")),
        text=INVOICE_TEXT,
        doc_type=DocType.INVOICE,
        low={"$1,136.65": 0.8},
    )

    line_items = results["line_items"]
    assert line_items.verdict is Verdict.FOUND
    assert line_items.evidence is not None
    assert line_items.evidence.text == (
        "Wheel alignment\nLabour (12.0 hours)\n$1,693.45\n$1,136.65"
    )
    # the weakest row decides
    assert line_items.confidence == pytest.approx(0.8)


def test_invented_line_item_is_not_found() -> None:
    items = [
        {"description": "Wheel alignment", "amount": "$1,693.45"},
        {"description": "Brake discs, replace", "amount": "$1,136.65"},
    ]
    results, _ = _run(
        _invoice(line_items=_f(items, source_text="Wheel alignment")),
        text=INVOICE_TEXT,
        doc_type=DocType.INVOICE,
    )

    line_items = results["line_items"]
    assert line_items.verdict is Verdict.LOW_CONFIDENCE
    assert line_items.details == "source_not_found"
    assert line_items.confidence == pytest.approx(0.95 * 0.5)


def test_line_item_with_an_invented_amount_is_not_found() -> None:
    items = [
        {"description": "Wheel alignment", "amount": "$1,693.45"},
        {"description": "Labour (12.0 hours)", "amount": "$1,136.95"},
    ]
    results, _ = _run(
        _invoice(line_items=_f(items, source_text="Wheel alignment")),
        text=INVOICE_TEXT,
        doc_type=DocType.INVOICE,
    )

    assert results["line_items"].details == "source_not_found"


def test_low_confidence_total_does_not_demote_line_items() -> None:
    # The items do not add up to this total, but the total is itself unconfirmed.
    items = [{"description": "Wheel alignment", "amount": "$1,693.45"}]
    results, _ = _run(
        _invoice(
            line_items=_f(items, source_text="Wheel alignment"),
            total=_f("$2,830.10", confidence=0.5),
        ),
        text=INVOICE_TEXT,
        doc_type=DocType.INVOICE,
    )

    assert results["total"].verdict is Verdict.LOW_CONFIDENCE
    assert results["line_items"].verdict is Verdict.FOUND


def test_line_items_that_do_not_add_up_to_the_total_are_low_confidence() -> None:
    # A table read with its amounts shifted by one row, the total taken as the last line.
    items = [
        {"description": "Wheel alignment", "amount": "$1,136.65"},
        {"description": "Labour (12.0 hours)", "amount": "$2,830.10"},
    ]
    results, _ = _run(
        _invoice(line_items=_f(items, source_text="Wheel alignment")),
        text=INVOICE_TEXT,
        doc_type=DocType.INVOICE,
    )

    line_items = results["line_items"]
    assert line_items.verdict is Verdict.LOW_CONFIDENCE
    assert line_items.details == "line_items_total_mismatch"
    # the total is read on its own and stays as it was
    assert results["total"].verdict is Verdict.FOUND


def test_line_items_are_not_checked_without_a_usable_total() -> None:
    items = [{"description": "Wheel alignment", "amount": "$1,693.45"}]
    results, _ = _run(
        _invoice(
            line_items=_f(items, source_text="Wheel alignment"),
            total={"value": None, "confidence": 0.9, "source_text": None},
        ),
        text=INVOICE_TEXT,
        doc_type=DocType.INVOICE,
    )

    assert results["line_items"].verdict is Verdict.FOUND


def test_schema_error_retries_once_then_marks_failed() -> None:
    error = SchemaError("bad json", raw="{")
    results, llm = _run([error, error])

    assert len(llm.calls) == 2
    assert list(results) == list(FIELD_NAMES[DocType.CLAIM_FORM])
    for result in results.values():
        assert result.verdict is Verdict.LOW_CONFIDENCE
        assert result.value is None
        assert result.details == "extraction_failed"
        assert result.confidence == 0.0
        assert result.evidence is not None
        assert result.evidence.document_id == "doc-1"


@pytest.mark.parametrize(
    "first", [GatewayError("timeout", provider="ollama"), SchemaError("bad json", raw="{")]
)
def test_retry_is_keyed_by_attempt(first: Exception) -> None:
    results, llm = _run([first, _claim()])

    assert results["claimant_name"].verdict is Verdict.FOUND
    # the failure is recorded under "k", so the retry needs a cassette of its own
    assert [c.cache_key for c in llm.calls] == ["k", "k:retry"]


def _timeout() -> GatewayError:
    return GatewayError("timeout", provider="ollama")


def _bad_json() -> SchemaError:
    return SchemaError("bad json", raw="{")


@pytest.mark.parametrize(
    "script",
    [
        [_timeout(), _bad_json()],
        [_bad_json(), _timeout()],
        [_timeout(), _timeout()],
        [_timeout(), _claim()],
        [_bad_json(), _claim()],
    ],
    ids=["timeout-then-schema", "schema-then-timeout", "timeout-twice", "timeout-ok", "schema-ok"],
)
def test_recorded_retries_replay_the_same_result(
    tmp_path: Path, script: list[BaseModel | Exception]
) -> None:
    inner = FakeGateway(script)
    recorded = extract(
        DocType.CLAIM_FORM,
        _ocr(CLAIM_TEXT),
        "doc-1",
        ReplayGateway(inner, tmp_path, record=True),
        cache_key="extract:PKT-1:doc_1.pdf",
    )
    replayed = extract(
        DocType.CLAIM_FORM,
        _ocr(CLAIM_TEXT),
        "doc-1",
        ReplayGateway(None, tmp_path, record=False),
        cache_key="extract:PKT-1:doc_1.pdf",
    )

    assert len(inner.calls) == 2
    assert replayed == recorded


def test_success_makes_one_call_with_the_cache_key() -> None:
    _, llm = _run(_claim())

    assert len(llm.calls) == 1
    call = llm.calls[0]
    assert call.cache_key == "k"
    assert call.schema is ClaimFormOut
    assert call.purpose == "extract"


def test_prompt_carries_doc_type_field_descriptions_and_text() -> None:
    _, llm = _run(_claim())

    call: FakeCall = llm.calls[0]
    assert call.prompt is not None
    assert "claim_form" in call.prompt
    assert CLAIM_TEXT in call.prompt
    for name, info in ClaimFormOut.model_fields.items():
        assert name in call.prompt
        assert info.description is not None
        assert info.description in call.prompt
    assert "null" in call.system
    assert "guess" in call.system.lower()


def test_prompt_text_is_capped() -> None:
    long_text = CLAIM_TEXT + "\n" + "x" * (MAX_CHARS * 2)
    _, llm = _run(_claim(), text=long_text)

    prompt = llm.calls[0].prompt
    assert prompt is not None
    assert "x" * (MAX_CHARS * 2) not in prompt
    assert len(prompt) < MAX_CHARS + 3000


def test_a_field_missing_from_a_truncated_document_is_low_confidence() -> None:
    """The model saw only the first MAX_CHARS characters: absence there is not absence."""
    long_text = CLAIM_TEXT + "\n" + "x" * (MAX_CHARS * 2)
    no_plate = {"value": None, "confidence": 0.9, "source_text": None}

    results, _ = _run(_claim(vehicle_plate=no_plate), text=long_text)

    plate = results["vehicle_plate"]
    assert plate.verdict is Verdict.LOW_CONFIDENCE
    assert plate.details == "truncated"
    assert plate.value is None
    # Fields the model did find are unaffected.
    assert results["claimant_name"].verdict is Verdict.FOUND


def test_a_field_missing_from_a_document_that_fits_is_still_missing() -> None:
    text = CLAIM_TEXT + "\n" + "x" * (MAX_CHARS - len(CLAIM_TEXT) - 1)
    assert len(text) == MAX_CHARS
    no_plate = {"value": None, "confidence": 0.9, "source_text": None}

    results, _ = _run(_claim(vehicle_plate=no_plate), text=text)

    assert results["vehicle_plate"].verdict is Verdict.MISSING


def test_other_doc_type_rejected() -> None:
    with pytest.raises(ValueError, match="other"):
        extract(DocType.OTHER, _ocr(CLAIM_TEXT), "doc-1", FakeGateway([_claim()]))


@pytest.mark.parametrize(
    ("doc_type", "model"),
    [
        (DocType.CLAIM_FORM, ClaimFormOut),
        (DocType.ID_CARD, IdCardOut),
        (DocType.INVOICE, InvoiceOut),
        (DocType.INCIDENT_REPORT, IncidentReportOut),
    ],
)
def test_schema_fields_match_the_dataset(doc_type: DocType, model: type[BaseModel]) -> None:
    assert SCHEMAS[doc_type] is model
    assert tuple(model.model_fields) == FIELD_NAMES[doc_type]
    for info in model.model_fields.values():
        assert info.description


def test_no_schema_for_other() -> None:
    assert DocType.OTHER not in SCHEMAS


def test_llm_field_parses_line_items() -> None:
    field = LLMField[list[LineItem]].model_validate(
        {"value": [{"description": "a", "amount": "1"}], "confidence": 1, "source_text": "a"}
    )
    assert field.value == [LineItem(description="a", amount="1")]
