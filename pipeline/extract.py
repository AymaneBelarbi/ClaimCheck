"""Field extraction: one schema-constrained LLM call per document, then deterministic checks.

The model returns each field with a confidence and a quote. Everything after the call is
deterministic: the quote is located in the OCR text for page and span, the value must appear
inside it, the confidence is capped by the OCR confidence under the value, and dates and
amounts are normalized here rather than by the model. Anything that would need a guess (a date
that reads either way round, an amount with a decimal comma) comes back `low_confidence` with
no value.
"""

import json
import logging
import math
import re
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from enum import StrEnum
from functools import cache
from importlib.resources import files
from typing import Any

from pydantic import BaseModel
from rapidfuzz import fuzz

from core.types import ClaimType, DocType, Evidence, Verdict
from gateway.base import GatewayError, LLMGateway, SchemaError
from pipeline.ocr import span_confidence
from pipeline.schemas import SCHEMAS, LineItem, LLMField
from pipeline.types import OcrResult

logger = logging.getLogger(__name__)

# The prompt carries at most this much OCR text; see _truncated for what that means for a field.
MAX_CHARS = 8000
LOW_CONFIDENCE_THRESHOLD = 0.70
MATCH_THRESHOLD = 85.0
# Applied when the quote cannot be found in the OCR text, so nothing backs the value.
UNSUPPORTED_PENALTY = 0.5

# Strings a model writes when it means "not there".
_PLACEHOLDERS = frozenset({"", "null", "none", "n/a", "na", "unknown", "not stated", "-"})


class _Kind(StrEnum):
    TEXT = "text"
    DATE = "date"
    AMOUNT = "amount"
    LINE_ITEMS = "line_items"
    INCIDENT_TYPE = "incident_type"


_KINDS: dict[str, _Kind] = {
    "incident_date": _Kind.DATE,
    "date_of_birth": _Kind.DATE,
    "expiry_date": _Kind.DATE,
    "invoice_date": _Kind.DATE,
    "total": _Kind.AMOUNT,
    "line_items": _Kind.LINE_ITEMS,
    "incident_type": _Kind.INCIDENT_TYPE,
}


@dataclass(frozen=True)
class FieldResult:
    field_name: str
    value: str | None
    confidence: float
    verdict: Verdict
    evidence: Evidence | None
    # Why a field is not `found` when the verdict alone does not say, e.g. "ambiguous_date".
    details: str | None


class NormalizationError(ValueError):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def extract(
    doc_type: DocType,
    ocr: OcrResult,
    document_id: str,
    llm: LLMGateway,
    *,
    cache_key: str | None = None,
) -> list[FieldResult]:
    """Extract the fields of `doc_type` from one document's OCR text.

    Never raises for an LLM failure: after one retry every field comes back `low_confidence`
    with `details="extraction_failed"`.
    """
    schema = SCHEMAS.get(doc_type)
    if schema is None:
        raise ValueError(f"no extraction schema for doc type {doc_type.value!r}")

    output = _call(schema, _prompt(doc_type, schema, ocr.text), llm, cache_key)
    if output is None:
        return [
            FieldResult(
                name,
                None,
                0.0,
                Verdict.LOW_CONFIDENCE,
                Evidence(document_id, None, None, None),
                "extraction_failed",
            )
            for name in schema.model_fields
        ]
    results = [
        _field_result(name, getattr(output, name), ocr, document_id) for name in schema.model_fields
    ]
    if doc_type is DocType.INVOICE:
        results = _check_items_against_total(results)
    if len(ocr.text) > MAX_CHARS:
        results = [_truncated(r) for r in results]
    return results


def _truncated(result: FieldResult) -> FieldResult:
    """A field the model did not find in a document it only saw the start of.

    `missing` means the document does not show the field; past the cut nobody knows, so the
    field goes to a person instead.
    """
    if result.verdict is not Verdict.MISSING:
        return result
    return FieldResult(
        result.field_name,
        None,
        result.confidence,
        Verdict.LOW_CONFIDENCE,
        result.evidence,
        "truncated",
    )


def _call(
    schema: type[BaseModel], prompt: str, llm: LLMGateway, cache_key: str | None
) -> BaseModel | None:
    # Keyed by attempt, not by what went wrong: replay cassettes record failures under the
    # key they happened on, so the retry always needs a key of its own.
    keys = (cache_key, f"{cache_key}:retry" if cache_key else None)
    for attempt, key in enumerate(keys, start=1):
        try:
            result = llm.structured(
                system=_system_prompt(),
                prompt=prompt,
                schema=schema,
                purpose="extract",
                cache_key=key,
            )
        except (GatewayError, SchemaError) as exc:
            logger.warning("extraction attempt %d failed (%s)", attempt, exc)
            continue
        return result.output
    return None


@cache
def _system_prompt() -> str:
    return files("pipeline").joinpath("prompts", "extract_system.txt").read_text(encoding="utf-8")


def _prompt(doc_type: DocType, schema: type[BaseModel], text: str) -> str:
    fields = "\n".join(
        f"- {name}: {info.description}" for name, info in schema.model_fields.items()
    )
    return (
        f"Document type: {doc_type.value}\n\n"
        f"Fields to extract:\n{fields}\n\n"
        f"Document text:\n<<<\n{text[:MAX_CHARS]}\n>>>"
    )


def _field_result(name: str, field: LLMField[Any], ocr: OcrResult, document_id: str) -> FieldResult:
    llm_confidence = _clamp(field.confidence)
    if _is_empty(field.value):
        return FieldResult(
            name,
            None,
            llm_confidence,
            Verdict.MISSING,
            Evidence(document_id, None, None, None),
            None,
        )

    kind = _KINDS.get(name, _Kind.TEXT)
    quote = (field.source_text or "").strip() or None
    value: str | None
    reason: str | None = None
    try:
        value = _normalize(kind, field.value, quote)
    except NormalizationError as exc:
        value, reason = None, exc.reason

    if isinstance(field.value, list):
        support = _line_items_support(field.value, ocr.text)
    else:
        support = _value_support(kind, field.value, value, quote, ocr.text)
    if support is None:
        confidence = llm_confidence * UNSUPPORTED_PENALTY
        evidence = Evidence(document_id, None, None, quote)
    else:
        (start, end), scored = support
        confidence = min(llm_confidence, *(span_confidence(ocr, a, b) for a, b in scored))
        evidence = Evidence(
            document_id, page_of(ocr.text, start), (start, end), ocr.text[start:end]
        )

    if reason is not None:
        return FieldResult(name, None, confidence, Verdict.LOW_CONFIDENCE, evidence, reason)
    if support is None:
        return FieldResult(
            name, value, confidence, Verdict.LOW_CONFIDENCE, evidence, "source_not_found"
        )
    if confidence < LOW_CONFIDENCE_THRESHOLD:
        return FieldResult(name, value, confidence, Verdict.LOW_CONFIDENCE, evidence, None)
    return FieldResult(name, value, confidence, Verdict.FOUND, evidence, None)


def _clamp(confidence: float) -> float:
    if math.isnan(confidence):
        return 0.0
    return min(max(confidence, 0.0), 1.0)


def _is_empty(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().casefold() in _PLACEHOLDERS
    if isinstance(value, list):
        return not value
    return False


def _check_items_against_total(results: list[FieldResult]) -> list[FieldResult]:
    """Demote line items that do not add up to the invoice's confirmed total.

    Assumes an invoice whose total is the plain sum of its lines, with no separate tax,
    discount or deposit lines; that holds for the repair invoices this system reads. A table
    whose amounts were paired one row off, or with the total read as a last line, still
    parses, and only the sum gives it away. The check runs only against a `found` total, and
    a mismatch hands the items to a person as `low_confidence`: no value is ever rewritten.
    """
    by_name = {r.field_name: r for r in results}
    items, total = by_name.get("line_items"), by_name.get("total")
    if (
        items is None
        or total is None
        or items.verdict is not Verdict.FOUND
        or total.verdict is not Verdict.FOUND
        or items.value is None
        or total.value is None
    ):
        return results
    amounts = sum((Decimal(i["amount"]) for i in json.loads(items.value)), Decimal("0"))
    if amounts == Decimal(total.value):
        return results
    flagged = FieldResult(
        items.field_name,
        items.value,
        items.confidence,
        Verdict.LOW_CONFIDENCE,
        items.evidence,
        "line_items_total_mismatch",
    )
    return [flagged if r is items else r for r in results]


# --- evidence ----------------------------------------------------------------------------

_Span = tuple[int, int]
# (evidence span, spans whose OCR confidence counts)
_Support = tuple[_Span, list[_Span]]


def _value_support(
    kind: _Kind, raw: Any, normalized: str | None, quote: str | None, text: str
) -> _Support | None:
    """Where the document shows this value, or None if it cannot be shown.

    Fuzzy matching is only used to find the model's quote. The value itself must then appear
    inside the located quote exactly (ignoring case and line breaks; dates and amounts may be
    printed in any accepted format), and the OCR confidence is taken under the value alone:
    Tesseract often scores small printed labels near 0 even when it reads them right. When
    the whole quote is not in the text (models stitch a label to a value printed a line
    away), the value is looked up in the text, but only if the quote itself contains it.
    """
    if quote is None:
        return None

    def find(haystack: str) -> _Span | None:
        if kind is _Kind.DATE and normalized is not None:
            return _find_date(normalized, haystack)
        if kind is _Kind.AMOUNT:
            return _find_amount(str(raw), haystack)
        return find_exact(str(raw), haystack)

    span = locate(quote, text)
    if span is not None:
        start, end = span
        inner = find(text[start:end])
        if inner is None:
            # The document disagrees with the model at the place the model cited. The same
            # string printed elsewhere (an OCR misread of the signature line, say) is no
            # support for it.
            return None
        return span, [(start + inner[0], start + inner[1])]
    if find(quote) is not None:
        found = find(text)
        if found is not None:
            return found, [found]
    return None


def _line_items_support(items: list[LineItem], text: str) -> _Support | None:
    """Every row's description and amount must be printed in the document.

    Tables often come out of OCR as a column of descriptions followed by a column of amounts,
    so rows are checked one by one rather than through a single quote. The evidence runs from
    the first located cell to the last; the weakest cell sets the OCR confidence.
    """
    spans: list[_Span] = []
    for item in items:
        description = find_exact(item.description, text)
        amount = _find_amount(item.amount, text)
        if description is None or amount is None:
            return None
        spans += [description, amount]
    return (min(a for a, _ in spans), max(b for _, b in spans)), spans


def _find_date(iso: str, haystack: str) -> _Span | None:
    for match in _DATE_SEARCH.finditer(haystack):
        try:
            if _date_from_match(match) == iso:
                return match.span()
        except NormalizationError:
            continue
    return None


_MONEY_SEARCH = re.compile(r"(?<![\w.,])(?:[$£€]\s?)?\d[\d,]*(?:\.\d+)?(?!\w)")


def _find_amount(raw: str, haystack: str) -> _Span | None:
    """An amount printed with the same value, whatever its symbol or thousands separators."""
    try:
        target = normalize_amount(raw)
    except NormalizationError:
        return find_exact(raw, haystack)
    for match in _MONEY_SEARCH.finditer(haystack):
        try:
            if normalize_amount(match.group(0)) == target:
                return match.span()
        except NormalizationError:
            continue
    return None


def find_exact(needle: str, text: str) -> _Span | None:
    """Span of `needle` in `text`, exactly or ignoring case and how whitespace is broken."""
    needle = needle.strip()
    tokens = needle.split()
    if not tokens:
        return None
    index = text.find(needle)
    if index >= 0:
        return index, index + len(needle)
    match = re.search(r"\s+".join(re.escape(t) for t in tokens), text, re.IGNORECASE)
    return match.span() if match else None


def locate(quote: str, text: str) -> _Span | None:
    """Span of the model's quote in `text`: exact, then ignoring case and whitespace, then fuzzy.

    The middle step matters because OCR breaks lines where the model's quote has a space; the
    fuzzy step tolerates OCR slips in the quote's label. Only quotes are matched fuzzily,
    never values.
    """
    exact = find_exact(quote, text)
    if exact is not None:
        return exact
    if not quote.strip() or len(quote) > len(text):
        return None
    # Lower-casing keeps offsets valid only if it keeps lengths, which holds for almost all text.
    haystack, needle = text.lower(), quote.lower()
    if len(haystack) != len(text) or len(needle) != len(quote):
        haystack, needle = text, quote
    alignment = fuzz.partial_ratio_alignment(needle, haystack, score_cutoff=MATCH_THRESHOLD)
    if alignment is None or alignment.dest_end <= alignment.dest_start:
        return None
    # The alignment window has the quote's length, so a slip that changes the length cuts a
    # word short; quotes are whole words, so widen to word boundaries.
    start, end = alignment.dest_start, alignment.dest_end
    while start > 0 and not text[start - 1].isspace():
        start -= 1
    while end < len(text) and not text[end].isspace():
        end += 1
    return start, end


def page_of(text: str, offset: int) -> int:
    """1-based page of a character offset; pages are joined with form feeds."""
    return text.count("\f", 0, offset) + 1


# --- normalization -----------------------------------------------------------------------


def _normalize(kind: _Kind, value: Any, quote: str | None) -> str:
    if kind is _Kind.LINE_ITEMS:
        return _normalize_line_items(value)
    text = str(value).strip()
    if kind is _Kind.DATE:
        iso = normalize_date(text)
        if quote:
            _check_date_against_quote(iso, quote)
        return iso
    if kind is _Kind.AMOUNT:
        return normalize_amount(text)
    if kind is _Kind.INCIDENT_TYPE:
        lowered = text.casefold()
        if lowered not in {t.value for t in ClaimType}:
            raise NormalizationError("unknown_incident_type")
        return lowered
    return text


_MONTHS: dict[str, int] = {}
for _number, _name in enumerate(
    (
        "january",
        "february",
        "march",
        "april",
        "may",
        "june",
        "july",
        "august",
        "september",
        "october",
        "november",
        "december",
    ),
    start=1,
):
    _MONTHS[_name] = _number
    _MONTHS[_name[:3]] = _number
_MONTHS["sept"] = 9

_ORDINAL = r"(?:st|nd|rd|th)?"
_ISO = r"(?P<iso_y>\d{4})-(?P<iso_m>\d{2})-(?P<iso_d>\d{2})"
_DAY_MONTH = rf"(?P<dm_d>\d{{1,2}}){_ORDINAL}\s+(?P<dm_m>[A-Za-z]+)\.?,?\s+(?P<dm_y>\d{{4}})"
_MONTH_DAY = rf"(?P<md_m>[A-Za-z]+)\.?\s+(?P<md_d>\d{{1,2}}){_ORDINAL},?\s+(?P<md_y>\d{{4}})"
_NUMERIC = r"(?P<n_a>\d{1,2})(?P<sep>[/-])(?P<n_b>\d{1,2})(?P=sep)(?P<n_y>\d{4})"
_DATE = "|".join((_ISO, _DAY_MONTH, _MONTH_DAY, _NUMERIC))
_DATE_FULL = re.compile(rf"^(?:{_DATE})$")
_DATE_SEARCH = re.compile(rf"(?<![\w/-])(?:{_DATE})(?![\w/-])")


def normalize_date(text: str) -> str:
    """ISO date for the formats that cannot be misread, else NormalizationError.

    Accepted: `2026-03-14`, `14 March 2026`, `March 14, 2026`, `14 Mar 2026`, and
    `14/03/2026` or `14-03-2026` only when the first number is above 12. A numeric date whose
    first number could be a month is refused as ambiguous; everything else is unparseable.
    """
    match = _DATE_FULL.match(" ".join(text.split()))
    if match is None:
        raise NormalizationError("unparseable_date")
    return _date_from_match(match)


def _date_from_match(match: re.Match[str]) -> str:
    g = match.groupdict()
    if g["iso_y"]:
        return _make_date(int(g["iso_y"]), int(g["iso_m"]), int(g["iso_d"]))
    if g["dm_y"]:
        return _make_date(int(g["dm_y"]), _month(g["dm_m"]), int(g["dm_d"]))
    if g["md_y"]:
        return _make_date(int(g["md_y"]), _month(g["md_m"]), int(g["md_d"]))
    first, second = int(g["n_a"]), int(g["n_b"])
    if first <= 12:
        # 03/04 reads either way, and 03/14 implies a month-first convention the rest of the
        # dataset does not use; neither is ours to decide.
        raise NormalizationError("ambiguous_date")
    return _make_date(int(g["n_y"]), second, first)


def _month(name: str) -> int:
    number = _MONTHS.get(name.casefold())
    if number is None:
        raise NormalizationError("unparseable_date")
    return number


def _make_date(year: int, month: int, day: int) -> str:
    try:
        return date(year, month, day).isoformat()
    except ValueError as exc:
        raise NormalizationError("unparseable_date") from exc


def _check_date_against_quote(iso: str, quote: str) -> None:
    """Refuse a date the quote does not support.

    The model may hand back ISO even when told not to. If its quote shows `03/04/2026`, the
    ISO value is the model's guess at the order; if the quote shows a different date, the
    value is not what the document says.
    """
    seen: list[str] = []
    ambiguous = False
    for match in _DATE_SEARCH.finditer(quote):
        try:
            seen.append(_date_from_match(match))
        except NormalizationError as exc:
            ambiguous = ambiguous or exc.reason == "ambiguous_date"
    if iso in seen:
        return
    if ambiguous:
        raise NormalizationError("ambiguous_date")
    if seen:
        raise NormalizationError("date_mismatch")


_CURRENCY = re.compile(r"^(?:[$£€]|USD|EUR|GBP)\s*|\s*(?:USD|EUR|GBP)$", re.IGNORECASE)
# Comma only as a thousands separator, dot only as the decimal point: "1.234,50" and "12,50"
# are refused rather than read one way or the other.
_AMOUNT = re.compile(r"^(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d{1,2})?$")


def normalize_amount(text: str) -> str:
    """Two-decimal string without symbol or separators, e.g. "$1,234.5" -> "1234.50"."""
    cleaned = _CURRENCY.sub("", text.strip()).strip()
    if not _AMOUNT.match(cleaned):
        raise NormalizationError("unparseable_amount")
    return str(Decimal(cleaned.replace(",", "")).quantize(Decimal("0.01")))


def canonical_line_items(items: list[tuple[str, str]]) -> str:
    """The one stored form of a line-item list: JSON, sorted keys, json.dumps separators."""
    return json.dumps(
        [{"description": description, "amount": amount} for description, amount in items],
        sort_keys=True,
    )


def _normalize_line_items(value: list[LineItem]) -> str:
    return canonical_line_items(
        [(item.description.strip(), normalize_amount(item.amount)) for item in value]
    )
