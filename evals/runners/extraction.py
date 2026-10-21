"""Extraction stage: real OCR, then field extraction with the gold doc type, scored per field.

A prediction is the value of a field whose verdict is `found`. A `low_confidence` value goes to
a person, so it is not a committed answer: it costs recall when gold has a value but is never
counted as a false positive. Per (document, field):

- gold set, prediction equal to gold      -> TP
- gold set, prediction set but different  -> FP and FN
- gold set, no prediction                 -> FN
- gold empty, prediction set              -> FP

Documents whose ground truth is unreadable by construction are skipped, as in classification.
"""

import json
from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation
from typing import Any

from core.types import DocType, Verdict
from evals.context import EvalContext
from evals.dataset.types import GroundTruth, GTDocument
from evals.report import StageResult
from evals.runners.classification import gold_unreadable
from gateway.redaction import scope_gateway
from pipeline.extract import FieldResult, extract
from pipeline.ocr import run_ocr

WORST_MISSES = 20


def prf1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def values_match(field: str, gold: str, predicted: str) -> bool:
    """Compare normalized values: text casefolded with whitespace collapsed, line items as lists."""
    if field == "line_items":
        gold_items = _parse_line_items(gold)
        return gold_items is not None and gold_items == _parse_line_items(predicted)
    return _text_key(gold) == _text_key(predicted)


def _text_key(text: str) -> str:
    return " ".join(text.split()).casefold()


def _parse_line_items(text: str) -> list[tuple[str, Decimal]] | None:
    try:
        items = json.loads(text)
        return [
            (_text_key(str(item["description"])), Decimal(str(item["amount"])).quantize(_CENT))
            for item in items
        ]
    except (ValueError, TypeError, KeyError, InvalidOperation):
        return None


_CENT = Decimal("0.01")


class _Tally:
    def __init__(self) -> None:
        self.tp = self.fp = self.fn = 0

    def summary(self) -> dict[str, float]:
        precision, recall, f1 = prf1(self.tp, self.fp, self.fn)
        return {
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }


def run(split: str, ctx: EvalContext) -> StageResult:
    splits = json.loads((ctx.data_dir / "splits.json").read_text(encoding="utf-8"))
    overall = _Tally()
    per_type: dict[str, _Tally] = defaultdict(_Tally)
    per_field: dict[str, _Tally] = defaultdict(_Tally)
    verdicts: Counter[str] = Counter()
    misses: list[dict[str, Any]] = []
    scored = skipped = 0

    for packet_id in splits[split]:
        folder = ctx.data_dir / "packets" / packet_id
        truth = GroundTruth.model_validate_json((folder / "ground_truth.json").read_text())
        # The wall time of a packet is OCR plus every extraction call, not the scoring below.
        with ctx.ledger.packet(packet_id):
            # The packet's other documents tell a cloud backend's redactor which names to hide.
            ocr_by_file = {
                doc.filename: run_ocr((folder / doc.filename).read_bytes(), doc.filename)
                for doc in truth.documents
            }
            gateway = scope_gateway(ctx.gateway, (o.text for o in ocr_by_file.values()))
            extracted = {
                doc.filename: extract(
                    doc.doc_type,
                    ocr_by_file[doc.filename],
                    f"{packet_id}/{doc.filename}",
                    gateway,
                    cache_key=f"extract:{packet_id}:{doc.filename}",
                )
                for doc in truth.documents
                if doc.doc_type is not DocType.OTHER and not gold_unreadable(doc)
            }
        ctx.extracted[packet_id] = extracted
        for doc in truth.documents:
            if doc.doc_type is DocType.OTHER:
                continue
            if gold_unreadable(doc):
                skipped += 1
                continue
            results = extracted[doc.filename]
            scored += 1
            by_name = {r.field_name: r for r in results}
            for name, gold in doc.fields.items():
                result = by_name.get(name)
                verdicts[result.verdict.value if result else "absent"] += 1
                outcome = _score(name, gold, result)
                for tally in (overall, per_type[doc.doc_type.value], per_field[_key(doc, name)]):
                    tally.tp += outcome.count("tp")
                    tally.fp += outcome.count("fp")
                    tally.fn += outcome.count("fn")
                if outcome != ["tp"] and outcome:
                    misses.append(_miss(packet_id, doc, name, gold, result, outcome))

    precision, recall, f1 = prf1(overall.tp, overall.fp, overall.fn)
    metrics = {
        "extraction.precision": precision,
        "extraction.recall": recall,
        "extraction.f1": f1,
    }
    order = [t.value for t in DocType]
    details: dict[str, Any] = {
        "split": split,
        "documents": {"scored": scored, "skipped_unreadable": skipped},
        "counts": overall.summary(),
        "verdicts": dict(sorted(verdicts.items())),
        "per_doc_type": {t: per_type[t].summary() for t in order if t in per_type},
        "per_field": {k: per_field[k].summary() for k in sorted(per_field)},
        "worst_misses": _worst(misses),
    }
    return StageResult(metrics=metrics, details=details)


def _score(name: str, gold: str | None, result: FieldResult | None) -> list[str]:
    predicted = result.value if result and result.verdict is Verdict.FOUND else None
    if gold is None:
        return ["fp"] if predicted is not None else []
    if predicted is None:
        return ["fn"]
    if values_match(name, gold, predicted):
        return ["tp"]
    return ["fp", "fn"]


def _key(doc: GTDocument, name: str) -> str:
    return f"{doc.doc_type.value}.{name}"


def _miss(
    packet_id: str,
    doc: GTDocument,
    name: str,
    gold: str | None,
    result: FieldResult | None,
    outcome: list[str],
) -> dict[str, Any]:
    return {
        "packet_id": packet_id,
        "filename": doc.filename,
        "field": name,
        "gold": gold,
        "predicted": result.value if result else None,
        "verdict": result.verdict.value if result else None,
        "confidence": round(result.confidence, 4) if result else 0.0,
        "details": result.details if result else None,
        # dropped before the report; only used to rank
        "_rank": 0 if outcome == ["fp", "fn"] else 1 if outcome == ["fp"] else 2,
    }


def _worst(misses: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Confidently wrong answers first, then spurious values, then values not delivered."""
    ranked = sorted(misses, key=lambda m: (m["_rank"], -m["confidence"]))
    return [{k: v for k, v in m.items() if k != "_rank"} for m in ranked[:WORST_MISSES]]
