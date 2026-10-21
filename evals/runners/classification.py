"""Classification stage: real OCR, then the classifier, over every document in a split.

With `ctx.record_all_fallbacks` (cassette recording only) the model fallback is also asked for
every document before it is classified, under the cache key `classify` would use. The answer is
not used, so the metric is the same as without it; it exists so a replay whose OCR text differs
(another Tesseract build in CI) finds a cassette when the embedding margin falls the other way.
"""

import json
from collections import Counter, defaultdict
from typing import Any

from core.types import DocType, Verdict
from evals.context import EvalContext
from evals.dataset.types import GroundTruth, GTDocument
from evals.report import StageResult
from gateway.redaction import scope_gateway
from pipeline.classify import classify, llm_fallback
from pipeline.ocr import run_ocr


def run(split: str, ctx: EvalContext) -> StageResult:
    splits = json.loads((ctx.data_dir / "splits.json").read_text(encoding="utf-8"))
    correct: Counter[str] = Counter()
    total: Counter[str] = Counter()
    confusion: dict[str, Counter[str]] = defaultdict(Counter)
    errors: list[dict[str, Any]] = []
    unreadable = flagged = fallbacks = documents = 0

    for packet_id in splits[split]:
        folder = ctx.data_dir / "packets" / packet_id
        truth = GroundTruth.model_validate_json((folder / "ground_truth.json").read_text())
        ocr_by_file = {
            doc.filename: run_ocr((folder / doc.filename).read_bytes(), doc.filename)
            for doc in truth.documents
        }
        gateway = scope_gateway(ctx.gateway, (o.text for o in ocr_by_file.values()))
        for doc in truth.documents:
            ocr = ocr_by_file[doc.filename]
            key = f"classify:{packet_id}:{doc.filename}"
            if ctx.record_all_fallbacks:
                # Outside the ledger: these calls are not part of classifying the packet.
                llm_fallback(ocr, gateway, cache_key=key)
            # Only the model calls are counted here; the packet's wall time is measured by the
            # extraction stage, which also reads the documents.
            with ctx.ledger.packet(packet_id, timed=False):
                result = classify(
                    ocr,
                    ctx.embedder,
                    gateway,
                    cache_key=key,
                )
            documents += 1
            fallbacks += result.method == "llm"
            if gold_unreadable(doc):
                # Nobody can be expected to classify these; the question is whether OCR
                # flagged them for review.
                unreadable += 1
                flagged += ocr.low_confidence
                continue
            gold = doc.doc_type.value
            total[gold] += 1
            confusion[gold][result.doc_type.value] += 1
            if result.doc_type is doc.doc_type:
                correct[gold] += 1
            else:
                errors.append(
                    {
                        "packet_id": packet_id,
                        "filename": doc.filename,
                        "gold": gold,
                        "predicted": result.doc_type.value,
                        "method": result.method,
                        "confidence": round(result.confidence, 4),
                        "ocr_low_confidence": ocr.low_confidence,
                    }
                )

    readable = sum(total.values())
    metrics: dict[str, float] = {
        "classification.accuracy": sum(correct.values()) / readable if readable else 0.0,
        "classification.llm_fallback_rate": fallbacks / documents if documents else 0.0,
    }
    if unreadable:
        metrics["classification.unreadable_flagged_rate"] = flagged / unreadable

    order = [t.value for t in DocType]
    details: dict[str, Any] = {
        "split": split,
        "documents": {"readable": readable, "unreadable": unreadable},
        "per_doc_type": {
            t: {"correct": correct[t], "total": total[t], "accuracy": correct[t] / total[t]}
            for t in order
            if total[t]
        },
        "confusion_matrix": {g: dict(confusion[g]) for g in order if g in confusion},
        "misclassified": errors,
    }
    return StageResult(metrics=metrics, details=details)


def gold_unreadable(doc: GTDocument) -> bool:
    """All gold fields are low_confidence: the scan was built to be unreadable."""
    verdicts = doc.field_verdicts.values()
    return bool(doc.field_verdicts) and all(v is Verdict.LOW_CONFIDENCE for v in verdicts)
