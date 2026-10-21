import json
from pathlib import Path

import pytest

from core.settings import get_settings
from core.types import DocType, Verdict
from evals.context import EvalContext
from evals.dataset.types import GroundTruth, GTDocument, Scenario
from evals.runners import classification
from gateway.base import LLMGateway
from pipeline.classify import ClassificationResult, DocTypeChoice, classify, llm_fallback
from pipeline.types import OcrPage, OcrResult
from rag.embedder import Embedder
from tests.fakes import FakeGateway, HashingEmbedder


def _doc(name: str, doc_type: DocType, verdict: Verdict) -> GTDocument:
    return GTDocument(
        filename=name,
        doc_type=doc_type,
        fields={"x": "y"},
        field_verdicts={"x": verdict},
    )


def _write_packet(root: Path, packet_id: str, docs: list[GTDocument]) -> None:
    folder = root / "packets" / packet_id
    folder.mkdir(parents=True)
    truth = GroundTruth(
        packet_id=packet_id,
        policy_number="AUT-000001",
        holder_id="H1",
        claim_type="collision",
        incident_date="2024-01-01",
        scenario=Scenario.CLEAN,
        variant=None,
        documents=docs,
        rule_verdicts={},
    )
    (folder / "ground_truth.json").write_text(truth.model_dump_json())
    for d in docs:
        (folder / d.filename).write_bytes(d.filename.encode())


def _ocr(name: bytes, filename: str, *, low: bool = False) -> OcrResult:
    text = filename
    return OcrResult(
        pages=[OcrPage(page_no=1, text=text)],
        words=[],
        text=text,
        mean_confidence=0.3 if low else 0.9,
        low_confidence=low,
    )


def test_runner_scores_readable_docs_and_flags_unreadable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    docs = [
        _doc("a.pdf", DocType.INVOICE, Verdict.FOUND),
        _doc("b.pdf", DocType.CLAIM_FORM, Verdict.FOUND),
        _doc("c.png", DocType.ID_CARD, Verdict.LOW_CONFIDENCE),
        _doc("d.png", DocType.INVOICE, Verdict.LOW_CONFIDENCE),
    ]
    _write_packet(tmp_path, "PKT-001", docs)
    (tmp_path / "splits.json").write_text(json.dumps({"dev": ["PKT-001"], "holdout": []}))

    monkeypatch.setattr(
        classification,
        "run_ocr",
        lambda data, filename: _ocr(
            data, filename, low=filename.endswith(".png") and filename != "d.png"
        ),
    )
    predictions = {
        "a.pdf": ClassificationResult(DocType.INVOICE, 0.9, "embedding"),
        "b.pdf": ClassificationResult(DocType.INCIDENT_REPORT, 0.7, "llm"),
        "d.png": ClassificationResult(DocType.INVOICE, 0.8, "embedding"),
        "c.png": ClassificationResult(DocType.OTHER, 0.3, "embedding"),
    }
    seen_keys: list[str | None] = []

    def fake_classify(
        ocr: OcrResult, embedder: Embedder, llm: LLMGateway | None, *, cache_key: str | None = None
    ) -> ClassificationResult:
        seen_keys.append(cache_key)
        return predictions[ocr.text]

    monkeypatch.setattr(classification, "classify", fake_classify)
    ctx = EvalContext(tmp_path, get_settings(), gateway=FakeGateway(), embedder=HashingEmbedder())

    result = classification.run("dev", ctx)

    assert result.metrics["classification.accuracy"] == 0.5
    # c.png was flagged low_confidence by OCR, d.png was not.
    assert result.metrics["classification.unreadable_flagged_rate"] == 0.5
    assert result.metrics["classification.llm_fallback_rate"] == 0.25
    assert result.details["per_doc_type"]["invoice"] == {"correct": 1, "total": 1, "accuracy": 1.0}
    assert result.details["per_doc_type"]["claim_form"]["accuracy"] == 0.0
    assert result.details["confusion_matrix"]["claim_form"] == {"incident_report": 1}
    assert result.details["documents"] == {"readable": 2, "unreadable": 2}
    assert "classify:PKT-001:a.pdf" in seen_keys


def _fallback_setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    docs = [
        _doc("a.pdf", DocType.INVOICE, Verdict.FOUND),
        _doc("b.pdf", DocType.CLAIM_FORM, Verdict.FOUND),
        _doc("c.png", DocType.ID_CARD, Verdict.LOW_CONFIDENCE),
    ]
    _write_packet(tmp_path, "PKT-001", docs)
    (tmp_path / "splits.json").write_text(json.dumps({"dev": ["PKT-001"], "holdout": []}))
    monkeypatch.setattr(
        classification,
        "run_ocr",
        lambda data, filename: _ocr(data, filename, low=filename == "c.png"),
    )
    predictions = {
        "a.pdf": ClassificationResult(DocType.INVOICE, 0.9, "embedding"),
        "b.pdf": ClassificationResult(DocType.CLAIM_FORM, 0.9, "embedding"),
        "c.png": ClassificationResult(DocType.OTHER, 0.3, "embedding"),
    }
    monkeypatch.setattr(
        classification,
        "classify",
        lambda ocr, embedder, llm, *, cache_key=None: predictions[ocr.text],
    )


def test_recording_every_fallback_asks_the_model_once_per_document(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fallback_setup(tmp_path, monkeypatch)
    plain_gateway = FakeGateway([DocTypeChoice(doc_type="invoice", confidence=0.9)])
    plain = classification.run(
        "dev",
        EvalContext(tmp_path, get_settings(), gateway=plain_gateway, embedder=HashingEmbedder()),
    )
    gateway = FakeGateway([DocTypeChoice(doc_type="invoice", confidence=0.9)])
    ctx = EvalContext(
        tmp_path,
        get_settings(),
        gateway=gateway,
        embedder=HashingEmbedder(),
        record_all_fallbacks=True,
    )

    result = classification.run("dev", ctx)

    # Every document, the unreadable one included: CI's OCR may not flag it the same way.
    assert [c.cache_key for c in gateway.calls] == [
        "classify:PKT-001:a.pdf",
        "classify:PKT-001:b.pdf",
        "classify:PKT-001:c.png",
    ]
    assert all(c.schema is DocTypeChoice for c in gateway.calls)
    assert plain_gateway.calls == []
    # The recorded answers are not used for the metric.
    assert result.metrics == plain.metrics


def test_the_fallback_prompt_is_the_one_classify_sends() -> None:
    ocr = _ocr(b"", "Repair invoice\nTotal 120.00")
    via_fallback = FakeGateway([DocTypeChoice(doc_type="invoice", confidence=0.8)])
    via_classify = FakeGateway([DocTypeChoice(doc_type="invoice", confidence=0.8)])

    result = llm_fallback(ocr, via_fallback, cache_key="classify:P:a.pdf")
    # A constant embedding puts every type at the same score, so classify falls back.
    classify(ocr, _ConstantEmbedder(), via_classify, cache_key="classify:P:a.pdf")

    assert result == ClassificationResult(DocType.INVOICE, 0.8, "llm")
    (sent,) = via_fallback.calls
    (expected,) = via_classify.calls
    assert (sent.system, sent.prompt, sent.cache_key) == (
        expected.system,
        expected.prompt,
        expected.cache_key,
    )


class _ConstantEmbedder:
    dim = 3
    name = "constant"

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0, 0.0] for _ in texts]
