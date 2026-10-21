import json
from pathlib import Path
from typing import Any

import pytest

from claimcheck.core.settings import get_settings
from claimcheck.core.types import DocType, Evidence, Verdict
from claimcheck.gateway.base import LLMGateway
from claimcheck.pipeline.extract import FieldResult
from claimcheck.pipeline.types import OcrPage, OcrResult
from evals import run as eval_run
from evals.context import EvalContext
from evals.dataset.types import GroundTruth, GTDocument, Scenario
from evals.runners import extraction
from evals.runners.extraction import prf1, values_match
from tests.fakes import FakeGateway, HashingEmbedder


def test_prf1() -> None:
    assert prf1(tp=9, fp=1, fn=1) == pytest.approx((0.9, 0.9, 0.9))


def test_prf1_with_nothing_scored_is_zero() -> None:
    assert prf1(tp=0, fp=0, fn=0) == (0.0, 0.0, 0.0)


def test_prf1_uneven() -> None:
    precision, recall, f1 = prf1(tp=8, fp=0, fn=2)
    assert (precision, recall) == (1.0, 0.8)
    assert f1 == pytest.approx(2 * 0.8 / 1.8)


@pytest.mark.parametrize(
    ("field", "gold", "predicted", "expected"),
    [
        ("vendor", "Garrett & Sons Panelworks", "GARRETT & SONS  PANELWORKS", True),
        ("description", "hit the\nfront left", "hit the front left", True),
        ("vendor", "Garrett & Sons", "Garrett and Sons", False),
        ("total", "8564.37", "8564.37", True),
        (
            "line_items",
            '[{"description": "Wheel alignment", "amount": "1693.45"}]',
            '[{"amount": "1693.45", "description": "wheel  alignment"}]',
            True,
        ),
        (
            "line_items",
            '[{"description": "Wheel alignment", "amount": "1693.45"}]',
            '[{"amount": "1693.40", "description": "Wheel alignment"}]',
            False,
        ),
        (
            "line_items",
            '[{"description": "A", "amount": "1.00"}, {"description": "B", "amount": "2.00"}]',
            '[{"amount": "2.00", "description": "B"}, {"amount": "1.00", "description": "A"}]',
            False,
        ),
        ("line_items", '[{"description": "A", "amount": "1.00"}]', "not json", False),
    ],
)
def test_values_match(field: str, gold: str, predicted: str, expected: bool) -> None:
    assert values_match(field, gold, predicted) is expected


def _gt(
    name: str, doc_type: DocType, fields: dict[str, str | None], verdict: Verdict
) -> GTDocument:
    return GTDocument(
        filename=name,
        doc_type=doc_type,
        fields=fields,
        field_verdicts={k: verdict for k in fields},
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


def _result(name: str, value: str | None, verdict: Verdict, confidence: float = 0.9) -> FieldResult:
    return FieldResult(
        field_name=name,
        value=value,
        confidence=confidence,
        verdict=verdict,
        evidence=Evidence("d", None, None, None),
        details=None,
    )


def test_runner_scores_fields_against_gold(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    docs = [
        _gt(
            "a.pdf",
            DocType.ID_CARD,
            {
                "full_name": "BECERRA DESIDERIO",
                "id_number": "D30736241",
                "date_of_birth": "1962-01-18",
                "expiry_date": None,
            },
            Verdict.FOUND,
        ),
        _gt(
            "b.pdf", DocType.INVOICE, {"vendor": "ClearView Glass", "total": "10.00"}, Verdict.FOUND
        ),
        # unreadable by construction: skipped and counted, never sent to the model
        _gt("c.png", DocType.INVOICE, {"vendor": "X", "total": "1.00"}, Verdict.LOW_CONFIDENCE),
    ]
    _write_packet(tmp_path, "PKT-001", docs)
    (tmp_path / "splits.json").write_text(json.dumps({"dev": ["PKT-001"], "holdout": []}))

    monkeypatch.setattr(
        extraction,
        "run_ocr",
        lambda data, filename: OcrResult(
            [OcrPage(1, filename)], [], filename, mean_confidence=0.9, low_confidence=False
        ),
    )
    predictions = {
        "a.pdf": [
            _result("full_name", "Becerra Desiderio", Verdict.FOUND),  # TP
            _result("id_number", "D30736249", Verdict.FOUND, 0.95),  # wrong: FP + FN
            # a low_confidence value is not a committed answer: FN, not FP
            _result("date_of_birth", "1962-01-18", Verdict.LOW_CONFIDENCE, 0.5),
            _result("expiry_date", "2030-03-06", Verdict.FOUND),  # gold null: FP
        ],
        "b.pdf": [
            _result("vendor", "ClearView Glass", Verdict.FOUND),  # TP
            _result("total", None, Verdict.MISSING),  # FN
        ],
    }
    seen: list[tuple[DocType, str | None]] = []

    def fake_extract(
        doc_type: DocType,
        ocr: OcrResult,
        document_id: str,
        llm: LLMGateway,
        *,
        cache_key: str | None = None,
    ) -> list[FieldResult]:
        seen.append((doc_type, cache_key))
        return predictions[ocr.text]

    monkeypatch.setattr(extraction, "extract", fake_extract)
    ctx = EvalContext(tmp_path, get_settings(), gateway=FakeGateway(), embedder=HashingEmbedder())

    result = extraction.run("dev", ctx)

    # tp=2, fp=2, fn=3
    assert result.metrics["extraction.precision"] == pytest.approx(0.5)
    assert result.metrics["extraction.recall"] == pytest.approx(0.4)
    assert result.metrics["extraction.f1"] == pytest.approx(2 * 0.5 * 0.4 / 0.9)
    assert seen == [
        (DocType.ID_CARD, "extract:PKT-001:a.pdf"),
        (DocType.INVOICE, "extract:PKT-001:b.pdf"),
    ]
    details: dict[str, Any] = result.details
    assert details["documents"] == {"scored": 2, "skipped_unreadable": 1}
    assert details["per_doc_type"]["invoice"]["tp"] == 1
    assert details["per_doc_type"]["invoice"]["fn"] == 1
    assert details["per_field"]["id_card.id_number"] == {
        "tp": 0,
        "fp": 1,
        "fn": 1,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
    }
    assert details["verdicts"] == {"found": 4, "low_confidence": 1, "missing": 1}
    misses = details["worst_misses"]
    assert len(misses) == 4
    # confidently wrong answers come first
    assert misses[0] == {
        "packet_id": "PKT-001",
        "filename": "a.pdf",
        "field": "id_number",
        "gold": "D30736241",
        "predicted": "D30736249",
        "verdict": "found",
        "confidence": 0.95,
        "details": None,
    }


def test_extraction_stage_is_registered() -> None:
    assert eval_run.STAGES["extraction"] is extraction.run
