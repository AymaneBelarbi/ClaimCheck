import json
from datetime import date
from pathlib import Path

from core.settings import get_settings
from core.types import ClaimType, DocType, Evidence, Verdict
from evals.context import EvalContext
from evals.dataset.types import RULE_IDS, GroundTruth, GTDocument, Scenario
from evals.report import StageResult
from evals.runners import validation
from pipeline.extract import FieldResult

POLICY_MD = """# Policy AUT-000001, version 1

## 4.1 Collision damage

We pay up to a limit of $15,000 per claim.
"""


def _doc(name: str, doc_type: DocType, **fields: str) -> GTDocument:
    return GTDocument(
        filename=name,
        doc_type=doc_type,
        fields=dict(fields),
        field_verdicts=dict.fromkeys(fields, Verdict.FOUND),
    )


def _write(root: Path, rule_verdicts: dict[str, Verdict], incident: str = "2025-06-01") -> None:
    (root / "policies" / "AUT-000001").mkdir(parents=True)
    (root / "policies" / "AUT-000001" / "v1.md").write_text(POLICY_MD)
    index = {
        "policies": [
            {
                "policy_number": "AUT-000001",
                "holder_id": "cust-1",
                "versions": [
                    {
                        "version_no": 1,
                        "effective_from": "2025-01-01",
                        "effective_to": "2026-01-01",
                        "file": "AUT-000001/v1.md",
                    }
                ],
            }
        ]
    }
    (root / "policies" / "index.json").write_text(json.dumps(index))
    (root / "splits.json").write_text(json.dumps({"dev": ["p1"], "holdout": []}))
    folder = root / "packets" / "p1"
    folder.mkdir(parents=True)
    truth = GroundTruth(
        packet_id="p1",
        policy_number="AUT-000001",
        holder_id="cust-1",
        claim_type=ClaimType.COLLISION,
        incident_date=date.fromisoformat(incident),
        scenario=Scenario.CLEAN,
        variant=None,
        documents=[
            _doc("form.pdf", DocType.CLAIM_FORM, claimant_name="Ana Diaz", incident_date=incident),
            _doc("id.pdf", DocType.ID_CARD, full_name="DIAZ ANA", expiry_date="2030-01-01"),
            _doc("inv.pdf", DocType.INVOICE, total="900.00", invoice_date="2025-06-02"),
            _doc("report.pdf", DocType.INCIDENT_REPORT, report_number="R1", incident_date=incident),
        ],
        rule_verdicts=rule_verdicts,
    )
    (folder / "ground_truth.json").write_text(truth.model_dump_json())


def _run(
    root: Path, extracted: dict[str, dict[str, list[FieldResult]]] | None = None
) -> StageResult:
    ctx = EvalContext(root, get_settings())
    ctx.extracted.update(extracted or {})
    return validation.run("dev", ctx)


def _found(name: str, value: str) -> FieldResult:
    return FieldResult(name, value, 0.95, Verdict.FOUND, Evidence("doc", 1, (0, 1), value), None)


def _extracted(**overrides: FieldResult) -> dict[str, dict[str, list[FieldResult]]]:
    """What the extraction stage would hand over for p1, gold values unless overridden."""
    gold = {
        "form.pdf": [_found("claimant_name", "Ana Diaz"), _found("incident_date", "2025-06-01")],
        "id.pdf": [_found("full_name", "DIAZ ANA"), _found("expiry_date", "2030-01-01")],
        "inv.pdf": [_found("total", "900.00"), _found("invoice_date", "2025-06-02")],
        "report.pdf": [_found("report_number", "R1"), _found("incident_date", "2025-06-01")],
    }
    for filename, fields in gold.items():
        gold[filename] = [overrides.get(f.field_name, f) for f in fields]
    return {"p1": gold}


def test_rules_agree_with_gold(tmp_path: Path) -> None:
    _write(tmp_path, dict.fromkeys(RULE_IDS, Verdict.FOUND))

    result = _run(tmp_path)

    assert result.metrics == {"validation.accuracy": 1.0}
    assert result.details["mismatches"] == []
    assert result.details["rule_checks"] == 7


def test_disagreement_is_reported_with_the_rule_and_both_verdicts(tmp_path: Path) -> None:
    gold = dict.fromkeys(RULE_IDS, Verdict.FOUND)
    gold["R5_ID_VALID_ON_INCIDENT"] = Verdict.CONFLICTING
    _write(tmp_path, gold)

    result = _run(tmp_path)

    assert result.metrics["validation.accuracy"] == 6 / 7
    (miss,) = result.details["mismatches"]
    assert (miss["rule"], miss["expected"], miss["actual"]) == (
        "R5_ID_VALID_ON_INCIDENT",
        "conflicting",
        "found",
    )
    assert result.details["per_rule_accuracy"]["R5_ID_VALID_ON_INCIDENT"] == 0.0


def test_rule_set_must_match_gold(tmp_path: Path) -> None:
    gold = dict.fromkeys(RULE_IDS, Verdict.FOUND)
    del gold["R4_INVOICE_AFTER_INCIDENT"]
    _write(tmp_path, gold)

    result = _run(tmp_path)

    (miss,) = result.details["mismatches"]
    assert (miss["rule"], miss["expected"]) == ("R4_INVOICE_AFTER_INCIDENT", "not run")


def test_incident_outside_every_version_conflicts_on_r2_and_misses_the_limit(
    tmp_path: Path,
) -> None:
    gold = dict.fromkeys(RULE_IDS, Verdict.FOUND)
    gold["R2_INCIDENT_IN_COVERAGE"] = Verdict.CONFLICTING
    gold["R3_INVOICE_WITHIN_LIMIT"] = Verdict.MISSING
    gold["R4_INVOICE_AFTER_INCIDENT"] = Verdict.CONFLICTING
    _write(tmp_path, gold, incident="2026-01-01")

    result = _run(tmp_path)

    # The version ends on 2026-01-01 and the end is exclusive. The invoice (2025-06-02) is
    # then before the incident, and the id (2030) is still valid.
    assert result.metrics["validation.accuracy"] == 1.0


def test_end_to_end_is_left_out_when_extraction_did_not_run(tmp_path: Path) -> None:
    _write(tmp_path, dict.fromkeys(RULE_IDS, Verdict.FOUND))

    result = _run(tmp_path)

    assert "validation.end_to_end_accuracy" not in result.metrics
    assert result.details["end_to_end"] is None


def test_end_to_end_scores_the_rules_on_the_extracted_fields(tmp_path: Path) -> None:
    _write(tmp_path, dict.fromkeys(RULE_IDS, Verdict.FOUND))
    # The model read the wrong total; on gold fields every rule passes.
    extracted = _extracted(total=_found("total", "90000.00"))

    result = _run(tmp_path, extracted)

    assert result.metrics["validation.accuracy"] == 1.0
    assert result.metrics["validation.end_to_end_accuracy"] == 6 / 7
    end_to_end = result.details["end_to_end"]
    assert (end_to_end["packets_scored"], end_to_end["rule_checks"]) == (1, 7)
    (miss,) = end_to_end["mismatches"]
    assert (miss["rule"], miss["expected"], miss["actual"]) == (
        "R3_INVOICE_WITHIN_LIMIT",
        "found",
        "conflicting",
    )


def test_end_to_end_skips_a_packet_with_a_document_that_was_not_extracted(
    tmp_path: Path,
) -> None:
    _write(tmp_path, dict.fromkeys(RULE_IDS, Verdict.FOUND))
    extracted = _extracted()
    del extracted["p1"]["id.pdf"]

    result = _run(tmp_path, extracted)

    assert "validation.end_to_end_accuracy" not in result.metrics
    assert result.details["end_to_end"]["packets_scored"] == 0
    assert result.details["end_to_end"]["packets_skipped"] == ["p1"]
