"""Validation stage: the rules applied to gold fields, compared with gold rule verdicts.

`validation.accuracy` (gated) needs no OCR and no model: each packet's ground truth becomes a
ClaimContext (a gold field is `found` with confidence 1.0 unless its gold verdict says
otherwise) and every rule must give the verdict the generator computed. The rules are
deterministic, so anything below 1.00 is a bug in a rule or in the generator. It checks the
rules against the generator, not the system's decisions.

`validation.end_to_end_accuracy` (not gated) runs the same rules on the fields the extraction
stage produced from real OCR and the model (replayed from the cassettes under
LLM_BACKEND=replay), and compares with the same gold verdicts. Documents carry their gold doc
type, as in the extraction stage, so classification errors are not part of it. It needs the
extraction stage to run first in the same invocation (`--stages extraction,validation`) and is
left out otherwise. A packet is scored only if every document that is not `other` was
extracted; the extraction stage skips scans that are unreadable by construction, so those
packets are listed as skipped instead of being scored on made-up fields.
"""

import json
from collections import Counter
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any

from core.types import DocType, Evidence, Verdict
from evals.context import EvalContext
from evals.dataset.types import GroundTruth, PolicyIndex, PolicyVersionEntry
from evals.report import StageResult
from pipeline.extract import FieldResult
from pipeline.validate import RuleResult, run_rules
from pipeline.validate.base import ClaimContext, DocFields, PolicyVersionInfo
from rag.chunking import chunk_clauses


def run(split: str, ctx: EvalContext) -> StageResult:
    splits = json.loads((ctx.data_dir / "splits.json").read_text(encoding="utf-8"))
    index = PolicyIndex.model_validate_json(
        (ctx.data_dir / "policies" / "index.json").read_text(encoding="utf-8")
    )
    gold = _Score()
    end_to_end = _Score()
    skipped: list[str] = []

    for packet_id in splits[split]:
        folder = ctx.data_dir / "packets" / packet_id
        truth = GroundTruth.model_validate_json((folder / "ground_truth.json").read_text())
        gold.add(truth, run_rules(_context(truth, index, ctx.data_dir)))
        extracted = ctx.extracted.get(packet_id)
        if extracted is None:
            continue
        documents = _extracted_documents(truth, extracted)
        if documents is None:
            skipped.append(packet_id)
            continue
        end_to_end.add(truth, run_rules(_context(truth, index, ctx.data_dir, documents)))

    metrics = {"validation.accuracy": gold.accuracy()}
    details: dict[str, Any] = {
        "split": split,
        "packets": len(splits[split]),
        **gold.details(),
        "end_to_end": None,
    }
    if ctx.extracted:
        details["end_to_end"] = {
            "doc_types": "gold",
            "packets_scored": end_to_end.packets,
            "packets_skipped": skipped,
            **end_to_end.details(),
        }
        if end_to_end.checked:
            metrics["validation.end_to_end_accuracy"] = end_to_end.accuracy()
    return StageResult(metrics=metrics, details=details)


class _Score:
    """Rule verdicts against gold, per rule, with every disagreement kept."""

    def __init__(self) -> None:
        self.correct: Counter[str] = Counter()
        self.total: Counter[str] = Counter()
        self.mismatches: list[dict[str, Any]] = []
        self.packets = 0

    @property
    def checked(self) -> int:
        return sum(self.total.values())

    def add(self, truth: GroundTruth, rule_results: list[RuleResult]) -> None:
        self.packets += 1
        results = {r.rule_id: r for r in rule_results}
        for rule_id in sorted(set(results) | set(truth.rule_verdicts)):
            expected = truth.rule_verdicts.get(rule_id)
            result = results.get(rule_id)
            actual = result.verdict if result else None
            self.total[rule_id] += 1
            if actual == expected:
                self.correct[rule_id] += 1
                continue
            self.mismatches.append(
                {
                    "packet_id": truth.packet_id,
                    "scenario": truth.scenario.value,
                    "variant": truth.variant,
                    "rule": rule_id,
                    "expected": expected.value if expected else "not run",
                    "actual": actual.value if actual else "not run",
                    "message": result.message if result else None,
                }
            )

    def accuracy(self) -> float:
        return sum(self.correct.values()) / self.checked if self.checked else 0.0

    def details(self) -> dict[str, Any]:
        return {
            "rule_checks": self.checked,
            "per_rule_accuracy": {r: self.correct[r] / self.total[r] for r in sorted(self.total)},
            "mismatches": self.mismatches,
        }


def _extracted_documents(
    truth: GroundTruth, extracted: dict[str, list[FieldResult]]
) -> list[DocFields] | None:
    """The packet's documents with the extraction stage's fields, or None if one is lacking."""
    documents: list[DocFields] = []
    for doc in truth.documents:
        if doc.doc_type is DocType.OTHER:
            # As in the worker: an `other` document is sent to a person and has no fields.
            documents.append(DocFields(doc.filename, doc.doc_type, {}))
            continue
        fields = extracted.get(doc.filename)
        if fields is None:
            return None
        documents.append(DocFields(doc.filename, doc.doc_type, {f.field_name: f for f in fields}))
    return documents


def _context(
    truth: GroundTruth,
    index: PolicyIndex,
    data_dir: Path,
    documents: list[DocFields] | None = None,
) -> ClaimContext:
    """The claim as the rules see it: gold fields unless `documents` is given."""
    version = _version_in_force(truth.policy_number, truth.incident_date, index)
    limit = _coverage_limit(version, truth.claim_type.value, data_dir) if version else None
    documents = documents if documents is not None else _gold_documents(truth)
    info = (
        PolicyVersionInfo(version.version_no, version.effective_from, version.effective_to)
        if version
        else None
    )
    return ClaimContext(truth.claim_type, truth.incident_date, documents, info, limit)


def _gold_documents(truth: GroundTruth) -> list[DocFields]:
    return [
        DocFields(
            doc.filename,
            doc.doc_type,
            {
                name: FieldResult(
                    field_name=name,
                    value=value,
                    confidence=1.0 if doc.field_verdicts[name] is Verdict.FOUND else 0.5,
                    verdict=doc.field_verdicts[name],
                    evidence=Evidence(doc.filename, None, None, value),
                    details=None,
                )
                for name, value in doc.fields.items()
            },
        )
        for doc in truth.documents
    ]


def _version_in_force(
    policy_number: str, day: date, index: PolicyIndex
) -> PolicyVersionEntry | None:
    """Half-open range, as policy_in_force does it in the database."""
    for policy in index.policies:
        if policy.policy_number != policy_number:
            continue
        for v in policy.versions:
            if v.effective_from <= day and (v.effective_to is None or day < v.effective_to):
                return v
    return None


def _coverage_limit(version: PolicyVersionEntry, claim_type: str, data_dir: Path) -> Decimal | None:
    markdown = (data_dir / "policies" / version.file).read_text(encoding="utf-8")
    for clause in chunk_clauses(markdown):
        if clause.coverage_type == claim_type and clause.limit_amount is not None:
            return clause.limit_amount
    return None
