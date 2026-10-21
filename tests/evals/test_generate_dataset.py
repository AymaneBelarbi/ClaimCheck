import hashlib
import json
import re
import shutil
import subprocess
import time
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path

import pytest
from rapidfuzz import fuzz

from core.types import ClaimType, DocType, Verdict
from evals.dataset.render import DATE_STYLES, format_date
from evals.dataset.types import (
    FIELD_NAMES,
    ID_NUMBER_PATTERN,
    POLICY_NUMBER_PATTERN,
    REQUIRED_DOCUMENTS,
    RULE_IDS,
    ClauseQuestion,
    GroundTruth,
    PolicyIndex,
    Scenario,
)
from evals.generate_dataset import generate

SEED = 20261020
HEADING = re.compile(r"^## (\d+\.\d+) (.+)$", re.MULTILINE)
LIMIT = re.compile(r"limit of \$([\d,]+(?:\.\d{2})?)")
ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
AMOUNT = re.compile(r"^\d+\.\d{2}$")


@dataclass(frozen=True)
class Generated:
    out: Path
    questions: Path
    seconds: float


def _run(root: Path) -> Generated:
    start = time.perf_counter()
    generate(out=root / "data", seed=SEED, questions_path=root / "clause_questions.jsonl")
    return Generated(root / "data", root / "clause_questions.jsonl", time.perf_counter() - start)


@pytest.fixture(scope="module")
def ds(tmp_path_factory: pytest.TempPathFactory) -> Generated:
    return _run(tmp_path_factory.mktemp("dataset"))


@pytest.fixture(scope="module")
def truths(ds: Generated) -> list[GroundTruth]:
    return [
        GroundTruth.model_validate_json(p.read_text())
        for p in sorted((ds.out / "packets").glob("*/ground_truth.json"))
    ]


@pytest.fixture(scope="module")
def index(ds: Generated) -> PolicyIndex:
    return PolicyIndex.model_validate_json((ds.out / "policies" / "index.json").read_text())


@pytest.fixture(scope="module")
def questions(ds: Generated) -> list[ClauseQuestion]:
    lines = ds.questions.read_text().splitlines()
    return [ClauseQuestion.model_validate_json(line) for line in lines]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _norm(name: str) -> str:
    decomposed = unicodedata.normalize("NFKD", name)
    return "".join(c for c in decomposed if not unicodedata.combining(c)).casefold()


def _fields(gt: GroundTruth, doc_type: DocType) -> list[dict[str, str | None]]:
    return [d.fields for d in gt.documents if d.doc_type == doc_type]


def _in_force(index: PolicyIndex, policy_number: str, day: date) -> int | None:
    policy = next(p for p in index.policies if p.policy_number == policy_number)
    for v in policy.versions:
        if v.effective_from <= day and (v.effective_to is None or day < v.effective_to):
            return v.version_no
    return None


def _clauses(ds: Generated, policy_number: str, version_no: int) -> dict[str, str]:
    text = (ds.out / "policies" / policy_number / f"v{version_no}.md").read_text()
    sections = HEADING.split(text)[1:]
    # split() with two groups yields [ref, title, body, ref, title, body, ...]
    return {sections[i]: sections[i + 2] for i in range(0, len(sections), 3)}


def _clause_titles(ds: Generated, policy_number: str, version_no: int) -> dict[str, str]:
    text = (ds.out / "policies" / policy_number / f"v{version_no}.md").read_text()
    return dict(HEADING.findall(text))


# --- shape -------------------------------------------------------------------------------


def test_generates_50_packets(ds: Generated, truths: list[GroundTruth]) -> None:
    assert len(truths) == 50
    assert len({gt.packet_id for gt in truths}) == 50
    for gt in truths:
        folder = ds.out / "packets" / gt.packet_id
        on_disk = {p.name for p in folder.iterdir()} - {"ground_truth.json"}
        assert on_disk == {d.filename for d in gt.documents}
        for doc in gt.documents:
            assert (folder / doc.filename).stat().st_size > 0


def test_generation_is_fast(ds: Generated) -> None:
    assert ds.seconds < 60


def test_split_is_35_15_and_disjoint(ds: Generated, truths: list[GroundTruth]) -> None:
    splits = json.loads((ds.out / "splits.json").read_text())
    assert set(splits) == {"dev", "holdout"}
    dev, holdout = set(splits["dev"]), set(splits["holdout"])
    assert len(splits["dev"]) == len(dev) == 35
    assert len(splits["holdout"]) == len(holdout) == 15
    assert not dev & holdout
    assert dev | holdout == {gt.packet_id for gt in truths}
    # Every failure mode is seen during development and checked again in CI.
    by_id = {gt.packet_id: gt.scenario for gt in truths}
    assert {by_id[p] for p in holdout} == set(Scenario)
    assert {by_id[p] for p in dev} == set(Scenario)


def test_deterministic(ds: Generated, tmp_path: Path) -> None:
    again = _run(tmp_path)
    first = sorted(p.relative_to(ds.out) for p in ds.out.rglob("*") if p.is_file())
    second = sorted(p.relative_to(again.out) for p in again.out.rglob("*") if p.is_file())
    assert first == second
    for rel in first:
        assert _sha(ds.out / rel) == _sha(again.out / rel), rel
    assert ds.questions.read_text() == again.questions.read_text()


# SHA-256 over "<path> <sha256>" lines of every generated file except ground_truth.json, sorted
# by path. The recorded cassettes were made from these exact files: a change to the generator
# that moves a single pixel or byte invalidates them, so it has to show up here first.
RENDERED_DIGEST = "149dd55676478454284a734246d71497c41e5a7c080ad5b62dc925e03c6cc3c0"


def test_rendered_files_are_unchanged(ds: Generated) -> None:
    files = sorted(p for p in ds.out.rglob("*") if p.is_file() and p.name != "ground_truth.json")
    # PNG bytes depend on the platform's Pillow and FreeType build, so scans count by name
    # only; every other rendered file is byte-identical on macOS and Linux.
    lines = "".join(
        f"{p.relative_to(ds.out).as_posix()} {'' if p.suffix == '.png' else _sha(p)}\n"
        for p in files
    )

    assert len(files) == 179
    assert hashlib.sha256(lines.encode()).hexdigest() == RENDERED_DIGEST


def test_every_scenario_present_at_least_3_times(truths: list[GroundTruth]) -> None:
    counts = Counter(gt.scenario for gt in truths)
    assert set(counts) == set(Scenario)
    assert all(n >= 3 for n in counts.values()), counts


def test_ground_truth_validates(ds: Generated) -> None:
    for path in sorted((ds.out / "packets").glob("*/ground_truth.json")):
        gt = GroundTruth.model_validate_json(path.read_text())
        assert gt.packet_id == path.parent.name


# --- ground truth content ---------------------------------------------------------------


def test_field_names_and_value_formats(truths: list[GroundTruth]) -> None:
    for gt in truths:
        for doc in gt.documents:
            assert set(doc.fields) == set(FIELD_NAMES[doc.doc_type])
            assert set(doc.field_verdicts) == set(FIELD_NAMES[doc.doc_type])
            f = doc.fields
            if doc.doc_type == DocType.CLAIM_FORM:
                assert f["incident_type"] == gt.claim_type.value
                assert f["policy_number"] == gt.policy_number
                assert f["incident_date"] == gt.incident_date.isoformat()
            if doc.doc_type == DocType.ID_CARD:
                assert re.fullmatch(ID_NUMBER_PATTERN, f["id_number"] or "")
                assert ISO_DATE.match(f["date_of_birth"] or "")
                assert ISO_DATE.match(f["expiry_date"] or "")
            if doc.doc_type == DocType.INVOICE:
                assert ISO_DATE.match(f["invoice_date"] or "")
                assert AMOUNT.match(f["total"] or "")
                items = json.loads(f["line_items"] or "")
                assert items and all(set(i) == {"description", "amount"} for i in items)
                assert all(AMOUNT.match(i["amount"]) for i in items)
                assert sum(Decimal(i["amount"]) for i in items) == Decimal(f["total"] or "")
            if doc.doc_type == DocType.INCIDENT_REPORT:
                assert f["incident_date"] == gt.incident_date.isoformat()


def test_id_and_policy_number_formats_do_not_overlap(truths: list[GroundTruth]) -> None:
    for gt in truths:
        assert re.fullmatch(POLICY_NUMBER_PATTERN, gt.policy_number)
        assert not re.search(ID_NUMBER_PATTERN, gt.policy_number)


def test_theft_packets_have_no_invoice_rules(truths: list[GroundTruth]) -> None:
    for gt in truths:
        expected = set(RULE_IDS)
        if gt.claim_type == ClaimType.THEFT:
            expected -= {"R3_INVOICE_WITHIN_LIMIT", "R4_INVOICE_AFTER_INCIDENT"}
        assert set(gt.rule_verdicts) == expected, gt.packet_id


def test_clean_packets_pass_every_rule(truths: list[GroundTruth]) -> None:
    for gt in truths:
        if gt.scenario == Scenario.CLEAN:
            assert set(gt.rule_verdicts.values()) == {Verdict.FOUND}, gt.packet_id
            for doc in gt.documents:
                assert set(doc.field_verdicts.values()) == {Verdict.FOUND}
            present = {d.doc_type for d in gt.documents}
            assert present == REQUIRED_DOCUMENTS[gt.claim_type]


def _not_found(gt: GroundTruth) -> dict[str, Verdict]:
    return {k: v for k, v in gt.rule_verdicts.items() if v != Verdict.FOUND}


def test_name_variants_sit_clearly_on_their_side_of_the_threshold(
    truths: list[GroundTruth],
) -> None:
    for gt in truths:
        claim = _fields(gt, DocType.CLAIM_FORM)
        ids = _fields(gt, DocType.ID_CARD)
        if not claim or not ids:
            continue
        score = fuzz.token_sort_ratio(
            _norm(claim[0]["claimant_name"] or ""), _norm(ids[0]["full_name"] or "")
        )
        if gt.scenario == Scenario.NAME_MISMATCH:
            assert score <= 85, (gt.packet_id, score)
            assert _not_found(gt) == {"R1_NAME_MATCH": Verdict.CONFLICTING}
        else:
            assert score >= 95, (gt.packet_id, score)
    # Clean packets include printed variants that differ from the claim form but match.
    differing = [
        gt
        for gt in truths
        if gt.scenario == Scenario.CLEAN
        and _fields(gt, DocType.CLAIM_FORM)[0]["claimant_name"]
        != _fields(gt, DocType.ID_CARD)[0]["full_name"]
    ]
    assert len(differing) >= 3


def test_incident_outside_coverage(truths: list[GroundTruth], index: PolicyIndex) -> None:
    cases = [gt for gt in truths if gt.scenario == Scenario.INCIDENT_OUTSIDE_COVERAGE]
    lapsed = {p.policy_number for p in index.policies if p.versions[-1].effective_to is not None}
    assert lapsed, "at least one policy must lapse"
    kinds = set()
    for gt in cases:
        assert _in_force(index, gt.policy_number, gt.incident_date) is None
        policy = next(p for p in index.policies if p.policy_number == gt.policy_number)
        if gt.incident_date < policy.versions[0].effective_from:
            kinds.add("before_first_version")
        else:
            assert gt.policy_number in lapsed
            kinds.add("after_lapse")
        expected = {"R2_INCIDENT_IN_COVERAGE": Verdict.CONFLICTING}
        if gt.claim_type != ClaimType.THEFT:
            expected["R3_INVOICE_WITHIN_LIMIT"] = Verdict.MISSING
        assert _not_found(gt) == expected
    assert kinds == {"before_first_version", "after_lapse"}
    for gt in truths:
        if gt.scenario != Scenario.INCIDENT_OUTSIDE_COVERAGE:
            assert _in_force(index, gt.policy_number, gt.incident_date) is not None


def test_glass_claims_only_against_versions_with_glass_cover(
    ds: Generated, truths: list[GroundTruth], index: PolicyIndex
) -> None:
    for gt in truths:
        if gt.claim_type != ClaimType.GLASS:
            continue
        version_no = _in_force(index, gt.policy_number, gt.incident_date)
        assert version_no is not None
        titles = _clause_titles(ds, gt.policy_number, version_no)
        assert any("glass" in t.casefold() for t in titles.values())


def test_id_expired(truths: list[GroundTruth]) -> None:
    for gt in truths:
        for f in _fields(gt, DocType.ID_CARD):
            expired = date.fromisoformat(f["expiry_date"] or "") < gt.incident_date
            assert expired == (gt.scenario == Scenario.ID_EXPIRED), gt.packet_id
        if gt.scenario == Scenario.ID_EXPIRED:
            assert _not_found(gt) == {"R5_ID_VALID_ON_INCIDENT": Verdict.CONFLICTING}


def test_missing_document_varies_which_document(truths: list[GroundTruth]) -> None:
    absent_types = []
    for gt in truths:
        present = {d.doc_type for d in gt.documents}
        absent = REQUIRED_DOCUMENTS[gt.claim_type] - present
        if gt.scenario != Scenario.MISSING_DOCUMENT:
            assert not absent, gt.packet_id
            continue
        assert len(absent) == 1
        absent_types.append(next(iter(absent)))
        assert gt.rule_verdicts["R6_REQUIRED_DOCUMENTS"] == Verdict.MISSING
    assert set(absent_types) == {DocType.ID_CARD, DocType.INVOICE, DocType.INCIDENT_REPORT}


def test_incident_dates_on_the_documents_match_the_claim(truths: list[GroundTruth]) -> None:
    for gt in truths:
        unreadable_form = gt.variant == "unreadable:claim_form"
        expected = Verdict.LOW_CONFIDENCE if unreadable_form else Verdict.FOUND
        assert gt.rule_verdicts["R7_INCIDENT_DATE_CONSISTENT"] == expected, gt.packet_id


def test_unreadable_scan_marks_every_field_low_confidence(truths: list[GroundTruth]) -> None:
    for gt in truths:
        low = [d for d in gt.documents if Verdict.LOW_CONFIDENCE in d.field_verdicts.values()]
        if gt.scenario != Scenario.UNREADABLE_SCAN:
            assert not low
            continue
        assert len(low) == 1
        doc = low[0]
        assert set(doc.field_verdicts.values()) == {Verdict.LOW_CONFIDENCE}
        assert all(v is not None for v in doc.fields.values())
        assert set(_not_found(gt).values()) == {Verdict.LOW_CONFIDENCE}


def test_invoice_scenarios(truths: list[GroundTruth], ds: Generated, index: PolicyIndex) -> None:
    for gt in truths:
        invoices = _fields(gt, DocType.INVOICE)
        if gt.scenario == Scenario.TOTAL_OVER_LIMIT:
            version_no = _in_force(index, gt.policy_number, gt.incident_date)
            assert version_no is not None
            clauses = _clauses(ds, gt.policy_number, version_no)
            titles = _clause_titles(ds, gt.policy_number, version_no)
            ref = next(r for r, t in titles.items() if gt.claim_type.value in t.casefold())
            match = LIMIT.search(clauses[ref])
            assert match
            assert Decimal(invoices[0]["total"] or "") > Decimal(match.group(1).replace(",", ""))
            assert _not_found(gt) == {"R3_INVOICE_WITHIN_LIMIT": Verdict.CONFLICTING}
        if gt.scenario == Scenario.INVOICE_BEFORE_INCIDENT:
            assert date.fromisoformat(invoices[0]["invoice_date"] or "") < gt.incident_date
            assert _not_found(gt) == {"R4_INVOICE_AFTER_INCIDENT": Verdict.CONFLICTING}
        if gt.scenario == Scenario.DUPLICATE_CONFLICTING_INVOICE:
            assert len(invoices) == 2
            assert invoices[0]["total"] != invoices[1]["total"]
            assert _not_found(gt) == {"R3_INVOICE_WITHIN_LIMIT": Verdict.CONFLICTING}
            for doc in gt.documents:
                if doc.doc_type == DocType.INVOICE:
                    assert doc.field_verdicts["total"] == Verdict.CONFLICTING
        elif gt.claim_type != ClaimType.THEFT and gt.scenario != Scenario.MISSING_DOCUMENT:
            assert len(invoices) == 1


def test_edge_cases_on_the_passing_side_are_present(truths: list[GroundTruth]) -> None:
    clean = [gt for gt in truths if gt.scenario == Scenario.CLEAN]
    same_day = [
        gt
        for gt in clean
        for f in _fields(gt, DocType.INVOICE)
        if f["invoice_date"] == gt.incident_date.isoformat()
    ]
    expires_on_day = [
        gt
        for gt in clean
        for f in _fields(gt, DocType.ID_CARD)
        if f["expiry_date"] == gt.incident_date.isoformat()
    ]
    assert same_day and expires_on_day


# --- dates and text helpers -------------------------------------------------------------


def test_printed_dates_are_never_ambiguous() -> None:
    day = date(2026, 1, 1)
    while day.year == 2026:
        for style in DATE_STYLES:
            text = format_date(day, style)
            numeric = re.fullmatch(r"(\d{2})/(\d{2})/(\d{4})", text)
            if numeric:
                assert int(numeric.group(1)) > 12, text
        day += timedelta(days=1)


# --- scans ------------------------------------------------------------------------------


def _mean_ocr_confidence(path: Path) -> float:
    tsv = subprocess.run(
        ["tesseract", str(path), "-", "tsv"], capture_output=True, text=True, check=True
    ).stdout
    rows = [r.split("\t") for r in tsv.splitlines()[1:]]
    scores = [float(r[10]) for r in rows if len(r) == 12 and r[11].strip() and float(r[10]) >= 0]
    return sum(scores) / len(scores) / 100 if scores else 0.0


@pytest.mark.ocr
@pytest.mark.skipif(shutil.which("tesseract") is None, reason="tesseract not installed")
def test_unreadable_scans_fall_below_the_ocr_threshold(
    ds: Generated, truths: list[GroundTruth]
) -> None:
    checked_ids = 0
    for gt in truths:
        if gt.scenario != Scenario.UNREADABLE_SCAN:
            continue
        for doc in gt.documents:
            path = ds.out / "packets" / gt.packet_id / doc.filename
            if Verdict.LOW_CONFIDENCE in doc.field_verdicts.values():
                assert _mean_ocr_confidence(path) < 0.40, doc.filename
            elif doc.doc_type == DocType.ID_CARD:
                assert _mean_ocr_confidence(path) > 0.80, doc.filename
                checked_ids += 1
    assert checked_ids >= 2


# --- policies and clause questions ------------------------------------------------------


def test_policy_index_and_markdown(ds: Generated, index: PolicyIndex) -> None:
    assert len(index.policies) == 3
    for policy in index.policies:
        assert 2 <= len(policy.versions) <= 3
        assert [v.version_no for v in policy.versions] == list(range(1, len(policy.versions) + 1))
        for prev, nxt in zip(policy.versions, policy.versions[1:], strict=False):
            assert prev.effective_to == nxt.effective_from
        for v in policy.versions:
            titles = _clause_titles(ds, policy.policy_number, v.version_no)
            clauses = _clauses(ds, policy.policy_number, v.version_no)
            assert 10 <= len(titles) <= 15
            for kind in ("collision", "theft"):
                refs = [r for r, t in titles.items() if kind in t.casefold()]
                assert len(refs) == 1
                assert LIMIT.search(clauses[refs[0]])
            glass = [r for r, t in titles.items() if "glass" in t.casefold()]
            assert len(glass) <= 1
            with_limit = [r for r, body in clauses.items() if LIMIT.search(body)]
            assert len(with_limit) == 2 + len(glass)
        first = _clause_titles(ds, policy.policy_number, 1)
        assert not any("glass" in t.casefold() for t in first.values())
        last = _clause_titles(ds, policy.policy_number, policy.versions[-1].version_no)
        assert any("glass" in t.casefold() for t in last.values())


def test_clause_questions_reference_existing_clauses(
    ds: Generated, questions: list[ClauseQuestion], index: PolicyIndex
) -> None:
    assert len(questions) >= 40
    assert len({q.id for q in questions}) == len(questions)
    for q in questions:
        assert _in_force(index, q.policy_number, q.incident_date) == q.expected_version_no
        titles = _clause_titles(ds, q.policy_number, q.expected_version_no)
        assert q.expected_clause_ref in titles, q


def test_clause_questions_depend_on_version(questions: list[ClauseQuestion]) -> None:
    refs: dict[tuple[str, str], set[str]] = defaultdict(set)
    for q in questions:
        refs[(q.policy_number, q.question)].add(q.expected_clause_ref)
    sensitive = [q for q in questions if len(refs[(q.policy_number, q.question)]) > 1]
    assert len(sensitive) >= 10


def test_committed_clause_questions_match_the_generator(ds: Generated) -> None:
    committed = Path(__file__).parents[2] / "evals" / "clause_questions.jsonl"
    assert committed.read_text() == ds.questions.read_text()
