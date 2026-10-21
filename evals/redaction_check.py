"""How much personal data gets past the redactor, measured on the generated packets.

    python -m evals.redaction_check --split all

Every document is read with the real OCR, then redacted the way a cloud call redacts it: a
`Redactor` that has learned the names labelled anywhere in the packet (what
`RedactingGateway.scoped` builds), fresh for each document. The whole OCR text of the document
is checked, including any part the extraction prompt would cut off.

What counts, per document:

- ID numbers: the packet's gold licence numbers that appear verbatim in the OCR text. One leaks
  if it is still there after redaction.
- Name pairs: each person of the packet (the gold claimant name on the claim form and the gold
  name printed on the licence; two people when they differ) whose name has a token in the OCR
  text. A token is a word of the name of two or more letters that is not a particle such as
  "de"; it is matched as a whole word, ignoring accents and case, and an occurrence counts only
  if it starts with a capital letter, as printed names do ("Mark" yes, "mark" no). A pair leaks
  if any occurrence of any of its tokens is left after redaction.
- Token occurrences: the same occurrences, counted one by one.

A name that OCR misread ("lain" for "Iain") is not counted as present, so it cannot count as a
leak either; the check measures the redactor on the text it is given. Names that are not in
the ground truth (the police officer on an incident report, a vendor named after a person) are
not counted.

Three configurations are measured on the same OCR text: the one in production, the same labels
without packet scope, and only the six labels the redactor started with (Name, Full name,
Insured, Claimant, Driver, Officer; colon or next-line values only) with packet scope.

The report is written to `evals/reports/<date>-redaction.json`.
"""

import argparse
import json
import re
import sys
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytesseract

from core.types import DocType
from evals.dataset.types import GroundTruth
from evals.report import StageResult, write_report
from gateway.redaction import Redactor
from pipeline.ocr import run_ocr

FIRST_LABELS = ("name", "full name", "insured", "claimant", "driver", "officer")
PARTICLES = frozenset(
    [
        "de",
        "da",
        "do",
        "dos",
        "das",
        "del",
        "della",
        "di",
        "van",
        "von",
        "der",
        "den",
        "la",
        "le",
        "el",
        "al",
        "bin",
        "ibn",
        "ben",
        "st",
    ]
)
CONFIGURATIONS = ("production", "no_packet_scope", "first_six_labels")


@dataclass
class _Tally:
    present: int = 0
    leaked: int = 0

    def as_dict(self) -> dict[str, int]:
        return {"present": self.present, "leaked": self.leaked}


@dataclass
class _Result:
    ids: _Tally = field(default_factory=_Tally)
    pairs: _Tally = field(default_factory=_Tally)
    tokens: _Tally = field(default_factory=_Tally)
    pairs_by_doc_type: dict[str, _Tally] = field(default_factory=lambda: defaultdict(_Tally))
    ids_by_doc_type: dict[str, _Tally] = field(default_factory=lambda: defaultdict(_Tally))
    leaks: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "id_numbers": self.ids.as_dict(),
            "name_pairs": self.pairs.as_dict(),
            "name_token_occurrences": self.tokens.as_dict(),
            "name_pairs_by_doc_type": {t: v.as_dict() for t, v in self.pairs_by_doc_type.items()},
            "id_numbers_by_doc_type": {t: v.as_dict() for t, v in self.ids_by_doc_type.items()},
            "leaks": self.leaks,
        }


def fold(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def name_tokens(name: str) -> set[str]:
    words = re.findall(r"[^\W\d_]+", fold(name))
    return {w.casefold() for w in words if len(w) >= 2 and w.casefold() not in PARTICLES}


def occurrences(text: str, tokens: set[str]) -> list[str]:
    """Capitalized whole-word occurrences of `tokens` in `text`, accents and case ignored."""
    found: list[str] = []
    for match in re.finditer(r"[^\W\d_]+", fold(text)):
        word = match.group(0)
        if word[0].isupper() and word.casefold() in tokens:
            found.append(word)
    return found


def people(truth: GroundTruth) -> list[str]:
    names: list[str] = []
    for doc in truth.documents:
        if doc.doc_type is DocType.CLAIM_FORM:
            value = doc.fields.get("claimant_name")
        elif doc.doc_type is DocType.ID_CARD:
            value = doc.fields.get("full_name")
        else:
            continue
        if value and name_tokens(value) not in [name_tokens(n) for n in names]:
            names.append(value)
    return names


def id_numbers(truth: GroundTruth) -> set[str]:
    return {
        value
        for doc in truth.documents
        if doc.doc_type is DocType.ID_CARD and (value := doc.fields.get("id_number"))
    }


def redactor(configuration: str, packet_texts: list[str]) -> Redactor:
    """The redactor a call on one of the packet's documents would start from."""
    if configuration == "first_six_labels":
        template = Redactor(labels=FIRST_LABELS, space_labels=())
    else:
        template = Redactor()
    if configuration != "no_packet_scope":
        template.learn_from(packet_texts)
    return template


def check(data_dir: Path, packet_ids: list[str]) -> dict[str, Any]:
    results = {c: _Result() for c in CONFIGURATIONS}
    documents = 0
    for packet_id in packet_ids:
        folder = data_dir / "packets" / packet_id
        truth = GroundTruth.model_validate_json((folder / "ground_truth.json").read_text())
        texts = {
            doc.filename: run_ocr((folder / doc.filename).read_bytes(), doc.filename).text
            for doc in truth.documents
        }
        persons = people(truth)
        ids = id_numbers(truth)
        documents += len(texts)
        for configuration, result in results.items():
            template = redactor(configuration, list(texts.values()))
            for doc in truth.documents:
                raw = texts[doc.filename]
                redacted, _ = template.fresh().redact(raw)
                doc_type = doc.doc_type.value
                for number in ids:
                    if number not in raw:
                        continue
                    leaked = number in redacted
                    for tally in (result.ids, result.ids_by_doc_type[doc_type]):
                        tally.present += 1
                        tally.leaked += leaked
                for person in persons:
                    tokens = name_tokens(person)
                    before = occurrences(raw, tokens)
                    if not before:
                        continue
                    after = occurrences(redacted, tokens)
                    result.tokens.present += len(before)
                    result.tokens.leaked += len(after)
                    for tally in (result.pairs, result.pairs_by_doc_type[doc_type]):
                        tally.present += 1
                        tally.leaked += bool(after)
                    if after:
                        result.leaks.append(
                            {
                                "packet_id": packet_id,
                                "filename": doc.filename,
                                "doc_type": doc_type,
                                "person": person,
                                "left": sorted(set(after)),
                            }
                        )
    return {
        "packets": len(packet_ids),
        "documents": documents,
        "configurations": {c: r.as_dict() for c, r in results.items()},
    }


def _metrics(summary: dict[str, Any]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for configuration, result in summary["configurations"].items():
        for kind in ("id_numbers", "name_pairs", "name_token_occurrences"):
            tally = result[kind]
            rate = tally["leaked"] / tally["present"] if tally["present"] else 0.0
            metrics[f"redaction.{configuration}.{kind}_leak_rate"] = rate
    return metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Measure what the redactor lets through.")
    parser.add_argument("--split", choices=["dev", "holdout", "all"], default="all")
    parser.add_argument("--data", type=Path, default=Path("evals/data"))
    parser.add_argument("--out-dir", type=Path, default=Path("evals/reports"))
    args = parser.parse_args(argv)

    splits = json.loads((args.data / "splits.json").read_text(encoding="utf-8"))
    packet_ids = (
        sorted(splits["dev"] + splits["holdout"]) if args.split == "all" else splits[args.split]
    )
    summary = check(args.data, packet_ids)
    info = {"packet_split": args.split, "tesseract": str(pytesseract.get_tesseract_version())}
    path = write_report(
        {"redaction": StageResult(_metrics(summary), summary)}, "redaction", args.out_dir, info=info
    )
    for configuration, result in summary["configurations"].items():
        shown = {
            kind: f"{result[kind]['leaked']}/{result[kind]['present']}"
            for kind in ("id_numbers", "name_pairs", "name_token_occurrences")
        }
        print(
            f"{configuration:<18} ids {shown['id_numbers']:>7}  "
            f"name pairs {shown['name_pairs']:>7}  "
            f"tokens {shown['name_token_occurrences']:>7}"
        )
    print(f"report: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
