"""Shared types for the validation rules, and the one place input precedence is decided.

Every rule reads some fields from some documents and ends in one verdict. Whatever the rule,
inputs are checked in the same order: an absent input is `missing`, then a `low_confidence`
input is `low_confidence`, then documents of the same type that disagree (or an input already
marked `conflicting`) are `conflicting`. Only when all inputs are clean does the rule compare
them. `read_inputs` implements that order so no rule has to.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation
from itertools import combinations
from typing import Protocol

from claimcheck.core.types import ClaimType, DocType, Evidence, Verdict
from claimcheck.pipeline.extract import FieldResult


@dataclass(frozen=True)
class PolicyVersionInfo:
    version_no: int
    effective_from: date
    effective_to: date | None


@dataclass(frozen=True)
class DocFields:
    document_id: str
    doc_type: DocType
    fields: dict[str, FieldResult]


@dataclass(frozen=True)
class RuleResult:
    rule_id: str
    verdict: Verdict
    message: str
    evidence: list[Evidence]


@dataclass(frozen=True)
class ClaimContext:
    claim_type: ClaimType
    incident_date: date
    documents: list[DocFields]
    # Resolved by the caller with policy_in_force on the claim's incident date.
    policy_version: PolicyVersionInfo | None
    # The limit for this claim type in that version; None if there is no version or no limit.
    coverage_limit: Decimal | None

    def values(self, doc_type: DocType, field: str) -> list[tuple[FieldResult, str]]:
        """The field from every document of `doc_type` that has it, with the document id."""
        return [
            (d.fields[field], d.document_id)
            for d in self.documents
            if d.doc_type is doc_type and field in d.fields
        ]


class Rule(Protocol):
    rule_id: str
    description: str
    applies_to: frozenset[ClaimType]

    def evaluate(self, ctx: ClaimContext) -> RuleResult: ...


Spec = tuple[DocType, str]


def _label(spec: Spec) -> str:
    return f"{spec[0].value}.{spec[1]}"


class Inputs:
    """The outcome of reading a rule's inputs: either a blocking verdict or agreed values."""

    def __init__(
        self,
        values: dict[Spec, str],
        evidence: list[Evidence],
        blocking: tuple[Verdict, str] | None,
    ) -> None:
        self._values = values
        self.evidence = evidence
        self._blocking = blocking

    def blocked(self, rule_id: str) -> RuleResult | None:
        """The rule's result if an input stops it from being evaluated, else None."""
        if self._blocking is None:
            return None
        verdict, message = self._blocking
        return RuleResult(rule_id, verdict, message, list(self.evidence))

    def get(self, doc_type: DocType, field: str) -> str:
        return self._values[(doc_type, field)]

    def result(self, rule_id: str, passed: bool, message: str) -> RuleResult:
        verdict = Verdict.FOUND if passed else Verdict.CONFLICTING
        return RuleResult(rule_id, verdict, message, list(self.evidence))


def read_inputs(
    ctx: ClaimContext,
    *specs: Spec,
    absent: Sequence[str] = (),
    agree: Mapping[Spec, Callable[[str, str], bool]] | None = None,
) -> Inputs:
    """Apply the precedence order to `specs`.

    `absent` names inputs that do not come from documents (a coverage limit, say) and are
    already known to be missing. Documents of one type agree on an input when their values are
    equal; `agree` replaces that test for inputs that are free text, where the same value can be
    printed in different ways. Every pair of documents must agree.
    """
    same = agree or {}
    evidence: list[Evidence] = []
    missing: list[str] = list(absent)
    low: list[str] = []
    conflicting: list[str] = []
    values: dict[Spec, str] = {}

    for spec in specs:
        found = ctx.values(*spec)
        of_type = sum(1 for d in ctx.documents if d.doc_type is spec[0])
        evidence += [f.evidence for f, _ in found if f.evidence is not None]
        label = _label(spec)
        # A document of the right type that lacks the field counts as the field being absent.
        if not found or len(found) < of_type:
            missing.append(label)
            continue
        verdicts = [f.verdict for f, _ in found]
        if any(_is_absent(f) for f, _ in found):
            missing.append(label)
        elif Verdict.LOW_CONFIDENCE in verdicts:
            low.append(label)
        elif Verdict.CONFLICTING in verdicts:
            conflicting.append(f"{label} is marked conflicting")
        else:
            equivalent = same.get(spec, _equal)
            values_seen = [f.value or "" for f, _ in found]
            if not all(equivalent(a, b) for a, b in combinations(values_seen, 2)):
                shown = ", ".join(f"'{f.value}' ({doc_id})" for f, doc_id in found)
                conflicting.append(f"{label} differs between documents: {shown}")
            else:
                values[spec] = found[0][0].value or ""

    if missing:
        return Inputs(values, evidence, (Verdict.MISSING, "Missing: " + ", ".join(missing)))
    if low:
        return Inputs(
            values, evidence, (Verdict.LOW_CONFIDENCE, "Low confidence: " + ", ".join(low))
        )
    if conflicting:
        return Inputs(values, evidence, (Verdict.CONFLICTING, "; ".join(conflicting)))
    return Inputs(values, evidence, None)


def _equal(a: str, b: str) -> bool:
    return a == b


def _is_absent(result: FieldResult) -> bool:
    if result.verdict is Verdict.MISSING:
        return True
    # A low-confidence field often has no value because it could not be normalized; that is
    # a question for a person, not a missing field.
    return result.value is None and result.verdict is not Verdict.LOW_CONFIDENCE


def unreadable(rule_id: str, inputs: Inputs, what: str) -> RuleResult:
    """A value the extraction step let through that still cannot be parsed."""
    return RuleResult(
        rule_id, Verdict.LOW_CONFIDENCE, f"Cannot read {what} as a value", list(inputs.evidence)
    )


def parse_date(text: str) -> date | None:
    try:
        return date.fromisoformat(text)
    except ValueError:
        return None


def parse_amount(text: str) -> Decimal | None:
    try:
        value = Decimal(text)
    except InvalidOperation:
        return None
    return value if value.is_finite() else None
