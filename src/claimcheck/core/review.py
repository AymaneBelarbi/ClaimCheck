"""The decision-ready review file: every document, field and rule of a claim with its verdict.

It is an aggregation and nothing more. A rule that did not end `found` blocks the decision and
is listed by id, so an adjuster sees what to check before anything else. Only a claim in
`needs_review` with rule results can be ready: a claim that failed, is still being processed or
was already decided has nothing to decide on.
"""

import uuid
from collections.abc import Iterable
from datetime import date

from pydantic import BaseModel, ConfigDict

from claimcheck.core.types import ClaimStatus, ClaimType, DocType, Evidence, Verdict


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True)


class ReviewClaim(_Frozen):
    id: uuid.UUID
    packet_id: str
    status: ClaimStatus
    claim_type: ClaimType
    incident_date: date


class ReviewPolicyVersion(_Frozen):
    version_no: int
    effective_from: date
    effective_to: date | None


class ReviewDocument(_Frozen):
    id: uuid.UUID
    filename: str
    # None until the pipeline has run.
    doc_type: DocType | None
    classification_confidence: float | None
    classification_method: str | None
    ocr_mean_confidence: float | None
    low_confidence: bool
    error: str | None


class ReviewField(_Frozen):
    document_id: uuid.UUID
    doc_type: DocType | None
    field_name: str
    value: str | None
    confidence: float
    verdict: Verdict
    evidence: Evidence | None


class ReviewRule(_Frozen):
    rule_id: str
    description: str
    verdict: Verdict
    message: str
    evidence: list[Evidence]


class ReviewSummary(_Frozen):
    # Rule verdicts; the decision depends on these.
    verdict_counts: dict[Verdict, int]
    field_verdict_counts: dict[Verdict, int]
    # Rule ids whose verdict is not `found`.
    blocking: list[str]
    ready_for_decision: bool


class ReviewFile(_Frozen):
    claim: ReviewClaim
    policy_version: ReviewPolicyVersion | None
    documents: list[ReviewDocument]
    fields: list[ReviewField]
    rules: list[ReviewRule]
    summary: ReviewSummary


def summarize(
    status: ClaimStatus, fields: list[ReviewField], rules: list[ReviewRule]
) -> ReviewSummary:
    blocking = [r.rule_id for r in rules if r.verdict is not Verdict.FOUND]
    return ReviewSummary(
        verdict_counts=_counts(r.verdict for r in rules),
        field_verdict_counts=_counts(f.verdict for f in fields),
        blocking=blocking,
        ready_for_decision=(status is ClaimStatus.NEEDS_REVIEW and bool(rules) and not blocking),
    )


def _counts(verdicts: Iterable[Verdict]) -> dict[Verdict, int]:
    counts = dict.fromkeys(Verdict, 0)
    for verdict in verdicts:
        counts[verdict] += 1
    return counts
