"""Shared vocabulary. Every package imports from here; this module imports none of them."""

from dataclasses import dataclass
from enum import StrEnum


class Verdict(StrEnum):
    """The only outcomes a field or a rule can end in."""

    FOUND = "found"
    MISSING = "missing"
    CONFLICTING = "conflicting"
    LOW_CONFIDENCE = "low_confidence"


class DocType(StrEnum):
    CLAIM_FORM = "claim_form"
    ID_CARD = "id_card"
    INVOICE = "invoice"
    INCIDENT_REPORT = "incident_report"
    OTHER = "other"  # goes to human review, never to extraction


class ClaimType(StrEnum):
    COLLISION = "collision"
    GLASS = "glass"
    THEFT = "theft"


class ClaimStatus(StrEnum):
    RECEIVED = "received"
    PROCESSING = "processing"
    NEEDS_REVIEW = "needs_review"
    APPROVED_FOR_REVIEW = "approved_for_review"
    REJECTED_INCOMPLETE = "rejected_incomplete"
    FAILED = "failed"


class Role(StrEnum):
    ADJUSTER = "adjuster"
    CUSTOMER = "customer"


@dataclass(frozen=True)
class Evidence:
    """Where a verdict came from: document, page, and the character span in its OCR text."""

    document_id: str
    page: int | None
    span: tuple[int, int] | None
    text: str | None
