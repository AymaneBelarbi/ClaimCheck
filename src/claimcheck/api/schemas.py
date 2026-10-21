"""Request and response bodies. The review file itself is `claimcheck.core.review.ReviewFile`."""

import uuid
from datetime import date, datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field

from claimcheck.core.types import ClaimStatus, ClaimType, DocType


class ClaimCreated(BaseModel):
    claim_id: uuid.UUID
    status: ClaimStatus


class ClaimSummary(BaseModel):
    claim_id: uuid.UUID
    packet_id: str
    status: ClaimStatus
    claim_type: ClaimType
    incident_date: date
    created_at: datetime


class DocumentOut(BaseModel):
    id: uuid.UUID
    filename: str
    # None until the pipeline has classified the document.
    doc_type: DocType | None
    classification_confidence: float | None
    low_confidence: bool
    error: str | None


class ClaimDetail(ClaimSummary):
    policy_number: str
    documents: list[DocumentOut]


class Decision(StrEnum):
    APPROVED_FOR_REVIEW = ClaimStatus.APPROVED_FOR_REVIEW.value
    REJECTED_INCOMPLETE = ClaimStatus.REJECTED_INCOMPLETE.value


class DecisionIn(BaseModel):
    decision: Decision
    reason: str = Field(min_length=1, max_length=2000)


class DecisionOut(BaseModel):
    claim_id: uuid.UUID
    status: ClaimStatus


class Health(BaseModel):
    status: Literal["ok", "unavailable"]
    db: bool


class AskIn(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


class AskOut(BaseModel):
    """The agent's answer. Tool arguments and results stay in the audit log, not here."""

    answer: str
    # Clause refs the policy search returned and the answer relies on.
    citations: list[str]
    # True when the answer rests on no retrieved clause, or cited one that was not retrieved.
    uncited: bool
    tools_called: list[str]
