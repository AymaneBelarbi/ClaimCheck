"""Ground-truth schema and the fixed vocabulary of the synthetic dataset.

Everything an eval runner or the seeder needs to read the generated files lives here, so they
never have to import the generator itself.
"""

from datetime import date
from enum import StrEnum

from pydantic import BaseModel, ConfigDict

from claimcheck.core.types import ClaimType, DocType, Verdict

# Driver licence numbers: "D" followed by eight digits. Redaction relies on this exact shape.
ID_NUMBER_PATTERN = r"\bD\d{8}\b"
POLICY_NUMBER_PATTERN = r"\bAUT-\d{6}\b"

RULE_IDS: tuple[str, ...] = (
    "R1_NAME_MATCH",
    "R2_INCIDENT_IN_COVERAGE",
    "R3_INVOICE_WITHIN_LIMIT",
    "R4_INVOICE_AFTER_INCIDENT",
    "R5_ID_VALID_ON_INCIDENT",
    "R6_REQUIRED_DOCUMENTS",
    "R7_INCIDENT_DATE_CONSISTENT",
)
# R3 and R4 read the repair invoice, which a theft claim does not have.
INVOICE_RULES = frozenset({"R3_INVOICE_WITHIN_LIMIT", "R4_INVOICE_AFTER_INCIDENT"})

NAME_MATCH_THRESHOLD = 90.0

REQUIRED_DOCUMENTS: dict[ClaimType, frozenset[DocType]] = {
    ClaimType.COLLISION: frozenset(
        {DocType.CLAIM_FORM, DocType.ID_CARD, DocType.INVOICE, DocType.INCIDENT_REPORT}
    ),
    ClaimType.GLASS: frozenset({DocType.CLAIM_FORM, DocType.ID_CARD, DocType.INVOICE}),
    ClaimType.THEFT: frozenset({DocType.CLAIM_FORM, DocType.ID_CARD, DocType.INCIDENT_REPORT}),
}

FIELD_NAMES: dict[DocType, tuple[str, ...]] = {
    DocType.CLAIM_FORM: (
        "claimant_name",
        "policy_number",
        "incident_date",
        "incident_type",
        "vehicle_plate",
        "description",
    ),
    DocType.ID_CARD: ("full_name", "id_number", "date_of_birth", "expiry_date"),
    DocType.INVOICE: ("vendor", "invoice_date", "line_items", "total"),
    DocType.INCIDENT_REPORT: ("report_number", "incident_date", "location", "description"),
}


class DatasetError(Exception):
    """The generator built something that does not test what it claims to test."""


def rules_for(claim_type: ClaimType) -> tuple[str, ...]:
    if claim_type == ClaimType.THEFT:
        return tuple(r for r in RULE_IDS if r not in INVOICE_RULES)
    return RULE_IDS


class Scenario(StrEnum):
    CLEAN = "clean"
    NAME_MISMATCH = "name_mismatch"
    INCIDENT_OUTSIDE_COVERAGE = "incident_outside_coverage"
    ID_EXPIRED = "id_expired"
    MISSING_DOCUMENT = "missing_document"
    UNREADABLE_SCAN = "unreadable_scan"
    TOTAL_OVER_LIMIT = "total_over_limit"
    INVOICE_BEFORE_INCIDENT = "invoice_before_incident"
    DUPLICATE_CONFLICTING_INVOICE = "duplicate_conflicting_invoice"


class GTDocument(BaseModel):
    """One file in a packet and what a perfect extractor would return for it.

    Values are normalized: dates `YYYY-MM-DD`, amounts as 2-dp strings, `line_items` as a JSON
    string of `[{"description": ..., "amount": "..."}]`, names exactly as printed.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    filename: str
    doc_type: DocType
    fields: dict[str, str | None]
    field_verdicts: dict[str, Verdict]


class GroundTruth(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    packet_id: str
    policy_number: str
    holder_id: str
    claim_type: ClaimType
    incident_date: date
    scenario: Scenario
    # Which sub-case of the scenario this packet exercises, e.g. "missing:invoice".
    variant: str | None
    documents: list[GTDocument]
    rule_verdicts: dict[str, Verdict]


class ClauseQuestion(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    policy_number: str
    incident_date: date
    question: str
    expected_clause_ref: str
    expected_version_no: int


class AgentQuestion(BaseModel):
    """An adjuster question about one dev packet's claim, for the agent eval.

    `expected_clause_ref` is the clause of the version in force on the packet's incident date
    that answers it, or None for a question about the claim's own data.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    packet_id: str
    question: str
    expected_clause_ref: str | None


class PolicyVersionEntry(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    version_no: int
    effective_from: date
    # None means the version is still current.
    effective_to: date | None
    # Relative to the policies directory, e.g. "AUT-104233/v1.md".
    file: str


class PolicyEntry(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    policy_number: str
    holder_id: str
    versions: list[PolicyVersionEntry]


class PolicyIndex(BaseModel):
    """`policies/index.json`: everything the seeder needs without parsing markdown."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    policies: list[PolicyEntry]
