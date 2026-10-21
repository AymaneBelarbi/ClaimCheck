"""Builders for rule tests: a passing collision claim that each test then breaks one way."""

from dataclasses import replace
from datetime import date
from decimal import Decimal
from typing import Any

from core.types import ClaimType, DocType, Evidence, Verdict
from pipeline.extract import FieldResult
from pipeline.validate.base import ClaimContext, DocFields, PolicyVersionInfo

INCIDENT = date(2026, 3, 10)


def field(
    name: str,
    value: str | None,
    verdict: Verdict = Verdict.FOUND,
    *,
    document_id: str = "doc",
) -> FieldResult:
    return FieldResult(
        field_name=name,
        value=value,
        confidence=0.5 if verdict is Verdict.LOW_CONFIDENCE else 1.0,
        verdict=verdict,
        evidence=Evidence(document_id, 1, (0, 1), f"{name}={value}"),
        details=None,
    )


def doc(
    doc_type: DocType, document_id: str, **fields: str | tuple[str | None, Verdict] | None
) -> DocFields:
    """A field is a value, or a (value, verdict) pair when it should not be `found`."""
    built: dict[str, FieldResult] = {}
    for name, spec in fields.items():
        value, verdict = spec if isinstance(spec, tuple) else (spec, Verdict.FOUND)
        built[name] = field(name, value, verdict, document_id=document_id)
    return DocFields(document_id, doc_type, built)


def passing_documents() -> list[DocFields]:
    return [
        doc(
            DocType.CLAIM_FORM,
            "form",
            claimant_name="Jose Alvarez",
            incident_date=INCIDENT.isoformat(),
        ),
        doc(DocType.ID_CARD, "id", full_name="Jose Alvarez", expiry_date="2027-01-01"),
        doc(DocType.INVOICE, "inv", total="1000.00", invoice_date="2026-03-12"),
        doc(DocType.INCIDENT_REPORT, "report", incident_date=INCIDENT.isoformat()),
    ]


def make_ctx(**overrides: Any) -> ClaimContext:
    base = ClaimContext(
        claim_type=ClaimType.COLLISION,
        incident_date=INCIDENT,
        documents=passing_documents(),
        policy_version=PolicyVersionInfo(2, date(2026, 1, 1), None),
        coverage_limit=Decimal("15000"),
    )
    return replace(base, **overrides)


def without(documents: list[DocFields], *doc_types: DocType) -> list[DocFields]:
    return [d for d in documents if d.doc_type not in doc_types]
