"""Cross-document validation: a small deterministic rule per file, no model involved."""

from pipeline.validate.base import ClaimContext, Rule, RuleResult
from pipeline.validate.id_valid_on_incident import IdValidOnIncident
from pipeline.validate.incident_date_consistent import IncidentDateConsistent
from pipeline.validate.incident_in_coverage import IncidentInCoverage
from pipeline.validate.invoice_after_incident import InvoiceAfterIncident
from pipeline.validate.invoice_within_limit import InvoiceWithinLimit
from pipeline.validate.name_match import NameMatch
from pipeline.validate.required_documents import RequiredDocuments

ALL_RULES: tuple[Rule, ...] = (
    NameMatch(),
    IncidentInCoverage(),
    InvoiceWithinLimit(),
    InvoiceAfterIncident(),
    IdValidOnIncident(),
    RequiredDocuments(),
    IncidentDateConsistent(),
)


def run_rules(ctx: ClaimContext) -> list[RuleResult]:
    """Every rule that applies to the claim type, in rule-id order."""
    return [rule.evaluate(ctx) for rule in ALL_RULES if ctx.claim_type in rule.applies_to]


__all__ = ["ALL_RULES", "ClaimContext", "Rule", "RuleResult", "run_rules"]
