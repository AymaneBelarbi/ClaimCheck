"""Cross-document validation: a small deterministic rule per file, no model involved."""

from claimcheck.pipeline.validate.base import ClaimContext, Rule, RuleResult
from claimcheck.pipeline.validate.id_valid_on_incident import IdValidOnIncident
from claimcheck.pipeline.validate.incident_date_consistent import IncidentDateConsistent
from claimcheck.pipeline.validate.incident_in_coverage import IncidentInCoverage
from claimcheck.pipeline.validate.invoice_after_incident import InvoiceAfterIncident
from claimcheck.pipeline.validate.invoice_within_limit import InvoiceWithinLimit
from claimcheck.pipeline.validate.name_match import NameMatch
from claimcheck.pipeline.validate.required_documents import RequiredDocuments

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
