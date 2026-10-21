from claimcheck.core.types import ClaimType, DocType, Verdict
from claimcheck.pipeline.validate.base import ClaimContext, RuleResult

REQUIRED_DOCUMENTS: dict[ClaimType, tuple[DocType, ...]] = {
    ClaimType.COLLISION: (
        DocType.CLAIM_FORM,
        DocType.ID_CARD,
        DocType.INVOICE,
        DocType.INCIDENT_REPORT,
    ),
    ClaimType.GLASS: (DocType.CLAIM_FORM, DocType.ID_CARD, DocType.INVOICE),
    ClaimType.THEFT: (DocType.CLAIM_FORM, DocType.ID_CARD, DocType.INCIDENT_REPORT),
}


class RequiredDocuments:
    rule_id = "R6_REQUIRED_DOCUMENTS"
    description = "The packet contains every document type the claim type requires."
    applies_to = frozenset(ClaimType)

    def evaluate(self, ctx: ClaimContext) -> RuleResult:
        present = {d.doc_type for d in ctx.documents}
        absent = [t for t in REQUIRED_DOCUMENTS[ctx.claim_type] if t not in present]
        if absent:
            names = ", ".join(t.value for t in absent)
            return RuleResult(
                self.rule_id,
                Verdict.MISSING,
                f"Missing for a {ctx.claim_type.value} claim: {names}",
                [],
            )
        return RuleResult(
            self.rule_id,
            Verdict.FOUND,
            f"All documents required for a {ctx.claim_type.value} claim are present",
            [],
        )
