from claimcheck.core.types import ClaimType, DocType
from claimcheck.pipeline.validate.base import (
    ClaimContext,
    RuleResult,
    parse_date,
    read_inputs,
    unreadable,
)


class InvoiceAfterIncident:
    rule_id = "R4_INVOICE_AFTER_INCIDENT"
    description = "The invoice is dated on or after the incident."
    applies_to = frozenset({ClaimType.COLLISION, ClaimType.GLASS})

    def evaluate(self, ctx: ClaimContext) -> RuleResult:
        inputs = read_inputs(ctx, (DocType.INVOICE, "invoice_date"))
        if (blocked := inputs.blocked(self.rule_id)) is not None:
            return blocked
        issued = parse_date(inputs.get(DocType.INVOICE, "invoice_date"))
        if issued is None:
            return unreadable(self.rule_id, inputs, "invoice.invoice_date")
        if issued >= ctx.incident_date:
            return inputs.result(
                self.rule_id,
                True,
                f"Invoice dated {issued}, not before the incident on {ctx.incident_date}",
            )
        return inputs.result(
            self.rule_id,
            False,
            f"Invoice dated {issued} is before the incident on {ctx.incident_date}",
        )
