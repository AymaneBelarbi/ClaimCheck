from claimcheck.core.types import ClaimType, DocType
from claimcheck.pipeline.validate.base import (
    ClaimContext,
    RuleResult,
    parse_amount,
    read_inputs,
    unreadable,
)


class InvoiceWithinLimit:
    rule_id = "R3_INVOICE_WITHIN_LIMIT"
    description = "The invoice total is within the coverage limit for the claim type."
    applies_to = frozenset({ClaimType.COLLISION, ClaimType.GLASS})

    def evaluate(self, ctx: ClaimContext) -> RuleResult:
        absent = (
            [] if ctx.coverage_limit is not None else ["coverage limit for this policy version"]
        )
        inputs = read_inputs(ctx, (DocType.INVOICE, "total"), absent=absent)
        if (blocked := inputs.blocked(self.rule_id)) is not None:
            return blocked
        limit = ctx.coverage_limit
        total = parse_amount(inputs.get(DocType.INVOICE, "total"))
        if total is None or limit is None:
            return unreadable(self.rule_id, inputs, "invoice.total")
        if total <= limit:
            return inputs.result(
                self.rule_id,
                True,
                f"Invoice total {total:,.2f} is within the limit of {limit:,.2f}",
            )
        return inputs.result(
            self.rule_id,
            False,
            f"Invoice total {total:,.2f} exceeds the limit of {limit:,.2f} by {total - limit:,.2f}",
        )
