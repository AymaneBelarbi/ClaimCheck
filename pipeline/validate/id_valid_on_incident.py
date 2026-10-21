from core.types import ClaimType, DocType
from pipeline.validate.base import ClaimContext, RuleResult, parse_date, read_inputs, unreadable


class IdValidOnIncident:
    rule_id = "R5_ID_VALID_ON_INCIDENT"
    description = "The ID card had not expired on the incident date."
    applies_to = frozenset(ClaimType)

    def evaluate(self, ctx: ClaimContext) -> RuleResult:
        inputs = read_inputs(ctx, (DocType.ID_CARD, "expiry_date"))
        if (blocked := inputs.blocked(self.rule_id)) is not None:
            return blocked
        expiry = parse_date(inputs.get(DocType.ID_CARD, "expiry_date"))
        if expiry is None:
            return unreadable(self.rule_id, inputs, "id_card.expiry_date")
        if expiry >= ctx.incident_date:
            return inputs.result(
                self.rule_id,
                True,
                f"ID expires {expiry}, valid on the incident date {ctx.incident_date}",
            )
        return inputs.result(
            self.rule_id,
            False,
            f"ID expired {expiry}, before the incident on {ctx.incident_date}",
        )
