from core.types import ClaimType, DocType
from pipeline.validate.base import ClaimContext, RuleResult, parse_date, read_inputs, unreadable


class IncidentDateConsistent:
    """The declared incident date picks the policy version, so the documents must back it.

    The claim's incident date comes from the upload, not from a document. The claim form must
    state the same date; an incident report, which not every claim type has, is compared when
    the packet contains one.
    """

    rule_id = "R7_INCIDENT_DATE_CONSISTENT"
    description = (
        "The incident date on the claim form, and on the incident report when there is one, "
        "matches the claim's incident date."
    )
    applies_to = frozenset(ClaimType)

    def evaluate(self, ctx: ClaimContext) -> RuleResult:
        specs = [(DocType.CLAIM_FORM, "incident_date")]
        if any(d.doc_type is DocType.INCIDENT_REPORT for d in ctx.documents):
            specs.append((DocType.INCIDENT_REPORT, "incident_date"))
        inputs = read_inputs(ctx, *specs)
        if (blocked := inputs.blocked(self.rule_id)) is not None:
            return blocked
        differing: list[str] = []
        for doc_type, name in specs:
            stated = parse_date(inputs.get(doc_type, name))
            if stated is None:
                return unreadable(self.rule_id, inputs, f"{doc_type.value}.{name}")
            if stated != ctx.incident_date:
                differing.append(f"{doc_type.value} says {stated}")
        if differing:
            return inputs.result(
                self.rule_id,
                False,
                f"The claim's incident date is {ctx.incident_date}, but the "
                + "; the ".join(differing),
            )
        sources = " and ".join(doc_type.value for doc_type, _ in specs)
        return inputs.result(
            self.rule_id, True, f"Incident date {ctx.incident_date} matches the {sources}"
        )
