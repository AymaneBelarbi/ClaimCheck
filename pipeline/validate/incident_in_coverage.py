from core.types import ClaimType, Verdict
from pipeline.validate.base import ClaimContext, RuleResult


class IncidentInCoverage:
    rule_id = "R2_INCIDENT_IN_COVERAGE"
    description = "The incident date falls inside a policy version that was in force."
    applies_to = frozenset(ClaimType)

    def evaluate(self, ctx: ClaimContext) -> RuleResult:
        version = ctx.policy_version
        if version is None:
            return RuleResult(
                self.rule_id,
                Verdict.CONFLICTING,
                f"No policy version was in force on the incident date {ctx.incident_date}",
                [],
            )
        until = version.effective_to.isoformat() if version.effective_to else "open-ended"
        return RuleResult(
            self.rule_id,
            Verdict.FOUND,
            f"Incident on {ctx.incident_date} is covered by version {version.version_no} "
            f"({version.effective_from} to {until})",
            [],
        )
