from collections.abc import Callable
from datetime import date

from claimcheck.core.types import Verdict
from claimcheck.pipeline.validate.base import ClaimContext, PolicyVersionInfo
from claimcheck.pipeline.validate.incident_in_coverage import IncidentInCoverage

Make = Callable[..., ClaimContext]


def test_version_in_force_found(make_ctx: Make) -> None:
    ctx = make_ctx(policy_version=PolicyVersionInfo(2, date(2026, 1, 1), None))

    result = IncidentInCoverage().evaluate(ctx)

    assert result.rule_id == "R2_INCIDENT_IN_COVERAGE"
    assert result.verdict is Verdict.FOUND
    assert "version 2" in result.message


def test_no_version_in_force_conflicting(make_ctx: Make) -> None:
    result = IncidentInCoverage().evaluate(make_ctx(policy_version=None))

    assert result.verdict is Verdict.CONFLICTING
    assert "2026-03-10" in result.message
