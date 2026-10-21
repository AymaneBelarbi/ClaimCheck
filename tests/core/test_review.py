import pytest

from claimcheck.core.review import ReviewRule, summarize
from claimcheck.core.types import ClaimStatus, Verdict


def _rule(rule_id: str, verdict: Verdict) -> ReviewRule:
    return ReviewRule(rule_id=rule_id, description="", verdict=verdict, message="", evidence=[])


def test_a_claim_in_review_with_every_rule_found_is_ready() -> None:
    summary = summarize(ClaimStatus.NEEDS_REVIEW, [], [_rule("R1_NAME_MATCH", Verdict.FOUND)])

    assert summary.blocking == []
    assert summary.ready_for_decision is True


def test_a_blocking_rule_means_not_ready() -> None:
    summary = summarize(
        ClaimStatus.NEEDS_REVIEW, [], [_rule("R3_INVOICE_WITHIN_LIMIT", Verdict.CONFLICTING)]
    )

    assert summary.ready_for_decision is False


def test_a_claim_in_review_with_no_rule_results_is_not_ready() -> None:
    assert summarize(ClaimStatus.NEEDS_REVIEW, [], []).ready_for_decision is False


@pytest.mark.parametrize(
    "status",
    [
        ClaimStatus.RECEIVED,
        ClaimStatus.PROCESSING,
        ClaimStatus.FAILED,
        ClaimStatus.APPROVED_FOR_REVIEW,
        ClaimStatus.REJECTED_INCOMPLETE,
    ],
)
def test_only_a_claim_in_review_can_be_ready(status: ClaimStatus) -> None:
    summary = summarize(status, [], [_rule("R1_NAME_MATCH", Verdict.FOUND)])

    assert summary.blocking == []
    assert summary.ready_for_decision is False
