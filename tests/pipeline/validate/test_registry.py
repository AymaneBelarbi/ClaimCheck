from collections.abc import Callable

from core.types import ClaimType, DocType, Verdict
from pipeline.validate import ALL_RULES, run_rules
from pipeline.validate.base import ClaimContext
from tests.pipeline.validate.factories import passing_documents, without

Make = Callable[..., ClaimContext]

IDS = {
    "R1_NAME_MATCH",
    "R2_INCIDENT_IN_COVERAGE",
    "R3_INVOICE_WITHIN_LIMIT",
    "R4_INVOICE_AFTER_INCIDENT",
    "R5_ID_VALID_ON_INCIDENT",
    "R6_REQUIRED_DOCUMENTS",
    "R7_INCIDENT_DATE_CONSISTENT",
}


def test_every_rule_registered_once() -> None:
    assert [r.rule_id for r in ALL_RULES] == sorted(IDS)
    assert {r.rule_id for r in ALL_RULES} == IDS


def test_every_rule_describes_itself() -> None:
    assert all(r.description and r.applies_to for r in ALL_RULES)


def test_collision_runs_every_rule(make_ctx: Make) -> None:
    results = run_rules(make_ctx())

    assert {r.rule_id for r in results} == IDS
    assert all(r.verdict is Verdict.FOUND for r in results)


def test_theft_runs_neither_invoice_rule(make_ctx: Make) -> None:
    ctx = make_ctx(
        claim_type=ClaimType.THEFT, documents=without(passing_documents(), DocType.INVOICE)
    )

    results = run_rules(ctx)

    assert {r.rule_id for r in results} == IDS - {
        "R3_INVOICE_WITHIN_LIMIT",
        "R4_INVOICE_AFTER_INCIDENT",
    }


def test_results_follow_rule_order(make_ctx: Make) -> None:
    ids = [r.rule_id for r in run_rules(make_ctx())]

    assert ids == sorted(ids)
