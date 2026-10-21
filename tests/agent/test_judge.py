from typing import Literal

import pytest

from agent.judge import JudgeVerdict, StatementCheck, StatementChecks, judge
from agent.loop import AgentAnswer
from rag.retrieve import RetrievedClause
from tests.fakes import FakeGateway

CLAUSE = RetrievedClause(
    clause_ref="4.2",
    title="Glass breakage",
    text="We cover glass breakage up to a limit of $1,000 per claim.",
    version_no=2,
    score=0.8,
)


def _answer(text: str, citations: list[str]) -> AgentAnswer:
    return AgentAnswer(answer=text, citations=citations, tool_calls=[], uncited=not citations)


def _check(
    statement: str,
    *,
    about: Literal["policy", "claim"] = "policy",
    quote: str = "",
    supported: bool = True,
) -> StatementCheck:
    return StatementCheck(
        statement=statement,
        about=about,
        quote=quote,
        supported=supported,
    )


def _judge(
    *checks: StatementCheck,
    clauses: list[RetrievedClause] | None = None,
    citations: tuple[str, ...] = ("4.2",),
    policy_question: bool = False,
) -> JudgeVerdict:
    gateway = FakeGateway(structured_outputs=[StatementChecks(statements=list(checks))])
    return judge(
        "Is glass covered?",
        _answer("Yes, up to $1,000.", list(citations)),
        [CLAUSE] if clauses is None else clauses,
        gateway,
        policy_question=policy_question,
    )


def test_judge_sends_question_answer_and_clause_text() -> None:
    gateway = FakeGateway(structured_outputs=[StatementChecks(statements=[])])

    judge(
        "Is glass covered?",
        _answer("Yes, up to $1,000.", ["4.2"]),
        [CLAUSE],
        gateway,
        cache_key="judge:Q-01",
    )

    (call,) = gateway.calls
    assert call.schema is StatementChecks
    assert call.cache_key == "judge:Q-01"
    assert call.prompt is not None
    assert "Is glass covered?" in call.prompt
    assert "Yes, up to $1,000." in call.prompt
    assert "4.2 Glass breakage" in call.prompt
    assert CLAUSE.text in call.prompt
    assert "supported" in call.system


def test_supported_statements_with_real_quotes_are_faithful() -> None:
    verdict = _judge(
        _check("Glass is covered", quote="We cover glass breakage"),
        _check("up to $1,000 per claim", quote="up to a limit of  $1,000 per claim."),
    )

    assert verdict == JudgeVerdict(faithful=True, unsupported_statements=[])


def test_a_statement_the_judge_marks_unsupported_is_listed() -> None:
    verdict = _judge(
        _check("Glass is covered", quote="We cover glass breakage"),
        _check("the glass excess is $100", supported=False),
    )

    assert verdict.faithful is False
    assert verdict.unsupported_statements == ["the glass excess is $100"]


def test_support_needs_a_quote_that_is_in_the_clauses() -> None:
    verdict = _judge(_check("the glass excess is $100", quote="The glass excess is $100."))

    assert verdict.faithful is False
    assert verdict.unsupported_statements == ["the glass excess is $100"]


def test_claim_statements_of_an_answer_without_citations_are_not_checked() -> None:
    verdict = _judge(
        _check("The invoice is dated before the incident", about="claim"), citations=()
    )

    assert verdict.faithful is True


def test_an_answer_without_citations_and_without_statements_is_faithful() -> None:
    assert _judge(citations=()).faithful is True


def test_an_answer_with_citations_has_every_statement_checked_whatever_its_label() -> None:
    verdict = _judge(
        _check("Glass is covered", quote="We cover glass breakage"),
        _check("Theft is covered too", about="claim"),
    )

    assert verdict.faithful is False
    assert verdict.unsupported_statements == ["Theft is covered too"]


def test_a_policy_question_has_every_statement_checked_even_without_citations() -> None:
    verdict = _judge(_check("Glass is covered", about="claim"), citations=(), policy_question=True)

    assert verdict.faithful is False


@pytest.mark.parametrize(("citations", "policy_question"), [(("4.2",), False), ((), True)])
def test_a_checked_answer_with_no_statements_is_unfaithful(
    citations: tuple[str, ...], policy_question: bool
) -> None:
    verdict = _judge(citations=citations, policy_question=policy_question)

    assert verdict.faithful is False
    assert verdict.unsupported_statements == ["Yes, up to $1,000."]


def test_policy_statements_without_clauses_are_unsupported() -> None:
    verdict = _judge(_check("Glass is covered", quote="We cover glass breakage"), clauses=[])

    assert verdict.faithful is False


def test_judge_says_when_there_are_no_clauses() -> None:
    gateway = FakeGateway(structured_outputs=[StatementChecks(statements=[])])

    judge("Is glass covered?", _answer("I cannot tell.", []), [], gateway)

    assert gateway.calls[0].prompt is not None
    assert "(no clauses" in gateway.calls[0].prompt


NO_VERSION = "no policy version was in force on the incident date, so no clause applies"


def _no_version_answer(text: str) -> AgentAnswer:
    return AgentAnswer(
        answer=text, citations=[], tool_calls=[], uncited=True, policy_notes=[NO_VERSION]
    )


def test_judge_is_shown_the_no_version_note() -> None:
    gateway = FakeGateway(structured_outputs=[StatementChecks(statements=[])])

    judge("Is collision covered?", _no_version_answer("No version applied."), [], gateway)

    prompt = gateway.calls[0].prompt
    assert prompt is not None
    assert NO_VERSION in prompt
    assert "quote the note's exact words" in prompt


def test_a_statement_that_repeats_the_no_version_note_is_supported() -> None:
    # Labelled "policy", as the judge does, it used to be checked against no evidence at all.
    gateway = FakeGateway(
        structured_outputs=[
            StatementChecks(
                statements=[
                    _check(
                        "No policy version was in force on the incident date.",
                        quote="no policy version was in force on the incident date",
                    ),
                    _check("No clause applies.", quote="so no clause applies"),
                ]
            )
        ]
    )

    verdict = judge(
        "Which policy version applies?",
        _no_version_answer("No policy version was in force, so no clause applies."),
        [],
        gateway,
    )

    assert verdict == JudgeVerdict(faithful=True, unsupported_statements=[])


def test_the_no_version_note_does_not_support_a_policy_term() -> None:
    gateway = FakeGateway(
        structured_outputs=[
            StatementChecks(
                statements=[
                    _check(
                        "The collision excess is $500.",
                        quote="no policy version was in force on the incident date",
                    )
                ]
            )
        ]
    )

    verdict = judge("What is the excess?", _no_version_answer("$500."), [], gateway)

    assert verdict.faithful is False


def test_a_quote_may_keep_the_clause_header() -> None:
    verdict = _judge(
        _check("Glass is covered", quote="[4.2 Glass breakage] We cover glass breakage")
    )

    assert verdict.faithful is True


def test_every_amount_in_the_statement_must_be_in_the_quote() -> None:
    verdict = _judge(
        _check("Covered up to $1,000.00 per claim", quote="up to a limit of $1,000 per claim"),
        _check("less an excess of $500", quote="We cover glass breakage up to a limit"),
    )

    assert verdict.faithful is False
    assert verdict.unsupported_statements == ["less an excess of $500"]


def test_durations_count_as_amounts() -> None:
    clause = RetrievedClause(
        "4.3", "Theft", "Report the theft within 48 hours of discovery.", 1, 0.9
    )

    verdict = _judge(
        _check("Report within 24 hours", quote="Report the theft within 48 hours"),
        clauses=[clause],
    )

    assert verdict.faithful is False


def test_a_quote_may_skip_from_the_header_to_a_later_sentence() -> None:
    clause = RetrievedClause(
        "6.3",
        "Windscreen",
        "Glass alone is not covered. Glass broken in a collision is paid.",
        1,
        0.9,
    )

    verdict = _judge(
        _check(
            "Glass broken in a collision is paid",
            quote="[6.3 Windscreen] Glass broken in a collision is paid",
        ),
        clauses=[clause],
    )

    assert verdict.faithful is True


EXCESS = RetrievedClause(
    clause_ref="5.1",
    title="Excess",
    text=(
        "You pay the following excess on each claim: collision claims $700; glass claims $90; "
        "theft claims $800. We deduct the excess from the amount we pay."
    ),
    version_no=1,
    score=0.9,
)


def test_a_quote_that_leaves_out_part_of_a_list_still_counts() -> None:
    # Skipping the collision entry put the whole-quote fuzzy match under its threshold.
    verdict = _judge(
        _check(
            "The glass excess is $90.",
            quote="You pay the following excess on each claim: glass claims $90;",
        ),
        clauses=[EXCESS],
        citations=("5.1",),
    )

    assert verdict == JudgeVerdict(faithful=True, unsupported_statements=[])


def test_each_part_of_a_shortened_quote_must_be_verbatim() -> None:
    verdict = _judge(
        _check(
            "The glass excess is $900.",
            quote="You pay the following excess on each claim: glass claims $900;",
        ),
        clauses=[EXCESS],
        citations=("5.1",),
    )

    assert verdict.faithful is False


def test_the_parts_of_a_shortened_quote_must_come_from_one_clause() -> None:
    verdict = _judge(
        _check(
            "Glass is covered with a $90 excess.",
            quote="We cover glass breakage up to a limit: glass claims $90",
        ),
        clauses=[CLAUSE, EXCESS],
        citations=("4.2", "5.1"),
    )

    assert verdict.faithful is False


def test_the_parts_of_a_shortened_quote_must_keep_the_clause_order() -> None:
    verdict = _judge(
        _check(
            "Theft claims carry an excess of $90.",
            quote="theft claims $800; glass claims $90",
        ),
        clauses=[EXCESS],
        citations=("5.1",),
    )

    assert verdict.faithful is False


def test_a_shortened_quote_needs_three_words_in_every_part() -> None:
    # "$90" alone could be lifted from anywhere in the clause.
    verdict = _judge(
        _check(
            "The collision excess is $90.",
            quote="You pay the following excess on each claim: $90",
        ),
        clauses=[EXCESS],
        citations=("5.1",),
    )

    assert verdict.faithful is False


def test_a_loosely_matched_quote_cannot_change_an_amount() -> None:
    # One amount changed in a long quote still scores above the fuzzy threshold.
    verdict = _judge(
        _check(
            "The collision excess is $90.",
            quote="You pay the following excess on each claim: collision claims $90;",
        ),
        clauses=[EXCESS],
        citations=("5.1",),
    )

    assert verdict.faithful is False


def test_a_loosely_matched_quote_with_the_right_amount_still_counts() -> None:
    verdict = _judge(
        _check(
            "The collision excess is $700.",
            quote="You pay the following excess on every claim: collision claims $700",
        ),
        clauses=[EXCESS],
        citations=("5.1",),
    )

    assert verdict.faithful is True


def test_a_shortened_quote_cannot_skip_a_sentence() -> None:
    # Skipping the exclusion sentence would turn it into its opposite.
    clause = RetrievedClause(
        "4.3",
        "Theft",
        "We cover theft of the vehicle. We do not cover theft if the doors were left unlocked. "
        "Report the theft within 2 days.",
        1,
        0.9,
    )

    verdict = _judge(
        _check(
            "Theft is covered if the doors were left unlocked.",
            quote="We cover theft of the vehicle: if the doors were left unlocked",
        ),
        clauses=[clause],
        citations=("4.3",),
    )

    assert verdict.faithful is False


@pytest.mark.parametrize(
    "quote",
    [
        "You pay the following excess on each claim: collision claims $70",
        "You pay the following excess on each claim: collision claims $7",
        "You pay the following excess on each claim: glass claims $9",
    ],
)
def test_a_quote_cannot_cut_an_amount_short(quote: str) -> None:
    amount = quote.rsplit(" ", 1)[1]
    verdict = _judge(
        _check(f"The excess is {amount}.", quote=quote),
        clauses=[EXCESS],
        citations=("5.1",),
    )

    assert verdict.faithful is False


def test_a_duration_cannot_be_cut_short() -> None:
    clause = RetrievedClause(
        "4.3", "Theft", "Report the theft within 48 hours of discovery.", 1, 0.9
    )

    verdict = _judge(
        _check("Report within 8 hours", quote="the theft within 8 hours of discovery"),
        clauses=[clause],
    )

    assert verdict.faithful is False
