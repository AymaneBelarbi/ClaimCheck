"""LLM-as-judge for the agent's answers: is every policy statement backed by clause text?

The judge sees the question, the answer and the text of the clauses the answer cites, and
nothing else, so it cannot excuse a statement with what it knows about insurance in general.
The one exception is a claim with no policy version in force on its incident date: there are
no clauses to cite, and the note the policy tools returned instead ("no policy version was in
force on the incident date, so no clause applies") is shown and quoted like clause text.

A 7B judge asked for a bare yes/no was wrong in both directions on the first dev run: it
rejected answers that paraphrased their clause closely and accepted excess amounts that came
from a clause the answer never cited. So the judge works statement by statement instead. It
splits the answer into single facts, says whether each is about the policy or about the claim,
and for a policy fact copies the clause words that support it. The verdict is then computed
here: a policy statement counts as supported only if the judge said so, its quote really is in
the cited clause text, and every amount in the statement ($500, 48 hours) is in the quote. The
last check exists because the judge accepted "less the collision excess in section 5.1" as
support for "less the collision excess of $500".

A quote is matched loosely (a changed word or punctuation mark is forgiven), but not its
numbers: every amount in the quote must be a whole amount in the stretch of clause text it
matched. "You pay the following excess on each claim: $90" does not pass on a clause that lists
$90 for glass after a $700 collision excess, and "$70" does not pass as the start of "$700".

A quote may also leave out entries of a list: on "You pay the following excess on each claim:
collision claims $700; glass claims $90; ...", the quote "You pay the following excess on each
claim: glass claims $90" skips the collision entry, and a fuzzy match over the whole quote
scores it under the threshold. Such a quote is split only at colons and semicolons. It counts
when every part is verbatim, at least three words long, keeps its amounts whole, and is found
in order in one clause with no sentence end in the words it skips, so it cannot borrow a part
from another clause, carry a different amount, or step over a sentence such as an exclusion.

The policy/claim label is the judge's own, so it cannot be what decides whether a statement is
checked: the judge labelled every statement of some policy answers "claim". When the answer
cites clauses, or the caller says the question is about the policy, every statement is checked
whatever its label, and an answer with no statements to check is not faithful. Only an answer
that cites nothing to a question that is not about the policy (claim data) has its "claim"
statements skipped.

The verdict is one input to `agent.faithfulness`; the eval runner adds the checks that need no
model (were the citations retrieved, was there a search at all).
"""

import re
from typing import Literal

from pydantic import BaseModel
from rapidfuzz import fuzz

from agent.loop import AgentAnswer
from gateway.base import LLMGateway
from rag.retrieve import RetrievedClause

SYSTEM = """You check answers that an insurance claims assistant gave to an adjuster.

Split the answer into short statements, one fact each. For every statement:
- about: "policy" if it says what the policy covers, excludes, pays, limits, charges as excess,
  or requires. "claim" if it is about this claim's own documents, fields, dates, validation
  rules or status, or if it only says that no clause applies, no policy version was in force,
  or the question cannot be answered. Examples of claim statements: "No policy version was in
  force on the incident date.", "R4_INVOICE_AFTER_INCIDENT blocks the decision.", "The invoice
  total is $4,698.62."
- quote: for a policy statement, copy the exact words from the clauses below that support it,
  without changing them. Leave it empty if no clause text supports it.
- supported: true only if the quoted clause words say the same thing as the statement, with
  the same amounts and conditions. A close paraphrase is supported. An amount, condition or
  promise that the quoted words do not contain is not supported, even if it sounds plausible.
  For a claim statement, supported is true.

Judge only against the clause text given. Do not use outside knowledge."""

_AMOUNT = re.compile(
    r"\$\s?(\d[\d,]*(?:\.\d+)?)|(?<![\d.])(\d+)\s*(?=(?:hours?|days?|months?|years?|%))",
    re.IGNORECASE,
)
_HEADER = re.compile(r"\[[^\]]*\]")
# Where a quote may have left out list entries.
_GAP = re.compile(r"\s*[;:]\s*")
_SENTENCE_END = re.compile(r"[.!?](?:\s|$)")
# A shorter quote ("we cover") would support almost anything.
MIN_QUOTE_WORDS = 3
QUOTE_MATCH = 90.0


class StatementCheck(BaseModel):
    statement: str
    about: Literal["policy", "claim"]
    quote: str
    supported: bool


class StatementChecks(BaseModel):
    statements: list[StatementCheck]


class JudgeVerdict(BaseModel):
    faithful: bool
    unsupported_statements: list[str]


def judge(
    question: str,
    answer: AgentAnswer,
    clauses: list[RetrievedClause],
    gateway: LLMGateway,
    *,
    policy_question: bool = False,
    cache_key: str | None = None,
) -> JudgeVerdict:
    result = gateway.structured(
        system=SYSTEM,
        prompt=_prompt(question, answer, clauses),
        schema=StatementChecks,
        purpose="judge",
        cache_key=cache_key,
    )
    statements = result.output.statements
    check_all = policy_question or bool(answer.citations)
    if check_all and not statements:
        return JudgeVerdict(faithful=False, unsupported_statements=[answer.answer])
    blocks = [_normalize(_clause_block(c)) for c in clauses]
    blocks += [_normalize(note) for note in answer.policy_notes]
    unsupported = [
        check.statement
        for check in statements
        if (check_all or check.about == "policy") and not _supported(check, blocks)
    ]
    return JudgeVerdict(faithful=not unsupported, unsupported_statements=unsupported)


def _supported(check: StatementCheck, blocks: list[str]) -> bool:
    if not (check.supported and _quoted(check.quote, blocks)):
        return False
    return _amounts(check.statement) <= _amounts(check.quote)


def _amounts(text: str) -> set[str]:
    """Money and durations, normalized: "$1,000.00" and "$1000" are the same amount."""
    found = set()
    for match in _AMOUNT.finditer(text):
        money, duration = match.groups()
        if money is not None:
            value = money.replace(",", "")
            found.add("$" + (value[:-3] if value.endswith(".00") else value))
        else:
            found.add(duration)
    return found


def _quoted(quote: str, blocks: list[str]) -> bool:
    """Is the quote in the evidence: the normalized clause blocks and policy notes?"""
    # A quote may start with a clause header and skip to a later sentence of that clause.
    words = _normalize(_HEADER.sub(" ", quote))
    if len(words.split()) < MIN_QUOTE_WORDS or not blocks:
        return False
    source = " ".join(blocks)
    if any(_keeps_amounts(words, source, i, i + len(words)) for i in _occurrences(words, source)):
        return True
    match = fuzz.partial_ratio_alignment(words, source)
    if (
        match is not None
        and match.score >= QUOTE_MATCH
        and _keeps_amounts(words, source, match.dest_start, match.dest_end)
    ):
        return True
    parts = [part for part in (_normalize(p) for p in _GAP.split(words)) if part]
    if len(parts) < 2 or any(len(part.split()) < MIN_QUOTE_WORDS for part in parts):
        return False
    return any(_in_order(parts, block, 0, first=True) for block in blocks)


def _in_order(parts: list[str], block: str, position: int, *, first: bool) -> bool:
    """Are the parts in the block in this order, each with its amounts whole, skipping only
    words inside one sentence between them?"""
    if not parts:
        return True
    part = parts[0]
    for found in _occurrences(part, block, position):
        if not first and _SENTENCE_END.search(block[position:found]):
            # Later occurrences are further away and skip the same sentence end.
            return False
        if _keeps_amounts(part, block, found, found + len(part)) and _in_order(
            parts[1:], block, found + len(part), first=False
        ):
            return True
    return False


def _occurrences(needle: str, haystack: str, start: int = 0) -> list[int]:
    found: list[int] = []
    index = haystack.find(needle, start)
    while index >= 0:
        found.append(index)
        index = haystack.find(needle, index + 1)
    return found


def _keeps_amounts(quote: str, source: str, start: int, end: int) -> bool:
    """Is every amount in the quote a whole amount in source[start:end]? The span is widened to
    whole words first, so "$9" matched inside "$90" is read as "$90" and does not count."""
    start = source.rfind(" ", 0, start) + 1
    stop = source.find(" ", end)
    return _amounts(quote) <= _amounts(source[start : stop if stop >= 0 else len(source)])


def _normalize(text: str) -> str:
    text = text.casefold().replace("\u2019", "'")
    text = re.sub(r"\s+", " ", text)
    return text.strip(" .,;:\"'…")


def _prompt(question: str, answer: AgentAnswer, clauses: list[RetrievedClause]) -> str:
    if clauses:
        wording = "\n\n".join(_clause_block(c) for c in clauses)
    elif answer.policy_notes:
        # Shown the note alone, the judge left its quotes empty. The instruction sits here, not
        # in SYSTEM, so the judge's prompt for every other answer stays as it was.
        notes = "\n".join(answer.policy_notes)
        wording = (
            "(none: the policy tools returned no clauses, only the note below. For a statement "
            f"that says what the note says, quote the note's exact words.)\n{notes}"
        )
    else:
        wording = (
            "(no clauses: the policy search returned none, for example because no policy "
            "version was in force on the incident date)"
        )
    return (
        f"Question:\n{question}\n\n"
        f"Answer:\n{answer.answer}\n\n"
        f"Clauses cited by the answer:\n{wording}"
    )


def _clause_block(clause: RetrievedClause) -> str:
    # Quotes are checked against this same text, so a quote that keeps the header still counts.
    return f"[{clause.clause_ref} {clause.title}]\n{clause.text}"
