import json
from datetime import date
from pathlib import Path

import pytest
from pydantic import SecretStr
from sqlalchemy import Engine, func, select
from sqlalchemy.engine import URL

from agent.judge import JudgeVerdict, StatementCheck, StatementChecks
from agent.loop import BUDGET_MESSAGE, AgentAnswer, StopReason
from core.settings import Settings
from db.models import Claim, ClaimDocument
from db.session import SessionFactory
from evals.context import EvalContext
from evals.dataset.types import AgentQuestion, GroundTruth, PolicyIndex
from evals.runners import agent as agent_runner
from evals.runners.agent import load_questions, score, summarize
from gateway.base import ChatTurn, ToolCall, Usage
from gateway.redaction import RedactingGateway
from mcp_server.server import tool_sessions
from rag.chunking import chunk_clauses
from tests.db.conftest import as_role
from tests.fakes import FakeGateway, HashingEmbedder
from tests.mcp_server.factories import CLAIMANT, add_reviewed_claim
from tests.worker.conftest import (  # noqa: F401
    app_role_engine,
    engine,
    sessions,
    worker_db_url,
)
from tests.worker.factories import add_dataset_policy, ground_truth
from worker.storage import Storage

DATA = Path(__file__).parents[2] / "evals" / "data"
QUESTIONS_FILE = DATA.parent / "agent_questions.jsonl"

USAGE = Usage(model="fake", input_tokens=0, output_tokens=0, cost_usd=0.0, latency_ms=0.0)
SEARCH = ToolCall(id="c1", name="search_policy_clauses", arguments={"query": "glass"})
FAITHFUL = JudgeVerdict(faithful=True, unsupported_statements=[])
UNFAITHFUL = JudgeVerdict(faithful=False, unsupported_statements=["glass is covered"])


def _question(expected: str | None) -> AgentQuestion:
    return AgentQuestion(
        id="AQ-T", packet_id="PKT-001", question="Is glass covered?", expected_clause_ref=expected
    )


def _answer(
    citations: list[str],
    *,
    dropped: list[str] | None = None,
    calls: list[ToolCall] | None = None,
    stop_reason: StopReason = "answered",
    text: str = "Yes.",
) -> AgentAnswer:
    dropped = dropped or []
    return AgentAnswer(
        answer=text,
        citations=citations,
        dropped_citations=dropped,
        tool_calls=calls if calls is not None else [SEARCH],
        uncited=not citations or bool(dropped) or stop_reason != "answered",
        stop_reason=stop_reason,
    )


def _version_clauses(policy_number: str, incident_date: date) -> set[str]:
    index = PolicyIndex.model_validate_json((DATA / "policies" / "index.json").read_text())
    entry = next(p for p in index.policies if p.policy_number == policy_number)
    for v in entry.versions:
        if v.effective_from <= incident_date and (
            v.effective_to is None or incident_date < v.effective_to
        ):
            text = (DATA / "policies" / v.file).read_text(encoding="utf-8")
            return {c.clause_ref for c in chunk_clauses(text)}
    return set()


def test_question_file_is_valid() -> None:
    questions = load_questions(QUESTIONS_FILE)
    splits = json.loads((DATA / "splits.json").read_text())

    dev = [q for q in questions if q.packet_id in splits["dev"]]
    holdout = [q for q in questions if q.packet_id in splits["holdout"]]

    assert len(dev) >= 20
    assert len(holdout) >= 8
    assert len(dev) + len(holdout) == len(questions)
    assert len({q.id for q in questions}) == len(questions)
    for subset in (dev, holdout):
        assert sum(q.expected_clause_ref is None for q in subset) >= 3
        assert sum(q.expected_clause_ref is not None for q in subset) >= 5
    for q in questions:
        truth = GroundTruth.model_validate_json(
            (DATA / "packets" / q.packet_id / "ground_truth.json").read_text()
        )
        if q.expected_clause_ref is not None:
            # The expected clause exists in the version in force on the claim's incident date.
            available = _version_clauses(truth.policy_number, truth.incident_date)
            assert q.expected_clause_ref in available, q.id


def test_clause_question_passes_when_faithful_and_cited() -> None:
    result = score(_question("4.2"), _answer(["4.2"]), FAITHFUL)

    assert result.passed is True
    assert result.citation_hit is True
    assert result.failure is None


def test_clause_question_fails_when_unfaithful() -> None:
    result = score(_question("4.2"), _answer(["4.2"]), UNFAITHFUL)

    assert result.passed is False
    assert result.failure == "unfaithful"


def test_clause_question_without_citations_fails_even_if_the_judge_agrees() -> None:
    result = score(_question("4.2"), _answer([]), FAITHFUL)

    assert result.passed is False
    assert result.citation_hit is False
    assert result.failure == "no_citation"


@pytest.mark.parametrize(
    ("answer", "failure"),
    [
        (_answer([], calls=[], stop_reason="answered"), "no_search"),
        (_answer(["4.2"], dropped=["9.9"]), "hallucinated_citation"),
        (_answer([], stop_reason="unparsed"), "bad_json"),
        (_answer([], stop_reason="budget", text=BUDGET_MESSAGE), "budget"),
    ],
)
def test_clause_question_failure_kinds(answer: AgentAnswer, failure: str) -> None:
    result = score(_question("4.2"), answer, FAITHFUL)

    assert result.passed is False
    assert result.failure == failure


def test_citing_a_retrieved_but_different_clause_is_faithful_but_misses() -> None:
    result = score(_question("4.2"), _answer(["4.1"]), FAITHFUL)

    assert result.passed is True
    assert result.citation_hit is False


def test_claim_data_question_needs_no_citation() -> None:
    result = score(_question(None), _answer([], calls=[]), FAITHFUL)

    assert result.passed is True
    assert result.citation_hit is None


def test_claim_data_question_fails_on_a_dropped_citation() -> None:
    result = score(_question(None), _answer([], dropped=["4.2"]), FAITHFUL)

    assert result.passed is False
    assert result.failure == "hallucinated_citation"


def test_claim_data_question_fails_on_the_budget() -> None:
    result = score(
        _question(None), _answer([], stop_reason="budget", text=BUDGET_MESSAGE), FAITHFUL
    )

    assert result.passed is False
    assert result.failure == "budget"


def test_summarize_rates() -> None:
    results = [
        score(_question("4.2"), _answer(["4.2"]), FAITHFUL),
        score(_question("4.2"), _answer(["4.1"]), FAITHFUL),
        score(_question("4.2"), _answer(["4.2"]), UNFAITHFUL),
        score(_question(None), _answer([], calls=[]), FAITHFUL),
        score(_question(None), _answer([], calls=[], stop_reason="unparsed"), FAITHFUL),
    ]

    metrics, failures = summarize(results)

    # Only policy questions are in the gated metric.
    assert metrics == {
        "agent.faithfulness": pytest.approx(2 / 3),
        "agent.citation_hit": pytest.approx(2 / 3),
        "agent.claim_data_accuracy": pytest.approx(1 / 2),
    }
    assert failures == {"bad_json": 1, "unfaithful": 1}


def test_claim_data_question_fails_on_plain_text() -> None:
    result = score(_question(None), _answer([], calls=[], stop_reason="unparsed"), FAITHFUL)

    assert result.passed is False
    assert result.failure == "bad_json"


def test_a_missing_verdict_fails_the_question() -> None:
    result = score(_question("4.2"), _answer(["4.2"]), None)

    assert result.passed is False
    assert result.failure == "judge_error"


def test_summarize_with_no_questions_produces_no_metrics() -> None:
    assert summarize([]) == ({}, {})


# ---- run() against a database --------------------------------------------------------------


@pytest.mark.db
def test_run_answers_judges_and_reports(
    tmp_path: Path,
    worker_db_url: URL,  # noqa: F811
    sessions: SessionFactory,  # noqa: F811
) -> None:
    with sessions() as session:
        add_reviewed_claim(session, packet_id="PKT-T1")
    data = tmp_path / "data"
    data.mkdir()
    (data / "splits.json").write_text(json.dumps({"dev": ["PKT-T1"], "holdout": []}))
    questions = [
        {
            "id": "AQ-1",
            "packet_id": "PKT-T1",
            "question": "Is glass covered?",
            "expected_clause_ref": "4.2",
        },
    ]
    (tmp_path / "agent_questions.jsonl").write_text(
        "\n".join(json.dumps(q) for q in questions) + "\n"
    )
    supported = StatementCheck(
        statement="Covered up to $1,000",
        about="policy",
        quote="We pay up to a limit of $1,000.00 for glass damage",
        supported=True,
    )
    # A cloud backend, so the judge's prompt goes through redaction.
    inner = FakeGateway(
        structured_outputs=[StatementChecks(statements=[supported])],
        chat_turns=[
            ChatTurn(
                text=None,
                tool_calls=[
                    ToolCall(
                        id="c", name="search_policy_clauses", arguments={"query": "glass damage"}
                    )
                ],
                usage=USAGE,
            ),
            ChatTurn(
                text=json.dumps(
                    {"answer": f"Covered up to $1,000 for {CLAIMANT}.", "citations": ["4.2"]}
                ),
                tool_calls=[],
                usage=USAGE,
            ),
        ],
        is_cloud=True,
    )
    settings = Settings(
        database_url=worker_db_url.render_as_string(hide_password=False),
        app_database_url=_rendered(as_role(worker_db_url, "claimcheck_app")),
        worker_database_url=_rendered(as_role(worker_db_url, "claimcheck_worker")),
        jwt_secret=SecretStr("x" * 40),
    )
    ctx = EvalContext(data, settings, gateway=RedactingGateway(inner), embedder=HashingEmbedder())

    result = agent_runner.run("dev", ctx)

    assert result.metrics == {"agent.faithfulness": 1.0, "agent.citation_hit": 1.0}
    (row,) = result.details["answers"]
    assert row["citations"] == ["4.2"]
    assert row["answer"] == f"Covered up to $1,000 for {CLAIMANT}."
    assert row["tool_calls"] == [
        {"name": "search_policy_clauses", "arguments": {"query": "glass damage"}}
    ]
    assert [c.cache_key for c in inner.calls] == [
        "agent:AQ-1:turn1",
        "agent:AQ-1:turn2",
        "judge:AQ-1",
    ]
    judged = inner.calls[2].prompt or ""
    assert "4.2 Glass damage" in judged
    # The judge goes through the claim's redaction scope: the name in the answer is learned
    # from the claim form's OCR text, not from anything in the judge's own prompt.
    assert "for " in judged and "Jane" not in judged and "Doe" not in judged

    holdout = agent_runner.run("holdout", ctx)
    assert holdout.metrics == {}
    assert holdout.details["questions"] == 0


def _rendered(url: URL) -> str:
    return url.render_as_string(hide_password=False)


@pytest.mark.db
def test_a_missing_packet_is_filed_as_an_adjuster_on_the_app_role(
    tmp_path: Path,
    sessions: SessionFactory,  # noqa: F811
    app_role_engine: Engine,  # noqa: F811
) -> None:
    packet_id = json.loads((DATA / "splits.json").read_text())["dev"][0]
    truth = ground_truth(DATA, packet_id)
    with sessions() as session:
        holder = add_dataset_policy(session, DATA, truth.policy_number, HashingEmbedder()).holder_id
    filing = tool_sessions(app_role_engine, agent_runner.ACTOR)

    claim_id, processed = agent_runner._claim_for(filing, DATA, packet_id, Storage(tmp_path / "a"))

    assert processed is False
    with sessions() as session:
        claim = session.get(Claim, claim_id)
        assert claim is not None and claim.created_by == holder
        documents = session.scalar(
            select(func.count())
            .select_from(ClaimDocument)
            .where(ClaimDocument.claim_id == claim_id)
        )
        assert documents == len(truth.documents)
    # An unprocessed claim from an earlier run is picked up again rather than filed twice.
    again = agent_runner._claim_for(filing, DATA, packet_id, Storage(tmp_path / "b"))
    assert again == (claim_id, False)
