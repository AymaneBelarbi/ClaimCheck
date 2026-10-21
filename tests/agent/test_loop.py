import json
import uuid
from collections.abc import Sequence
from typing import Any

import anyio
import pytest
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from agent.loop import BUDGET_MESSAGE, AgentAnswer, FinalMessage, answer_question, parse_final
from gateway.base import ChatTurn, GatewayError, ToolCall, Usage
from mcp_server.server import connect
from tests.fakes import FakeGateway

CLAIM_ID = uuid.UUID("00000000-0000-4000-8000-000000000001")
USAGE = Usage(model="fake", input_tokens=0, output_tokens=0, cost_usd=0.0, latency_ms=0.0)

GLASS = {
    "clause_ref": "4.2",
    "title": "Glass breakage",
    "text": "We cover glass breakage up to $1,000 per claim.",
    "version_no": 2,
    "score": 0.81,
}


NO_VERSION = "no policy version was in force on the incident date, so no clause applies"


class FakeTools:
    """A stand-in MCP server with the shapes of the real tools and no database."""

    def __init__(self, *, in_force: bool = True) -> None:
        self.received: list[tuple[str, dict[str, Any]]] = []
        self.server = MCPServer("fake-claimcheck")

        def search_policy_clauses(claim_id: str, query: str) -> dict[str, Any]:
            """Search the clauses of the policy version in force for the claim."""
            self.received.append(("search_policy_clauses", {"claim_id": claim_id, "query": query}))
            if not in_force:
                return {"version_no": None, "clauses": [], "note": NO_VERSION}
            return {"version_no": 2, "clauses": [GLASS], "note": None}

        def get_policy_version(claim_id: str) -> dict[str, Any]:
            """The policy version in force on the incident date."""
            self.received.append(("get_policy_version", {"claim_id": claim_id}))
            if not in_force:
                return {"version_no": None, "note": NO_VERSION}
            return {"version_no": 2, "note": "current version, no end date"}

        def get_validation_results(claim_id: str) -> dict[str, Any]:
            """Rule results of the claim."""
            self.received.append(("get_validation_results", {"claim_id": claim_id}))
            return {"rules": [{"rule_id": "R6_REQUIRED_DOCUMENTS", "verdict": "missing"}]}

        def flag_for_review(
            claim_id: str, reason: str, human_confirmed: bool = False
        ) -> dict[str, Any]:
            """Flag the claim for a second look."""
            self.received.append(
                (
                    "flag_for_review",
                    {"claim_id": claim_id, "reason": reason, "human_confirmed": human_confirmed},
                )
            )
            return {"status": "confirmation_required"}

        def get_claim_summary(claim_id: str) -> dict[str, Any]:
            """Fails like a missing claim does."""
            raise ToolError(f"claim {claim_id} not found")

        for fn in (
            search_policy_clauses,
            get_policy_version,
            get_validation_results,
            flag_for_review,
            get_claim_summary,
        ):
            self.server.add_tool(fn)


def tool_turn(*calls: tuple[str, dict[str, Any]]) -> ChatTurn:
    return ChatTurn(
        text=None,
        tool_calls=[
            ToolCall(id=f"call_{i}", name=name, arguments=args)
            for i, (name, args) in enumerate(calls)
        ],
        usage=USAGE,
    )


def final(answer: str, citations: Sequence[str]) -> ChatTurn:
    body = json.dumps({"answer": answer, "citations": list(citations)})
    return ChatTurn(text=body, tool_calls=[], usage=USAGE)


def ask(
    turns: Sequence[ChatTurn | Exception],
    tools: FakeTools | None = None,
    *,
    max_turns: int = 6,
    cache_key: str | None = None,
) -> tuple[AgentAnswer, FakeGateway, FakeTools]:
    tools = tools or FakeTools()
    gateway = FakeGateway(chat_turns=turns)

    async def run() -> AgentAnswer:
        async with connect(tools.server) as session:
            return await answer_question(
                claim_id=CLAIM_ID,
                question="Is glass damage covered?",
                gateway=gateway,
                mcp=session,
                max_turns=max_turns,
                cache_key=cache_key,
            )

    return anyio.run(run), gateway, tools


def test_answer_cites_a_clause_it_retrieved() -> None:
    answer, gateway, _ = ask(
        [
            tool_turn(("search_policy_clauses", {"query": "glass damage"})),
            final("Yes, glass breakage is covered up to $1,000.", ["4.2"]),
        ]
    )

    assert answer.citations == ["4.2"]
    assert answer.uncited is False
    assert answer.dropped_citations == []
    assert [c.name for c in answer.tool_calls] == ["search_policy_clauses"]
    assert [c.clause_ref for c in answer.clauses] == ["4.2"]
    assert answer.clauses[0].text == GLASS["text"]
    assert answer.stop_reason == "answered"
    assert len(gateway.calls) == 2


def test_tool_results_go_back_to_the_model() -> None:
    _, gateway, _ = ask(
        [
            tool_turn(("search_policy_clauses", {"query": "glass damage"})),
            final("Covered.", ["4.2"]),
        ]
    )

    second = gateway.calls[1].messages
    assert [m.role for m in second] == ["user", "assistant", "tool"]
    assert second[2].tool_call_id == "call_0"
    assert json.loads(second[2].content)["clauses"][0]["clause_ref"] == "4.2"


def test_a_citation_that_was_never_retrieved_is_dropped() -> None:
    answer, _, _ = ask(
        [
            tool_turn(("search_policy_clauses", {"query": "glass damage"})),
            final("Covered under 4.2 and 9.9.", ["4.2", "9.9"]),
        ]
    )

    assert answer.citations == ["4.2"]
    assert answer.dropped_citations == ["9.9"]
    assert answer.uncited is True


def test_citations_without_any_search_are_all_dropped() -> None:
    answer, _, _ = ask([final("Glass is covered.", ["4.2"])])

    assert answer.citations == []
    assert answer.dropped_citations == ["4.2"]
    assert answer.uncited is True
    assert answer.tool_calls == []


def test_an_answer_with_no_citations_is_uncited_but_drops_nothing() -> None:
    answer, _, _ = ask(
        [
            tool_turn(("get_validation_results", {})),
            final("R6_REQUIRED_DOCUMENTS is blocking: documents are missing.", []),
        ]
    )

    assert answer.citations == []
    assert answer.dropped_citations == []
    assert answer.uncited is True
    assert answer.stop_reason == "answered"


def test_the_tool_budget_ends_the_run() -> None:
    turns = [tool_turn(("search_policy_clauses", {"query": f"try {n}"})) for n in range(7)]

    answer, gateway, tools = ask(turns)

    assert answer.answer == BUDGET_MESSAGE
    assert answer.uncited is True
    assert answer.citations == []
    assert answer.stop_reason == "budget"
    assert len(gateway.calls) == 6
    # The sixth turn's calls are not run: nothing could read their results.
    assert len(tools.received) == 5
    # They are handed back, with the claim id pinned, so the caller can audit them.
    (unrun,) = answer.unexecuted_calls
    assert (unrun.name, unrun.arguments) == (
        "search_policy_clauses",
        {"claim_id": str(CLAIM_ID), "query": "try 5"},
    )


def test_an_answered_run_leaves_nothing_unexecuted() -> None:
    answer, _, _ = ask([final("Covered.", [])])

    assert answer.unexecuted_calls == []


def test_each_turn_has_its_own_cache_key() -> None:
    _, gateway, _ = ask(
        [
            tool_turn(("search_policy_clauses", {"query": "glass"})),
            final("Covered.", ["4.2"]),
        ],
        cache_key="agent:Q-01",
    )

    assert [c.cache_key for c in gateway.calls] == ["agent:Q-01:turn1", "agent:Q-01:turn2"]


def test_the_model_never_chooses_the_claim() -> None:
    other = str(uuid.uuid4())

    answer, gateway, tools = ask(
        [
            tool_turn(("search_policy_clauses", {"claim_id": other, "query": "glass"})),
            final("Covered.", ["4.2"]),
        ]
    )

    assert tools.received == [
        ("search_policy_clauses", {"claim_id": str(CLAIM_ID), "query": "glass"})
    ]
    assert answer.tool_calls[0].arguments["claim_id"] == str(CLAIM_ID)
    shown = {t.name: t.input_schema for t in gateway.calls[0].tools}
    assert "claim_id" not in shown["search_policy_clauses"]["properties"]
    assert "claim_id" not in shown["search_policy_clauses"].get("required", [])


def test_the_model_cannot_confirm_a_flag_on_a_humans_behalf() -> None:
    _, gateway, tools = ask(
        [
            tool_turn(("flag_for_review", {"reason": "altered invoice", "human_confirmed": True})),
            final("I asked for the claim to be flagged; a person has to confirm it.", []),
        ]
    )

    assert tools.received == [
        (
            "flag_for_review",
            {"claim_id": str(CLAIM_ID), "reason": "altered invoice", "human_confirmed": False},
        )
    ]
    shown = {t.name: t.input_schema for t in gateway.calls[0].tools}
    assert "human_confirmed" not in shown["flag_for_review"]["properties"]


def test_a_tool_error_is_reported_to_the_model() -> None:
    _, gateway, _ = ask(
        [
            tool_turn(("get_claim_summary", {})),
            final("I could not read the claim.", []),
        ]
    )

    tool_message = gateway.calls[1].messages[-1]
    assert tool_message.role == "tool"
    assert "not found" in tool_message.content
    assert tool_message.content.startswith("error")


def test_an_unknown_tool_is_reported_to_the_model() -> None:
    answer, gateway, _ = ask(
        [
            tool_turn(("approve_claim", {})),
            final("I cannot approve claims.", []),
        ]
    )

    assert "approve_claim" in gateway.calls[1].messages[-1].content
    assert answer.stop_reason == "answered"


def test_an_unparsable_final_message_keeps_the_text() -> None:
    answer, _, _ = ask(
        [ChatTurn(text="Glass is covered under clause 4.2.", tool_calls=[], usage=USAGE)]
    )

    assert answer.answer == "Glass is covered under clause 4.2."
    assert answer.citations == []
    assert answer.uncited is True
    assert answer.stop_reason == "unparsed"


@pytest.mark.parametrize(
    "text",
    [
        '```json\n{"answer": "Covered.", "citations": ["4.2"]}\n```',
        'Here is my answer:\n{"answer": "Covered.", "citations": ["4.2"]}',
        '{"answer": "Covered.", "citations": "4.2"}',
        '{"answer": "Covered.", "citations": ["Clause 4.2"]}',
        '{"answer": "Covered.", "citations": ["4.2 Glass breakage", "R1_NAME_MATCH"]}',
    ],
)
def test_parse_final_is_tolerant(text: str) -> None:
    assert parse_final(text) == FinalMessage("Covered.", ["4.2"], [])


@pytest.mark.parametrize("text", ["", "no json here", '{"citations": ["4.2"]}', "[1, 2]"])
def test_parse_final_gives_up_on_text_without_an_answer(text: str) -> None:
    assert parse_final(text) is None


def test_a_gateway_error_reaches_the_caller_as_itself() -> None:
    with pytest.raises(GatewayError, match="refused"):
        ask([GatewayError("connection refused", provider="ollama")])


def test_rule_ids_are_not_clause_citations() -> None:
    assert parse_final('{"answer": "R1 blocks it.", "citations": ["R1_NAME_MATCH"]}') == (
        FinalMessage("R1 blocks it.", [], [])
    )


def test_clauses_named_in_the_answer_count_when_the_list_is_empty() -> None:
    text = '{"answer": "Not covered: punctures are excluded (clause_ref: 6.2).", "citations": []}'

    assert parse_final(text) == FinalMessage(
        "Not covered: punctures are excluded (clause_ref: 6.2).", ["6.2"], []
    )


def test_amounts_in_the_answer_are_not_read_as_clauses() -> None:
    text = '{"answer": "Covered up to 1.5 times the limit, 4.2 days.", "citations": []}'

    assert parse_final(text) == FinalMessage("Covered up to 1.5 times the limit, 4.2 days.", [], [])


def test_a_numeric_citation_is_not_trusted() -> None:
    assert parse_final('{"answer": "Covered.", "citations": [4.10, "4.2"]}') == FinalMessage(
        "Covered.", ["4.2"], ["4.1"]
    )


def test_a_numeric_citation_counts_as_dropped() -> None:
    answer, _, _ = ask(
        [
            tool_turn(("search_policy_clauses", {"query": "glass"})),
            ChatTurn(text='{"answer": "Covered.", "citations": [4.2]}', tool_calls=[], usage=USAGE),
        ]
    )

    assert answer.citations == []
    assert answer.dropped_citations == ["4.2"]
    assert answer.uncited is True


@pytest.mark.parametrize("tool", ["search_policy_clauses", "get_policy_version"])
def test_a_policy_tools_no_version_note_is_kept_for_the_judge(tool: str) -> None:
    arguments = {"query": "collision"} if tool == "search_policy_clauses" else {}

    answer, _, _ = ask(
        [tool_turn((tool, arguments)), final("No version was in force.", [])],
        FakeTools(in_force=False),
    )

    assert answer.policy_notes == [NO_VERSION]


def test_a_note_about_a_version_in_force_is_not_kept() -> None:
    answer, _, _ = ask([tool_turn(("get_policy_version", {})), final("Version 2.", [])])

    assert answer.policy_notes == []


def test_the_no_version_note_is_kept_once() -> None:
    answer, _, _ = ask(
        [
            tool_turn(("get_policy_version", {}), ("search_policy_clauses", {"query": "x"})),
            final("No version was in force.", []),
        ],
        FakeTools(in_force=False),
    )

    assert answer.policy_notes == [NO_VERSION]
