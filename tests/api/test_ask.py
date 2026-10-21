import json
import time
import uuid
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from mcp import MCPError
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from sqlalchemy import select

from claimcheck.api.deps import get_embedder, get_llm
from claimcheck.api.main import app
from claimcheck.db.models import AuditLog
from claimcheck.db.session import SessionFactory
from claimcheck.gateway.base import (
    CassetteMissError,
    ChatTurn,
    GatewayError,
    Message,
    ToolCall,
    ToolSpec,
    Usage,
)
from claimcheck.gateway.tracing import TracingGateway
from tests.api.conftest import Headers
from tests.fakes import FakeGateway, HashingEmbedder
from tests.mcp_server.factories import CLAIMANT, add_reviewed_claim

pytestmark = pytest.mark.db

USAGE = Usage(model="fake", input_tokens=0, output_tokens=0, cost_usd=0.0, latency_ms=0.0)
QUESTION = "Is glass damage covered for this claim?"


def _turns() -> list[ChatTurn]:
    return [
        ChatTurn(
            text=None,
            tool_calls=[
                ToolCall(id="c1", name="search_policy_clauses", arguments={"query": "glass"})
            ],
            usage=USAGE,
        ),
        ChatTurn(
            text=json.dumps({"answer": "Glass is covered up to $1,000.", "citations": ["4.2"]}),
            tool_calls=[],
            usage=USAGE,
        ),
    ]


@pytest.fixture
def gateway() -> FakeGateway:
    return FakeGateway(chat_turns=_turns())


@pytest.fixture
def agent_client(client: TestClient, gateway: FakeGateway) -> Iterator[TestClient]:
    app.dependency_overrides[get_llm] = lambda: gateway
    app.dependency_overrides[get_embedder] = HashingEmbedder
    yield client


@pytest.fixture
def claim_id(sessions: SessionFactory) -> uuid.UUID:
    with sessions() as session:
        return add_reviewed_claim(session).id


def test_customer_cannot_ask(
    agent_client: TestClient, cust1: Headers, claim_id: uuid.UUID, gateway: FakeGateway
) -> None:
    response = agent_client.post(
        f"/claims/{claim_id}/ask", headers=cust1, json={"question": QUESTION}
    )

    assert response.status_code == 403
    assert gateway.calls == []


def test_adjuster_gets_a_cited_answer(
    agent_client: TestClient,
    adjuster: Headers,
    claim_id: uuid.UUID,
    sessions: SessionFactory,
    gateway: FakeGateway,
) -> None:
    response = agent_client.post(
        f"/claims/{claim_id}/ask", headers=adjuster, json={"question": QUESTION}
    )

    assert response.status_code == 200
    assert response.json() == {
        "answer": "Glass is covered up to $1,000.",
        "citations": ["4.2"],
        "uncited": False,
        "tools_called": ["search_policy_clauses"],
    }
    with sessions() as session:
        rows = list(session.scalars(select(AuditLog).order_by(AuditLog.id)))
    question = [r for r in rows if r.action == "agent.question"]
    assert len(question) == 1
    assert question[0].actor == "adj-1"
    assert question[0].payload == {"question": QUESTION}
    assert question[0].entity_id == str(claim_id)
    calls = [r for r in rows if r.action == "agent.tool_call"]
    assert [r.payload["tool"] for r in calls] == ["search_policy_clauses"]
    assert calls[0].actor == "adj-1"
    keys = [c.cache_key or "" for c in gateway.calls]
    # Keyed by packet id, which a freshly seeded database gives the same claim again.
    assert keys[0].startswith("agent:PKT-T1:") and keys[0].endswith(":turn1")
    assert keys[1].endswith(":turn2")


def test_a_cloud_backend_never_sees_the_claimants_name(
    agent_client: TestClient, adjuster: Headers, claim_id: uuid.UUID
) -> None:
    from claimcheck.gateway.redaction import RedactingGateway

    inner = FakeGateway(
        chat_turns=[
            ChatTurn(
                text=None,
                tool_calls=[ToolCall(id="c1", name="get_extracted_fields", arguments={})],
                usage=USAGE,
            ),
            ChatTurn(
                text=json.dumps({"answer": "The claimant is named on the form.", "citations": []}),
                tool_calls=[],
                usage=USAGE,
            ),
        ],
        is_cloud=True,
    )
    app.dependency_overrides[get_llm] = lambda: RedactingGateway(inner)

    response = agent_client.post(
        f"/claims/{claim_id}/ask", headers=adjuster, json={"question": "Who is the claimant?"}
    )

    assert response.status_code == 200
    sent = inner.calls[1].messages[-1].content
    assert "claimant_name" in sent
    assert "Jane" not in sent and "Doe" not in sent
    assert CLAIMANT not in sent


def test_unknown_claim_is_404(agent_client: TestClient, adjuster: Headers) -> None:
    response = agent_client.post(
        f"/claims/{uuid.uuid4()}/ask", headers=adjuster, json={"question": QUESTION}
    )

    assert response.status_code == 404


def test_empty_question_is_rejected(
    agent_client: TestClient, adjuster: Headers, claim_id: uuid.UUID
) -> None:
    response = agent_client.post(f"/claims/{claim_id}/ask", headers=adjuster, json={"question": ""})

    assert response.status_code == 422


def test_backend_failure_is_502(
    agent_client: TestClient, adjuster: Headers, claim_id: uuid.UUID
) -> None:
    failing = FakeGateway(chat_turns=[GatewayError("connection refused", provider="ollama")])
    app.dependency_overrides[get_llm] = lambda: failing

    response = agent_client.post(
        f"/claims/{claim_id}/ask", headers=adjuster, json={"question": QUESTION}
    )

    assert response.status_code == 502


def test_a_question_with_no_recording_is_503_under_replay(
    agent_client: TestClient, adjuster: Headers, claim_id: uuid.UUID
) -> None:
    missing = FakeGateway(chat_turns=[CassetteMissError("agent:PKT-1:abc:turn1")])
    app.dependency_overrides[get_llm] = lambda: missing

    response = agent_client.post(
        f"/claims/{claim_id}/ask", headers=adjuster, json={"question": QUESTION}
    )

    assert response.status_code == 503
    assert "LLM_BACKEND=replay" in response.json()["detail"]


def test_a_tool_server_failure_is_502(
    agent_client: TestClient,
    adjuster: Headers,
    claim_id: uuid.UUID,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def broken(*args: object, **kwargs: object) -> None:
        raise MCPError(code=-32603, message="tool server crashed")

    monkeypatch.setattr("claimcheck.api.routes.claims.answer_question", broken)

    response = agent_client.post(
        f"/claims/{claim_id}/ask", headers=adjuster, json={"question": QUESTION}
    )

    assert response.status_code == 502
    assert "tool server" in response.json()["detail"]


class _SlowGateway(FakeGateway):
    def chat(
        self,
        *,
        system: str,
        messages: list[Message],
        tools: list[ToolSpec],
        cache_key: str | None = None,
    ) -> ChatTurn:
        time.sleep(1.0)
        return super().chat(system=system, messages=messages, tools=tools, cache_key=cache_key)


def test_a_question_past_the_deadline_is_504(
    agent_client: TestClient,
    adjuster: Headers,
    claim_id: uuid.UUID,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("claimcheck.api.routes.claims.ASK_DEADLINE_SECONDS", 0.2)
    app.dependency_overrides[get_llm] = lambda: _SlowGateway(chat_turns=_turns())
    started = time.monotonic()

    response = agent_client.post(
        f"/claims/{claim_id}/ask", headers=adjuster, json={"question": QUESTION}
    )

    assert response.status_code == 504
    assert time.monotonic() - started < 1.0


def test_tools_called_matches_the_audit_trail_including_rejected_calls(
    agent_client: TestClient, adjuster: Headers, claim_id: uuid.UUID, sessions: SessionFactory
) -> None:
    turns = [
        ChatTurn(
            text=None,
            tool_calls=[
                ToolCall(id="c1", name="approve_claim", arguments={}),
                ToolCall(id="c2", name="search_policy_clauses", arguments={"q": "glass"}),
                ToolCall(id="c3", name="search_policy_clauses", arguments={"query": "glass"}),
            ],
            usage=USAGE,
        ),
        ChatTurn(
            text=json.dumps({"answer": "Glass is covered.", "citations": ["4.2"]}),
            tool_calls=[],
            usage=USAGE,
        ),
    ]
    app.dependency_overrides[get_llm] = lambda: FakeGateway(chat_turns=turns)

    response = agent_client.post(
        f"/claims/{claim_id}/ask", headers=adjuster, json={"question": QUESTION}
    )

    assert response.status_code == 200
    called = response.json()["tools_called"]
    assert called == ["approve_claim", "search_policy_clauses", "search_policy_clauses"]
    with sessions() as session:
        rows = list(
            session.scalars(
                select(AuditLog).where(AuditLog.action == "agent.tool_call").order_by(AuditLog.id)
            )
        )
    assert [r.payload["tool"] for r in rows] == called
    assert [r.payload["outcome"].split(":")[0] for r in rows] == ["rejected", "rejected", "ok"]


def test_calls_proposed_on_the_last_turn_are_audited_as_rejected(
    agent_client: TestClient, adjuster: Headers, claim_id: uuid.UUID, sessions: SessionFactory
) -> None:
    turns = [
        ChatTurn(
            text=None,
            tool_calls=[
                ToolCall(id=f"c{n}", name="search_policy_clauses", arguments={"query": f"q{n}"})
            ],
            usage=USAGE,
        )
        for n in range(6)
    ]
    app.dependency_overrides[get_llm] = lambda: FakeGateway(chat_turns=turns)

    response = agent_client.post(
        f"/claims/{claim_id}/ask", headers=adjuster, json={"question": QUESTION}
    )

    assert response.status_code == 200
    assert response.json()["tools_called"] == ["search_policy_clauses"] * 5
    with sessions() as session:
        rows = list(
            session.scalars(
                select(AuditLog).where(AuditLog.action == "agent.tool_call").order_by(AuditLog.id)
            )
        )
    assert [r.payload["outcome"] for r in rows] == ["ok"] * 5 + ["rejected: budget"]
    assert rows[-1].payload["arguments"] == {"claim_id": str(claim_id), "query": "q5"}
    assert rows[-1].entity_id == str(claim_id)


def test_each_model_call_is_a_child_of_the_ask_span(
    agent_client: TestClient,
    adjuster: Headers,
    claim_id: uuid.UUID,
    gateway: FakeGateway,
    spans: InMemorySpanExporter,
) -> None:
    app.dependency_overrides[get_llm] = lambda: TracingGateway(gateway, system="replay")

    response = agent_client.post(
        f"/claims/{claim_id}/ask", headers=adjuster, json={"question": QUESTION}
    )

    assert response.status_code == 200
    finished = list(spans.get_finished_spans())
    (root,) = [s for s in finished if s.name == "agent.ask"]
    assert (root.attributes or {})["claim.id"] == str(claim_id)
    calls = [s for s in finished if s.name == "llm.call"]
    assert len(calls) == 2
    for call in calls:
        assert call.parent is not None and call.parent.span_id == root.context.span_id
        assert (call.attributes or {})["claimcheck.purpose"] == "agent.chat"
    assert QUESTION not in repr([dict(s.attributes or {}) for s in finished])
