import uuid
from datetime import date
from typing import Any

import anyio
import pytest
from mcp.server.mcpserver import MCPServer
from sqlalchemy import Engine, select, text
from sqlalchemy.engine import URL

from core.settings import get_settings
from core.types import ClaimStatus
from db.models import AuditLog, Claim
from db.session import SessionFactory
from mcp_server import __main__ as stdio
from mcp_server import allowlist
from mcp_server.allowlist import ALLOWED_TOOLS, WRITE_TOOLS
from mcp_server.server import build_server, connect, tool_sessions
from tests.db.conftest import as_role
from tests.fakes import HashingEmbedder
from tests.mcp_server.factories import add_reviewed_claim

pytestmark = pytest.mark.db

ACTOR = "adj-1"


@pytest.fixture
def server(sessions: SessionFactory, app_role_engine: Engine) -> MCPServer:
    # `sessions` empties the tables; the tools themselves run as an adjuster on claimcheck_app.
    return build_server(tool_sessions(app_role_engine, ACTOR), HashingEmbedder(), ACTOR)


@pytest.fixture
def claim_id(sessions: SessionFactory) -> uuid.UUID:
    with sessions() as session:
        return add_reviewed_claim(session).id


def call(server: MCPServer, name: str, arguments: dict[str, Any]) -> tuple[bool, Any]:
    """Call one tool through an MCP client session; (is_error, structured result or text)."""

    async def run() -> tuple[bool, Any]:
        async with connect(server) as session:
            result = await session.call_tool(name, arguments)
        if result.is_error:
            return True, " ".join(getattr(block, "text", "") for block in result.content)
        return False, result.structured_content

    return anyio.run(run)


def listed(server: MCPServer) -> dict[str, dict[str, Any]]:
    async def run() -> dict[str, dict[str, Any]]:
        async with connect(server) as session:
            tools = await session.list_tools()
        return {t.name: t.input_schema for t in tools.tools}

    return anyio.run(run)


def audit_rows(sessions: SessionFactory, action: str) -> list[AuditLog]:
    with sessions() as session:
        return list(
            session.scalars(select(AuditLog).where(AuditLog.action == action).order_by(AuditLog.id))
        )


def test_server_exposes_exactly_the_allowlist(server: MCPServer) -> None:
    assert set(listed(server)) == ALLOWED_TOOLS
    assert {"flag_for_review"} == WRITE_TOOLS


def test_a_tool_outside_the_allowlist_cannot_be_called_and_is_audited(
    server: MCPServer, sessions: SessionFactory
) -> None:
    target = str(uuid.uuid4())

    is_error, text = call(server, "delete_claim", {"claim_id": target})

    assert is_error
    assert "delete_claim" in text
    (row,) = audit_rows(sessions, "agent.tool_call")
    assert row.payload["tool"] == "delete_claim"
    assert row.payload["arguments"] == {"claim_id": target}
    assert row.payload["outcome"].startswith("rejected: ")
    assert row.entity_id == target
    assert row.actor == ACTOR


@pytest.mark.parametrize(
    ("arguments", "missing_claim_id"),
    [
        ({"q": "glass"}, True),
        ({"query": 5}, True),
        ({"claim_id": "set below", "query": 5}, False),
        ({"claim_id": "set below"}, False),
    ],
)
def test_calls_with_invalid_arguments_are_rejected_and_audited(
    server: MCPServer,
    sessions: SessionFactory,
    claim_id: uuid.UUID,
    arguments: dict[str, Any],
    missing_claim_id: bool,
) -> None:
    if not missing_claim_id:
        arguments = {**arguments, "claim_id": str(claim_id)}

    is_error, _ = call(server, "search_policy_clauses", arguments)

    assert is_error
    (row,) = audit_rows(sessions, "agent.tool_call")
    assert row.payload["tool"] == "search_policy_clauses"
    assert row.payload["arguments"] == arguments
    assert row.payload["outcome"].startswith("rejected: ")
    assert row.entity_id == ("" if missing_claim_id else str(claim_id))


def test_build_fails_when_an_allowlisted_tool_has_no_implementation(
    sessions: SessionFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(allowlist, "ALLOWED_TOOLS", ALLOWED_TOOLS | {"approve_claim"})

    with pytest.raises(RuntimeError, match="approve_claim"):
        build_server(sessions, HashingEmbedder(), ACTOR)


def test_every_tool_call_writes_one_audit_row(
    server: MCPServer, sessions: SessionFactory, claim_id: uuid.UUID
) -> None:
    arguments: dict[str, dict[str, Any]] = {
        name: {"claim_id": str(claim_id)} for name in ALLOWED_TOOLS
    }
    arguments["search_policy_clauses"]["query"] = "glass damage"
    arguments["flag_for_review"].update(reason="invoice looks altered", human_confirmed=False)

    for name in sorted(ALLOWED_TOOLS):
        is_error, _ = call(server, name, arguments[name])
        assert not is_error, name

    rows = audit_rows(sessions, "agent.tool_call")
    assert sorted(r.payload["tool"] for r in rows) == sorted(ALLOWED_TOOLS)
    for row in rows:
        assert row.actor == ACTOR
        assert row.entity == "claim"
        assert row.entity_id == str(claim_id)
        assert row.payload["arguments"] == arguments[row.payload["tool"]]
        assert row.payload["outcome"] == "ok"


def test_a_failed_call_is_audited_too(server: MCPServer, sessions: SessionFactory) -> None:
    missing = str(uuid.uuid4())

    is_error, text = call(server, "get_claim_summary", {"claim_id": missing})

    assert is_error
    assert "not found" in text
    (row,) = audit_rows(sessions, "agent.tool_call")
    assert row.payload["tool"] == "get_claim_summary"
    assert row.payload["outcome"] == f"rejected: claim {missing} not found"


def test_a_malformed_claim_id_is_a_tool_error(server: MCPServer) -> None:
    is_error, text = call(server, "get_claim_summary", {"claim_id": "PKT-001"})

    assert is_error
    assert "claim" in text


def test_flag_for_review_without_confirmation_changes_nothing(
    server: MCPServer, sessions: SessionFactory, claim_id: uuid.UUID
) -> None:
    is_error, result = call(
        server, "flag_for_review", {"claim_id": str(claim_id), "reason": "check the invoice"}
    )

    assert not is_error
    assert result == {"status": "confirmation_required"}
    assert audit_rows(sessions, "claim.flagged") == []
    with sessions() as session:
        claim = session.get(Claim, claim_id)
        assert claim is not None and claim.status == ClaimStatus.NEEDS_REVIEW.value


def test_flag_for_review_with_confirmation_is_audited_and_keeps_the_status(
    server: MCPServer, sessions: SessionFactory, claim_id: uuid.UUID
) -> None:
    is_error, result = call(
        server,
        "flag_for_review",
        {"claim_id": str(claim_id), "reason": "check the invoice", "human_confirmed": True},
    )

    assert not is_error
    assert result["status"] == "flagged"
    (row,) = audit_rows(sessions, "claim.flagged")
    assert row.actor == ACTOR
    assert row.payload == {"reason": "check the invoice"}
    with sessions() as session:
        claim = session.get(Claim, claim_id)
        assert claim is not None and claim.status == ClaimStatus.NEEDS_REVIEW.value


def test_search_takes_no_date_and_returns_only_the_version_in_force(
    server: MCPServer, claim_id: uuid.UUID
) -> None:
    schema = listed(server)["search_policy_clauses"]
    assert set(schema["properties"]) == {"claim_id", "query"}

    is_error, result = call(
        server,
        "search_policy_clauses",
        {"claim_id": str(claim_id), "query": "glass damage limit"},
    )

    assert not is_error
    clauses = result["clauses"]
    assert {c["clause_ref"] for c in clauses} == {"4.1", "4.2"}
    assert {c["version_no"] for c in clauses} == {1}
    assert set(clauses[0]) == {"clause_ref", "title", "text", "version_no", "score"}
    assert result["version_no"] == 1


def test_search_for_a_claim_with_no_version_in_force_returns_no_clauses(
    sessions: SessionFactory, server: MCPServer
) -> None:
    with sessions() as session:
        early = add_reviewed_claim(session, packet_id="PKT-T2", incident_date=date(2023, 6, 1)).id

    is_error, result = call(
        server, "search_policy_clauses", {"claim_id": str(early), "query": "collision"}
    )

    assert not is_error
    assert result["clauses"] == []
    assert result["version_no"] is None
    assert "no policy version" in result["note"]


def test_get_policy_version(server: MCPServer, claim_id: uuid.UUID) -> None:
    is_error, result = call(server, "get_policy_version", {"claim_id": str(claim_id)})

    assert not is_error
    assert result["version_no"] == 1
    assert result["effective_from"] == "2024-01-01"
    assert result["effective_to"] == "2025-01-01"
    assert result["incident_date"] == "2024-06-01"


def test_claim_tools_return_compact_claim_data(server: MCPServer, claim_id: uuid.UUID) -> None:
    _, summary = call(server, "get_claim_summary", {"claim_id": str(claim_id)})
    _, documents = call(server, "list_claim_documents", {"claim_id": str(claim_id)})
    _, fields = call(server, "get_extracted_fields", {"claim_id": str(claim_id)})
    _, rules = call(server, "get_validation_results", {"claim_id": str(claim_id)})

    assert summary["packet_id"] == "PKT-T1"
    assert summary["status"] == "needs_review"
    assert summary["blocking_rules"] == ["R6_REQUIRED_DOCUMENTS"]
    assert summary["ready_for_decision"] is False
    assert documents["documents"][0]["filename"] == "claim_form.pdf"
    assert documents["documents"][0]["doc_type"] == "claim_form"
    assert fields["fields"][0]["field_name"] == "claimant_name"
    assert fields["fields"][0]["document"] == "claim_form.pdf"
    assert fields["fields"][0]["verdict"] == "found"
    by_rule = {r["rule_id"]: r for r in rules["rules"]}
    assert by_rule["R6_REQUIRED_DOCUMENTS"]["verdict"] == "missing"
    assert "id_card" in by_rule["R6_REQUIRED_DOCUMENTS"]["message"]
    # Database ids would make the agent's prompts differ from one database to the next.
    for payload in (summary, documents, fields, rules):
        assert str(claim_id) not in repr(payload)


def test_stdio_server_works_as_an_adjuster_on_the_app_role(
    sessions: SessionFactory, worker_db_url: URL, monkeypatch: pytest.MonkeyPatch
) -> None:
    app_url = as_role(worker_db_url, "claimcheck_app").render_as_string(hide_password=False)
    monkeypatch.setenv("APP_DATABASE_URL", app_url)
    get_settings.cache_clear()
    built: dict[str, Any] = {}

    class Started:
        def run(self, transport: str) -> None:
            built["transport"] = transport

    def build(tool_db: SessionFactory, embedder: Any, actor: str) -> Started:
        built["sessions"], built["actor"] = tool_db, actor
        return Started()

    monkeypatch.setattr(stdio, "build_server", build)
    monkeypatch.setattr(stdio, "FastEmbedEmbedder", HashingEmbedder)
    stdio.main(["--actor", "adj-7"])

    assert (built["transport"], built["actor"]) == ("stdio", "adj-7")
    probe = text(
        "SELECT current_user, current_setting('app.role', true),"
        " current_setting('app.user_id', true)"
    )
    with built["sessions"]() as session:
        assert tuple(session.execute(probe).one()) == ("claimcheck_app", "adjuster", "adj-7")
