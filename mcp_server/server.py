"""The MCP server for the adjuster agent, and an in-process client connection to it.

`build_server` registers exactly the allowlisted tools. The API and the evals talk to it
through `connect`, an MCP client session over the SDK's in-memory transport, so the agent uses
the same protocol whether the server runs in-process or as `python -m mcp_server` on stdio.

Every tool call is audited here, at the point where the server dispatches it, rather than in the
tools: an unknown tool name or arguments that fail the tool's schema never reach a tool, and
those attempts belong in the trail as much as the calls that succeed. Each attempt writes one
`agent.tool_call` row ({tool, arguments, outcome}) in a transaction of its own before the
result goes back to the client. `outcome` is `ok`, `rejected: <reason>` for anything the server
or a tool refused, or `error` for a crash.
"""

import inspect
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from functools import partial
from typing import Any

import anyio.to_thread
from mcp import Client, ClientSession
from mcp.server.mcpserver import Context, MCPServer
from mcp.server.mcpserver.exceptions import ToolError, UnexpectedToolError
from mcp_types import CallToolResult, InputRequiredResult, ToolAnnotations
from sqlalchemy import Engine, select
from sqlalchemy.orm import Session

from core.types import Role
from db.models import ClaimDocument
from db.session import SessionFactory, session_factory
from mcp_server import allowlist
from mcp_server.tools import ClaimTools
from rag.embedder import Embedder
from worker.audit import record

INSTRUCTIONS = (
    "Read-only access to one insurer's auto claims and the policy wording that applied to "
    "each claim. Every call is written to the audit log. flag_for_review does nothing unless "
    "human_confirmed is true, which only a person may set."
)


class AuditedServer(MCPServer[Any]):
    """An MCPServer that writes one audit row per attempted tool call, whatever its outcome."""

    def __init__(self, sessions: SessionFactory, actor: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._sessions = sessions
        self._actor = actor

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        context: Context[Any, Any] | None = None,
    ) -> CallToolResult | InputRequiredResult:
        try:
            result = await super().call_tool(name, arguments, context)
        except UnexpectedToolError:
            await self._audit(name, arguments, "error")
            raise
        except ToolError as exc:
            await self._audit(name, arguments, f"rejected: {_reason(exc)}")
            raise
        except Exception:
            await self._audit(name, arguments, "error")
            raise
        outcome = "rejected: tool reported an error" if result_is_error(result) else "ok"
        await self._audit(name, arguments, outcome)
        return result

    async def _audit(self, tool: str, arguments: dict[str, Any], outcome: str) -> None:
        await anyio.to_thread.run_sync(partial(self._write_audit, tool, arguments, outcome))

    def _write_audit(self, tool: str, arguments: dict[str, Any], outcome: str) -> None:
        claim_id = arguments.get("claim_id")
        with self._sessions() as session:
            record(
                session,
                actor=self._actor,
                action="agent.tool_call",
                entity="claim",
                entity_id=claim_id if isinstance(claim_id, str) else "",
                payload={"tool": tool, "arguments": arguments, "outcome": outcome},
            )


def result_is_error(result: CallToolResult | InputRequiredResult) -> bool:
    return isinstance(result, CallToolResult) and result.is_error


def _reason(exc: ToolError) -> str:
    cause = exc.__cause__
    errors = getattr(cause, "errors", None)
    if callable(errors):
        # Argument validation: name the fields, as the SDK's own log does.
        fields = sorted({".".join(str(p) for p in e["loc"]) for e in errors()})
        return f"invalid arguments: {', '.join(fields)}"
    # The SDK wraps a tool's own ToolError in "Error executing tool ...".
    return str(cause) if isinstance(cause, ToolError) else str(exc)


def tool_sessions(engine: Engine, actor: str) -> SessionFactory:
    """Transactions for the tools: an adjuster's view of the claims, as `actor`.

    The agent works for an adjuster, so it reads what an adjuster may read and nothing more;
    `engine` should log in as claimcheck_app, the role row-level security applies to.
    """
    return session_factory(engine, Role.ADJUSTER, actor)


def build_server(sessions: SessionFactory, embedder: Embedder, actor: str) -> MCPServer[Any]:
    """A server whose tools are exactly `ALLOWED_TOOLS`; audit rows carry `actor`."""
    implemented = ClaimTools(sessions, embedder, actor).implementations()
    allowed = allowlist.ALLOWED_TOOLS
    missing = sorted(allowed - implemented.keys())
    if missing:
        raise RuntimeError(f"allowlisted tools without an implementation: {', '.join(missing)}")

    server = AuditedServer(sessions, actor, name="claimcheck", instructions=INSTRUCTIONS)
    registered: set[str] = set()
    for name in sorted(allowed):
        read_only = name not in allowlist.WRITE_TOOLS
        fn = implemented[name]
        server.add_tool(
            fn,
            name=name,
            # One line: the docstring's wrapping is noise in the model's prompt.
            description=" ".join((inspect.getdoc(fn) or "").split()),
            annotations=ToolAnnotations(read_only_hint=read_only, destructive_hint=False),
        )
        registered.add(name)
    if registered != allowed:
        raise RuntimeError(f"registered tools {sorted(registered)} differ from the allowlist")
    return server


@asynccontextmanager
async def connect(server: MCPServer[Any]) -> AsyncIterator[ClientSession]:
    """An initialized MCP client session to `server`, running in this event loop.

    `legacy` mode keeps the JSON-RPC framing and initialize handshake of a real transport,
    so in-process calls serialize their arguments and results exactly as stdio would.

    An exception from the body is re-raised as itself once the connection is closed, rather
    than inside the exception groups of the client's task groups.
    """
    error: Exception | None = None
    async with Client(server, mode="legacy") as client:
        try:
            yield client.session
        except Exception as exc:
            error = exc
    if error is not None:
        raise error


def claim_ocr_texts(session: Session, claim_id: uuid.UUID) -> list[str]:
    """The OCR text of every document of a claim: the scope for redacting the agent's calls."""
    return [
        text
        for text in session.scalars(
            select(ClaimDocument.ocr_text).where(ClaimDocument.claim_id == claim_id)
        )
        if text
    ]
