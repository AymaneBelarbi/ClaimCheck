"""The adjuster Q&A agent: one model, the MCP tools, a fixed turn budget.

The loop is deliberately plain. Each turn sends the conversation and the tool list to the
gateway; tool calls go to the MCP server, their results come back as tool messages; a turn
without tool calls is the final answer, which must be a JSON object `{answer, citations}`.

Three things are decided here rather than by the model:

- The claim. The agent answers about the claim it was opened on, so `claim_id` is removed from
  every tool schema the model sees and set on every call. A 7B model cannot mistype a UUID it
  never writes, and no prompt can point it at another claim.
- Confirmation. `human_confirmed` is likewise hidden and always sent as false: the model can
  ask for a claim to be flagged, but only a person can confirm it.
- Citations. A clause ref counts only if `search_policy_clauses` returned it during this run.
  Anything else is dropped and the answer is marked `uncited`, as is an answer with no
  citations at all, one whose final message was not the JSON asked for, and one that ran out
  of turns.
"""

import json
import re
import uuid
from functools import partial
from pathlib import Path
from typing import Any, Literal, NamedTuple

import anyio.to_thread
from mcp import ClientSession, MCPError
from pydantic import BaseModel, Field

from claimcheck.gateway.base import LLMGateway, Message, ToolCall, ToolSpec
from claimcheck.rag.retrieve import RetrievedClause

BUDGET_MESSAGE = "I could not answer within the tool budget."
SYSTEM_PROMPT = (Path(__file__).parent / "prompts" / "system.txt").read_text(encoding="utf-8")
SEARCH_TOOL = "search_policy_clauses"
VERSION_TOOL = "get_policy_version"

# Set by the loop on every call, never by the model.
_PINNED = "claim_id"
_CONFIRMATION = "human_confirmed"

StopReason = Literal["answered", "unparsed", "budget"]


class AgentAnswer(BaseModel):
    answer: str
    # Clause refs that search_policy_clauses returned during this run.
    citations: list[str]
    # As executed, with the claim id the loop pinned.
    tool_calls: list[ToolCall]
    uncited: bool
    # Refs the model cited that no search returned.
    dropped_citations: list[str] = Field(default_factory=list)
    # Every clause a search returned, in the order first seen.
    clauses: list[RetrievedClause] = Field(default_factory=list)
    # What the policy tools said when they had no version to answer from ("no policy version
    # was in force on the incident date, ..."). It is the only evidence for such an answer.
    policy_notes: list[str] = Field(default_factory=list)
    stop_reason: StopReason = "answered"
    # Calls the model proposed on the last turn of the budget, with the claim id pinned. They
    # are never sent to the tool server, so the caller audits them as rejected.
    unexecuted_calls: list[ToolCall] = Field(default_factory=list)


async def answer_question(
    *,
    claim_id: uuid.UUID,
    question: str,
    gateway: LLMGateway,
    mcp: ClientSession,
    max_turns: int = 6,
    cache_key: str | None = None,
) -> AgentAnswer:
    tools, needs_confirmation = await _tools_for_model(mcp)
    messages = [Message(role="user", content=question)]
    retrieved: dict[str, RetrievedClause] = {}
    notes: list[str] = []
    executed: list[ToolCall] = []
    unexecuted: list[ToolCall] = []

    for turn in range(1, max_turns + 1):
        reply = await anyio.to_thread.run_sync(
            partial(
                gateway.chat,
                system=SYSTEM_PROMPT,
                messages=list(messages),
                tools=tools,
                cache_key=f"{cache_key}:turn{turn}" if cache_key else None,
            ),
            # A caller's deadline must not wait for a model call that is still running.
            abandon_on_cancel=True,
        )
        if not reply.tool_calls:
            return _finish(reply.text or "", executed, retrieved, notes)
        if turn == max_turns:
            # Nothing would read the results of this turn's calls.
            unexecuted = [
                call.model_copy(
                    update={
                        "arguments": _pin(call.arguments, claim_id, call.name in needs_confirmation)
                    }
                )
                for call in reply.tool_calls
            ]
            break
        messages.append(
            Message(role="assistant", content=reply.text or "", tool_calls=reply.tool_calls)
        )
        for call in reply.tool_calls:
            arguments = _pin(call.arguments, claim_id, call.name in needs_confirmation)
            executed.append(call.model_copy(update={"arguments": arguments}))
            content, clauses, note = await _execute(mcp, call.name, arguments)
            for clause in clauses:
                retrieved.setdefault(clause.clause_ref, clause)
            if note is not None and note not in notes:
                notes.append(note)
            messages.append(Message(role="tool", content=content, tool_call_id=call.id))

    return AgentAnswer(
        answer=BUDGET_MESSAGE,
        citations=[],
        tool_calls=executed,
        uncited=True,
        clauses=list(retrieved.values()),
        policy_notes=notes,
        stop_reason="budget",
        unexecuted_calls=unexecuted,
    )


async def _tools_for_model(mcp: ClientSession) -> tuple[list[ToolSpec], set[str]]:
    """The tools as the model sees them, and the names of those that take a confirmation."""
    listing = await mcp.list_tools()
    specs = [
        ToolSpec(
            name=tool.name,
            description=tool.description or "",
            input_schema=_without(tool.input_schema, {_PINNED, _CONFIRMATION}),
        )
        for tool in listing.tools
    ]
    confirmable = {
        tool.name
        for tool in listing.tools
        if _CONFIRMATION in (tool.input_schema.get("properties") or {})
    }
    return specs, confirmable


def _without(schema: dict[str, Any], hidden: set[str]) -> dict[str, Any]:
    result = dict(schema)
    properties = result.get("properties")
    if isinstance(properties, dict):
        result["properties"] = {k: v for k, v in properties.items() if k not in hidden}
    required = result.get("required")
    if isinstance(required, list):
        result["required"] = [name for name in required if name not in hidden]
    return result


def _pin(arguments: dict[str, Any], claim_id: uuid.UUID, confirmable: bool) -> dict[str, Any]:
    pinned = {k: v for k, v in arguments.items() if k not in {_PINNED, _CONFIRMATION}}
    pinned = {_PINNED: str(claim_id), **pinned}
    if confirmable:
        pinned[_CONFIRMATION] = False
    return pinned


class ToolOutput(NamedTuple):
    # What goes back to the model.
    content: str
    # The clauses a search returned.
    clauses: list[RetrievedClause]
    # A policy tool's note when it found no version in force.
    note: str | None


async def _execute(mcp: ClientSession, name: str, arguments: dict[str, Any]) -> ToolOutput:
    """Call one tool and read what the loop keeps from its result."""
    try:
        result = await mcp.call_tool(name, arguments)
    except MCPError as exc:
        return ToolOutput(f"error: {exc}", [], None)
    text = " ".join(getattr(block, "text", "") for block in result.content).strip()
    if result.is_error:
        return ToolOutput(f"error: {text or 'the tool failed'}", [], None)
    payload = result.structured_content
    if payload is None:
        return ToolOutput(text, [], None)
    content = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    clauses = _clauses(payload) if name == SEARCH_TOOL else []
    return ToolOutput(content, clauses, _no_version_note(name, payload))


def _no_version_note(name: str, payload: Any) -> str | None:
    if name not in {SEARCH_TOOL, VERSION_TOOL} or not isinstance(payload, dict):
        return None
    note = payload.get("note")
    if payload.get("version_no") is None and isinstance(note, str) and note.strip():
        return note.strip()
    return None


def _clauses(payload: Any) -> list[RetrievedClause]:
    rows = payload.get("clauses") if isinstance(payload, dict) else None
    clauses: list[RetrievedClause] = []
    for row in rows if isinstance(rows, list) else []:
        try:
            clauses.append(
                RetrievedClause(
                    clause_ref=str(row["clause_ref"]),
                    title=str(row["title"]),
                    text=str(row["text"]),
                    version_no=int(row["version_no"]),
                    score=float(row["score"]),
                )
            )
        except (KeyError, TypeError, ValueError):
            continue
    return clauses


def _finish(
    text: str,
    executed: list[ToolCall],
    retrieved: dict[str, RetrievedClause],
    notes: list[str],
) -> AgentAnswer:
    parsed = parse_final(text)
    if parsed is None:
        return AgentAnswer(
            answer=text.strip(),
            citations=[],
            tool_calls=executed,
            uncited=True,
            clauses=list(retrieved.values()),
            policy_notes=notes,
            stop_reason="unparsed",
        )
    answer, cited, rejected = parsed
    kept = [ref for ref in cited if ref in retrieved]
    dropped = [ref for ref in cited if ref not in retrieved] + rejected
    return AgentAnswer(
        answer=answer,
        citations=kept,
        tool_calls=executed,
        uncited=bool(dropped) or not kept,
        dropped_citations=dropped,
        clauses=list(retrieved.values()),
        policy_notes=notes,
    )


_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)
# "4.2", "Clause 4.2", "section 4.2", "clause_ref: 4.2", "4.2 Glass breakage".
_REF = re.compile(r"(?:(?:clause_ref|clause|section)\s*:?\s*|§\s*)?(\d+\.\d+)\b", re.IGNORECASE)
# A clause named in the answer's prose; the keyword is required so amounts are not read as refs.
_MENTION = re.compile(r"\b(?:clause_ref|clause|section)\s*:?\s*(\d+\.\d+)\b", re.IGNORECASE)


class FinalMessage(NamedTuple):
    answer: str
    # Clause refs, still to be checked against what was retrieved.
    citations: list[str]
    # Citation entries that were not strings, as JSON; they count as dropped citations.
    rejected: list[str]


def parse_final(text: str) -> FinalMessage | None:
    """The answer and cited refs from the model's final message, or None if there are none.

    Small models wrap JSON in code fences, put a sentence before it, write a single citation as
    a bare string, or write "Clause 4.2"; all of that is accepted. The first JSON object with a
    non-empty string `answer` wins. A citation must be a string: a number is not trusted as a
    ref (4.10 arrives as 4.1), so it is rejected and counts as a dropped citation. Strings that
    are not clause refs (a rule id such as R1_NAME_MATCH) are ignored. When the list holds no
    clause ref but the answer names clauses in its text ("excluded under clause 6.2"), those
    are taken as its citations; they are checked against the retrieved clauses like any other.
    """
    fenced = _FENCE.search(text)
    candidates = [fenced.group(1)] if fenced else []
    candidates.append(text)
    for candidate in candidates:
        found = _first_answer_object(candidate)
        if found is not None:
            answer, citations = found
            refs, rejected = _refs(citations)
            return FinalMessage(answer, refs or _unique(_MENTION.findall(answer)), rejected)
    return None


def _first_answer_object(text: str) -> tuple[str, Any] | None:
    decoder = json.JSONDecoder()
    for start in (i for i, ch in enumerate(text) if ch == "{"):
        try:
            value, _ = decoder.raw_decode(text, start)
        except ValueError:
            continue
        if isinstance(value, dict) and isinstance(value.get("answer"), str):
            answer = value["answer"].strip()
            if answer:
                return answer, value.get("citations")
    return None


def _refs(raw: Any) -> tuple[list[str], list[str]]:
    """(clause refs, rejected non-string entries)."""
    items = raw if isinstance(raw, list) else [raw] if raw is not None else []
    refs: list[str] = []
    rejected: list[str] = []
    for item in items:
        if not isinstance(item, str):
            rejected.append(json.dumps(item))
            continue
        match = _REF.match(item.strip())
        if match:
            refs.append(match.group(1))
    return _unique(refs), rejected


def _unique(refs: list[str]) -> list[str]:
    return list(dict.fromkeys(refs))
