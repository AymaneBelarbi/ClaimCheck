"""Agent stage: does the adjuster agent answer from the clauses it retrieved?

Reads `agent_questions.jsonl` next to the data directory (`evals/agent_questions.jsonl` for the
default `evals/data`). Every question names a packet; a question belongs to the split its packet
is in. The holdout questions were written from the holdout packets' ground truth and the policy
text only. Their cassettes were recorded once, and once more after the judge fixes described in
`docs/failure-analysis.md`; prompts are not tuned against them.

Before asking, the runner makes sure each referenced packet is a processed claim in the seeded
database: a missing claim is created from the packet's files (stored in a temporary directory)
and run through `process_claim` with the eval gateway, so with `LLM_BACKEND=replay` the pipeline
answers from the same extraction cassettes as the extraction stage. A claim that is already
past processing is used as it is.

Each question is answered by `answer_question` over an in-process MCP connection to the real
tool server (cache keys `agent:{question_id}:turn{n}`), then judged by
`claimcheck.agent.judge.judge` against the text of the clauses the answer cites (cache key
`judge:{question_id}`).

Questions come in two kinds, scored separately:

- Policy questions have an expected clause. One passes when the answer is not `uncited` (the
  final message was the JSON asked for, it cites at least one clause, and every clause it
  cites was returned by a search during the run, which also means the agent searched) and the
  judge finds it faithful to the cited clauses. The judge checks every statement of these
  answers against quoted clause text (see `claimcheck.agent.judge`).
- Claim-data questions have no expected clause: the claim's own documents, fields and rules,
  or a claim with no policy version in force. One passes when the final message was the JSON
  asked for (not plain text, not the budget message), it cites no clause that no search
  returned, and the judge finds no unsupported policy statement in it. Citing nothing is
  expected. Apart from the note a policy tool returns when no version was in force, the judge
  does not see tool results, so the claim facts themselves are only spot-checked by hand from
  the report's `answers`.

Metrics:
    agent.faithfulness          policy questions: share that pass (gated)
    agent.citation_hit          policy questions: share whose citations include the expected clause
    agent.claim_data_accuracy   claim-data questions: share that pass (not gated)

Each failed question gets one failure kind, the first that applies: budget, bad_json, no_search,
hallucinated_citation, no_citation, judge_error, unfaithful (no_search and no_citation only
apply to policy questions).
"""

import json
import tempfile
import uuid
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anyio
from mcp.server.mcpserver import MCPServer
from sqlalchemy import func, select

from claimcheck.agent.judge import JudgeVerdict, judge
from claimcheck.agent.loop import SEARCH_TOOL, AgentAnswer, answer_question
from claimcheck.core.types import ClaimStatus
from claimcheck.db.models import Claim, ClaimDocument, Policy
from claimcheck.db.session import SessionFactory, engine_for
from claimcheck.gateway.base import LLMGateway, SchemaError
from claimcheck.gateway.redaction import scope_gateway
from claimcheck.mcp_server.server import build_server, claim_ocr_texts, connect, tool_sessions
from claimcheck.worker.audit import record
from claimcheck.worker.main import worker_sessions
from claimcheck.worker.pipeline_runner import PipelineDeps, process_claim
from claimcheck.worker.storage import Storage
from evals.context import EvalContext, StageUnavailable
from evals.dataset.types import AgentQuestion, GroundTruth
from evals.report import StageResult

ACTOR = "eval"
_PROCESSED = {
    ClaimStatus.NEEDS_REVIEW.value,
    ClaimStatus.APPROVED_FOR_REVIEW.value,
    ClaimStatus.REJECTED_INCOMPLETE.value,
}


@dataclass(frozen=True)
class QuestionScore:
    # A policy question (it has an expected clause), as opposed to a claim-data question.
    policy: bool
    passed: bool
    # None for questions without an expected clause.
    citation_hit: bool | None
    failure: str | None


def load_questions(path: Path) -> list[AgentQuestion]:
    return [
        AgentQuestion.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def score(
    question: AgentQuestion, answer: AgentAnswer, verdict: JudgeVerdict | None
) -> QuestionScore:
    faithful = verdict is not None and verdict.faithful
    if question.expected_clause_ref is not None:
        return QuestionScore(
            policy=True,
            passed=faithful and not answer.uncited,
            citation_hit=question.expected_clause_ref in answer.citations,
            failure=_failure(answer, verdict, policy=True),
        )
    return QuestionScore(
        policy=False,
        passed=faithful and answer.stop_reason == "answered" and not answer.dropped_citations,
        citation_hit=None,
        failure=_failure(answer, verdict, policy=False),
    )


def _failure(answer: AgentAnswer, verdict: JudgeVerdict | None, *, policy: bool) -> str | None:
    """Why a question failed, for the report; None when it passed."""
    if answer.stop_reason == "budget":
        return "budget"
    if answer.stop_reason == "unparsed":
        return "bad_json"
    if policy and not any(call.name == SEARCH_TOOL for call in answer.tool_calls):
        return "no_search"
    if answer.dropped_citations:
        return "hallucinated_citation"
    if policy and not answer.citations:
        return "no_citation"
    if verdict is None:
        return "judge_error"
    if not verdict.faithful:
        return "unfaithful"
    return None


def summarize(results: list[QuestionScore]) -> tuple[dict[str, float], dict[str, int]]:
    """(metrics, failure counts). A kind of question with none asked gives no metric, so no
    gate is checked for it."""
    metrics: dict[str, float] = {}
    policy = [r for r in results if r.policy]
    claim_data = [r for r in results if not r.policy]
    if policy:
        metrics["agent.faithfulness"] = sum(r.passed for r in policy) / len(policy)
        metrics["agent.citation_hit"] = sum(bool(r.citation_hit) for r in policy) / len(policy)
    if claim_data:
        metrics["agent.claim_data_accuracy"] = sum(r.passed for r in claim_data) / len(claim_data)
    failures = Counter(r.failure for r in results if r.failure is not None)
    return metrics, dict(sorted(failures.items()))


def run(split: str, ctx: EvalContext) -> StageResult:
    splits = json.loads((ctx.data_dir / "splits.json").read_text(encoding="utf-8"))
    in_split = set(splits[split])
    questions = [
        q
        for q in load_questions(ctx.data_dir.parent / "agent_questions.jsonl")
        if q.packet_id in in_split
    ]
    if not questions:
        return StageResult(
            metrics={}, details={"split": split, "questions": 0, "note": "no questions"}
        )

    # The roles the services use: claims are filed and the tools read as an adjuster on
    # claimcheck_app, and the pipeline runs as the worker on claimcheck_worker.
    sessions = tool_sessions(engine_for(ctx.settings.app_database_url), ACTOR)
    worker = worker_sessions(ctx.settings)
    claims = _ensure_claims(sessions, worker, ctx, sorted({q.packet_id for q in questions}))
    server = build_server(sessions, ctx.embedder, ACTOR)

    results: list[QuestionScore] = []
    answers: list[dict[str, Any]] = []
    for q in questions:
        claim_id = claims[q.packet_id]
        with sessions() as session:
            texts = claim_ocr_texts(session, claim_id)
        gateway = scope_gateway(ctx.gateway, texts)
        answer = anyio.run(_ask, server, claim_id, q, gateway)
        cited = [c for c in answer.clauses if c.clause_ref in answer.citations]
        try:
            # The scoped gateway: the answer can quote the claimant's name.
            verdict: JudgeVerdict | None = judge(
                q.question,
                answer,
                cited,
                gateway,
                policy_question=q.expected_clause_ref is not None,
                cache_key=f"judge:{q.id}",
            )
        except SchemaError:
            verdict = None
        result = score(q, answer, verdict)
        results.append(result)
        answers.append(_detail(q, answer, verdict, result))

    metrics, failures = summarize(results)
    details: dict[str, Any] = {
        "split": split,
        "questions": len(questions),
        "policy_questions": sum(q.expected_clause_ref is not None for q in questions),
        "failures": failures,
        "answers": answers,
    }
    return StageResult(metrics=metrics, details=details)


async def _ask(
    server: MCPServer, claim_id: uuid.UUID, question: AgentQuestion, gateway: LLMGateway
) -> AgentAnswer:
    async with connect(server) as mcp:
        return await answer_question(
            claim_id=claim_id,
            question=question.question,
            gateway=gateway,
            mcp=mcp,
            cache_key=f"agent:{question.id}",
        )


def _detail(
    q: AgentQuestion, answer: AgentAnswer, verdict: JudgeVerdict | None, result: QuestionScore
) -> dict[str, Any]:
    return {
        "id": q.id,
        "packet_id": q.packet_id,
        "question": q.question,
        "expected_clause_ref": q.expected_clause_ref,
        "answer": answer.answer,
        "citations": answer.citations,
        "dropped_citations": answer.dropped_citations,
        "uncited": answer.uncited,
        "stop_reason": answer.stop_reason,
        "tool_calls": [
            {"name": c.name, "arguments": {k: v for k, v in c.arguments.items() if k != "claim_id"}}
            for c in answer.tool_calls
        ],
        "retrieved": [c.clause_ref for c in answer.clauses],
        "faithful": verdict.faithful if verdict else None,
        "unsupported_statements": verdict.unsupported_statements if verdict else None,
        "passed": result.passed,
        "citation_hit": result.citation_hit,
        "failure": result.failure,
    }


def _ensure_claims(
    sessions: SessionFactory, worker: SessionFactory, ctx: EvalContext, packet_ids: list[str]
) -> dict[str, uuid.UUID]:
    """A processed claim for every packet, creating (in `sessions`) and processing (in
    `worker`) the ones not there yet."""
    with sessions() as session:
        if not session.scalar(select(func.count()).select_from(Policy)):
            raise StageUnavailable("no policies in the database; run make seed first")

    claims: dict[str, uuid.UUID] = {}
    with tempfile.TemporaryDirectory(prefix="claimcheck-eval-") as tmp:
        storage = Storage(Path(tmp))
        deps = PipelineDeps(storage=storage, embedder=ctx.embedder, llm=ctx.gateway, actor=ACTOR)
        for packet_id in packet_ids:
            claim_id, processed = _claim_for(sessions, ctx.data_dir, packet_id, storage)
            if not processed:
                process_claim(worker, claim_id, deps)
            claims[packet_id] = claim_id
    return claims


def _claim_for(
    sessions: SessionFactory, data_dir: Path, packet_id: str, storage: Storage
) -> tuple[uuid.UUID, bool]:
    """(claim id, already processed). Files of an unprocessed claim are put in `storage`."""
    folder = data_dir / "packets" / packet_id
    with sessions() as session:
        claim = session.scalar(select(Claim).where(Claim.packet_id == packet_id))
        if claim is not None:
            if claim.status in _PROCESSED:
                return claim.id, True
            # Left unprocessed by an earlier run: storage keys are content hashes, so putting
            # the same files again recreates exactly the keys the rows point at.
            for document in session.scalars(
                select(ClaimDocument).where(ClaimDocument.claim_id == claim.id)
            ):
                storage.put(claim.id, document.filename, (folder / document.filename).read_bytes())
            return claim.id, False

        truth = GroundTruth.model_validate_json(
            (folder / "ground_truth.json").read_text(encoding="utf-8")
        )
        policy = session.scalar(select(Policy).where(Policy.policy_number == truth.policy_number))
        if policy is None:
            raise StageUnavailable(
                f"policy {truth.policy_number} is not in the database; run make seed first"
            )
        claim = Claim(
            packet_id=packet_id,
            policy_id=policy.id,
            claim_type=truth.claim_type.value,
            incident_date=truth.incident_date,
            created_by=policy.holder_id,
        )
        session.add(claim)
        session.flush()
        for doc in truth.documents:
            key = storage.put(claim.id, doc.filename, (folder / doc.filename).read_bytes())
            session.add(ClaimDocument(claim_id=claim.id, filename=doc.filename, storage_key=key))
        record(
            session,
            actor=ACTOR,
            action="claim.created",
            entity="claim",
            entity_id=str(claim.id),
            payload={
                "packet_id": packet_id,
                "policy_number": truth.policy_number,
                "documents": len(truth.documents),
            },
        )
        return claim.id, False
