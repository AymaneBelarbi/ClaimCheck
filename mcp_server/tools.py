"""The agent's tools, as plain functions over the database.

Each tool takes the claim id, checks that the claim exists, and returns a small JSON-ready
dict: filenames and clause refs rather than database ids, rounded scores, ISO dates. Auditing
happens one level up, in the server (`mcp_server.server`), so that calls which never reach a
tool (an unknown name, arguments that fail validation) are recorded too.

The docstrings are what the model reads as tool descriptions.
"""

import uuid
from collections.abc import Callable
from typing import Any

from mcp.server.mcpserver.exceptions import ToolError
from sqlalchemy.orm import Session

from db.models import Claim, Policy
from db.policy_in_force import policy_in_force
from db.session import SessionFactory
from rag.embedder import Embedder
from rag.retrieve import retrieve_clauses
from worker.audit import record
from worker.review_file import build_review_file

# Fixed project-wide: the agent sees the same five clauses the retrieval eval measures.
K = 5

ToolResult = dict[str, Any]


class ClaimTools:
    def __init__(self, sessions: SessionFactory, embedder: Embedder, actor: str) -> None:
        self._sessions = sessions
        self._embedder = embedder
        self._actor = actor

    def implementations(self) -> dict[str, Callable[..., ToolResult]]:
        """Every tool this module implements, by name. The server registers the allowlisted ones."""
        return {
            "get_claim_summary": self.get_claim_summary,
            "list_claim_documents": self.list_claim_documents,
            "get_extracted_fields": self.get_extracted_fields,
            "get_validation_results": self.get_validation_results,
            "search_policy_clauses": self.search_policy_clauses,
            "get_policy_version": self.get_policy_version,
            "flag_for_review": self.flag_for_review,
        }

    # ---- tools ----------------------------------------------------------------------------

    def get_claim_summary(self, claim_id: str) -> ToolResult:
        """Status, type and incident date of the claim, the policy version in force on that
        date, and which validation rules block a decision."""
        return self._run(claim_id, self._summary)

    def list_claim_documents(self, claim_id: str) -> ToolResult:
        """The claim's uploaded documents: filename, document type, classification confidence,
        and whether OCR flagged the scan as low confidence."""
        return self._run(claim_id, self._documents)

    def get_extracted_fields(self, claim_id: str) -> ToolResult:
        """Fields extracted from each document (names, dates, totals, ...) with their verdict
        (found, missing, conflicting, low_confidence) and the text they were read from."""
        return self._run(claim_id, self._fields)

    def get_validation_results(self, claim_id: str) -> ToolResult:
        """Result of every cross-document validation rule for the claim, with its verdict and
        message. Rules whose verdict is not `found` block the decision."""
        return self._run(claim_id, self._rules)

    def search_policy_clauses(self, claim_id: str, query: str) -> ToolResult:
        """Search the policy wording that applied to this claim (the version in force on its
        incident date) and return the 5 closest clauses with their clause_ref. Use short
        queries in the policy's words, e.g. "glass breakage limit" or "theft excess"."""
        return self._run(claim_id, lambda s, c: self._search(s, c, query))

    def get_policy_version(self, claim_id: str) -> ToolResult:
        """Which version of the policy was in force on the claim's incident date, and the
        dates that version covers."""
        return self._run(claim_id, self._version)

    def flag_for_review(
        self, claim_id: str, reason: str, human_confirmed: bool = False
    ) -> ToolResult:
        """Ask for the claim to be flagged for a second look. Nothing happens until a person
        confirms it; without that the result is `confirmation_required`."""
        return self._run(
            claim_id,
            lambda s, c: self._flag(s, c, reason, human_confirmed),
        )

    # ---- plumbing -------------------------------------------------------------------------

    def _run(self, claim_id: str, work: Callable[[Session, Claim], ToolResult]) -> ToolResult:
        with self._sessions() as session:
            return work(session, _claim(session, claim_id))

    # ---- work -----------------------------------------------------------------------------

    def _summary(self, session: Session, claim: Claim) -> ToolResult:
        review = build_review_file(session, claim.id)
        policy = session.get(Policy, claim.policy_id)
        version = review.policy_version
        return {
            "packet_id": claim.packet_id,
            "status": claim.status,
            "claim_type": claim.claim_type,
            "incident_date": claim.incident_date.isoformat(),
            "policy_number": policy.policy_number if policy else None,
            "policy_version_no": version.version_no if version else None,
            "documents": len(review.documents),
            "rule_verdicts": _counts(review.summary.verdict_counts),
            "field_verdicts": _counts(review.summary.field_verdict_counts),
            "blocking_rules": review.summary.blocking,
            "ready_for_decision": review.summary.ready_for_decision,
        }

    def _documents(self, session: Session, claim: Claim) -> ToolResult:
        review = build_review_file(session, claim.id)
        return {
            "documents": [
                {
                    "filename": d.filename,
                    "doc_type": d.doc_type.value if d.doc_type else None,
                    "classification_confidence": _round(d.classification_confidence),
                    "classification_method": d.classification_method,
                    "ocr_mean_confidence": _round(d.ocr_mean_confidence),
                    "low_confidence": d.low_confidence,
                    "error": d.error,
                }
                for d in review.documents
            ]
        }

    def _fields(self, session: Session, claim: Claim) -> ToolResult:
        review = build_review_file(session, claim.id)
        filenames = {d.id: d.filename for d in review.documents}
        return {
            "fields": [
                {
                    "document": filenames[f.document_id],
                    "doc_type": f.doc_type.value if f.doc_type else None,
                    "field_name": f.field_name,
                    "value": f.value,
                    "verdict": f.verdict.value,
                    "confidence": _round(f.confidence),
                    "page": f.evidence.page if f.evidence else None,
                    "source_text": f.evidence.text if f.evidence else None,
                }
                for f in review.fields
            ]
        }

    def _rules(self, session: Session, claim: Claim) -> ToolResult:
        review = build_review_file(session, claim.id)
        return {
            "rules": [
                {
                    "rule_id": r.rule_id,
                    "description": r.description,
                    "verdict": r.verdict.value,
                    "message": r.message,
                }
                for r in review.rules
            ],
            "blocking_rules": review.summary.blocking,
        }

    def _search(self, session: Session, claim: Claim, query: str) -> ToolResult:
        if not query.strip():
            raise ToolError("query must not be empty")
        version = policy_in_force(session, claim.policy_id, claim.incident_date)
        clauses = retrieve_clauses(
            session,
            policy_id=claim.policy_id,
            incident_date=claim.incident_date,
            query=query,
            embedder=self._embedder,
            k=K,
        )
        return {
            "incident_date": claim.incident_date.isoformat(),
            "version_no": version.version_no if version else None,
            "clauses": [
                {
                    "clause_ref": c.clause_ref,
                    "title": c.title,
                    "text": c.text,
                    "version_no": c.version_no,
                    "score": round(c.score, 4),
                }
                for c in clauses
            ],
            "note": None if version else _NO_VERSION,
        }

    def _version(self, session: Session, claim: Claim) -> ToolResult:
        policy = session.get(Policy, claim.policy_id)
        version = policy_in_force(session, claim.policy_id, claim.incident_date)
        result: ToolResult = {
            "policy_number": policy.policy_number if policy else None,
            "incident_date": claim.incident_date.isoformat(),
            "version_no": None,
            "effective_from": None,
            "effective_to": None,
            "note": _NO_VERSION,
        }
        if version is not None:
            result.update(
                version_no=version.version_no,
                effective_from=version.effective_from.isoformat(),
                effective_to=version.effective_to.isoformat() if version.effective_to else None,
                note=None if version.effective_to else "current version, no end date",
            )
        return result

    def _flag(self, session: Session, claim: Claim, reason: str, confirmed: bool) -> ToolResult:
        if not reason.strip():
            raise ToolError("reason must not be empty")
        if not confirmed:
            return {"status": "confirmation_required"}
        # A flag is a note on the claim's trail, not a status: only transition() moves status.
        record(
            session,
            actor=self._actor,
            action="claim.flagged",
            entity="claim",
            entity_id=str(claim.id),
            payload={"reason": reason},
        )
        return {"status": "flagged"}


_NO_VERSION = "no policy version was in force on the incident date, so no clause applies"


def _claim(session: Session, claim_id: str) -> Claim:
    try:
        parsed = uuid.UUID(claim_id)
    except ValueError:
        raise ToolError(f"claim {claim_id!r} not found: not a claim id") from None
    claim = session.get(Claim, parsed)
    if claim is None:
        raise ToolError(f"claim {claim_id} not found")
    return claim


def _round(value: float | None) -> float | None:
    return None if value is None else round(value, 3)


def _counts(counts: dict[Any, int]) -> dict[str, int]:
    return {str(getattr(k, "value", k)): v for k, v in counts.items()}
