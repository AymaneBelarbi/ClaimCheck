# ADR-001: Architecture

Status: accepted, Oct 20, 2026

## Context

ClaimCheck takes a claim packet (claim form, ID, invoice, incident report), checks it against the policy version that was in force on the incident date, and gives an adjuster a decision-ready file. Scope is one insurer, auto claims only, synthetic data.

The work splits into four different problems: intake and validation, field extraction from messy scans, clause lookup, and open-ended follow-up questions. They do not need the same machinery. I want each one built with the cheapest mechanism that does the job, and I want to be able to say why for each.

Constraints that shape the choice: the same packet must give the same result every time, every verdict needs evidence an adjuster can check, the LLM backend must be swappable between a cloud API and a local model, and I am building and operating this alone.

## Decision

| Capability | Mechanism | Why not more |
| --- | --- | --- |
| Packet intake and validation | Deterministic Python workflow | Same input must give same output; auditable, testable, cheap |
| Field extraction from OCR text | One schema-constrained LLM call per document (Pydantic output + confidence) | Messy scans defeat regex; a full agent adds nothing to a fixed task |
| Clause questions (is glass covered?) | RAG over policy clauses in pgvector, filtered to the version in force on the incident date | Answers must cite the exact clause that applied, not the newest one |
| Adjuster follow-ups | One tool-calling agent over MCP | Open-ended questions need tool choice; multiple agents add cost and failure modes without new capability |

**The pipeline is deterministic.** The worker runs OCR, classify, extract, validate in a fixed order. No model decides what runs next. The only model calls are inside stages with a typed output (extraction, and a classification fallback when embedding similarity is ambiguous). Validation is a rule engine: each rule is a small class with an id and one verdict, and no LLM is involved. This is what makes validation accuracy on ground-truth fields a number that should be exactly 1.00 in the evals, and what lets me replay a packet and get the same file. When a field is unreadable the pipeline records `low_confidence` or `missing`; it never guesses.

**One agent.** Adjuster questions are open-ended ("why was the invoice flagged, and does the policy cover the glass?"), so the model has to choose which lookup to make. That is the one place where tool choice earns its cost. The agent has a capped number of turns and an explicit allowlist of MCP tools, read-only except `flag_for_review`, which needs a human-confirmation flag. A second agent would add coordination failures without adding a capability, so there is not one.

**Postgres is the job queue.** The API inserts the claim row and the job row in the same transaction. Workers claim jobs with `SELECT ... FOR UPDATE SKIP LOCKED`, run them, and mark them done, retrying with backoff up to a fixed attempt count before the job goes `dead` and the claim `failed`. Packet id is the idempotency key.

Postgres is also the source of truth for policies, versions, claims, extracted fields and the audit log; pgvector holds clause embeddings as a derived index that can be rebuilt from it.

## Alternatives considered

- **A single agent for the whole packet.** Simplest to describe, and it would handle odd packets flexibly. Rejected: results would vary between runs, validation could not be tested as plain rules, and every packet would pay for many model calls on a task with a fixed shape.
- **Regex and templates for extraction.** Free and deterministic, but scanned invoices and free-text incident reports do not hold a layout. A schema-constrained call with per-field confidence handles them and still fails safely (`null` plus a low confidence).
- **Full-text or keyword search for clauses.** Cheaper than embeddings, but weak on questions phrased differently from the clause wording. Retrieval is measured separately (recall@5, MRR) so this can be revisited with data. Either way the filter on policy version is applied in SQL, not left to the model.
- **Multiple cooperating agents** (a reviewer, a retriever, a validator). Rejected for the reason above: more cost and more failure modes, no new capability.
- **Redis with Celery or RQ as the queue.** Mature and fast, but it adds a service to run, back up and monitor, and enqueueing becomes a second write that can disagree with the first. Throughput here is a handful of packets a minute, nowhere near where Postgres struggles.

## Consequences

- Adding a rule is adding a file in `src/claimcheck/pipeline/validate/`. Adding a doc type touches the classifier prototypes, one extraction schema and the required-documents table. Nothing else.
- Enqueueing is transactional with the claim row, so there is no state where a claim exists with no job or a job with no claim. The cost is that queue throughput is bounded by Postgres and polling adds up to one poll interval of latency. Both are fine at this scale; if volume grew by orders of magnitude, a dedicated broker would be the thing to revisit, and the job table is small enough to migrate away from.
- Retries and the dead state are plain rows, so a failed packet can be inspected and reprocessed with SQL.
- Extraction quality depends on the model, so it is gated by its own eval metric, and the backend switch (`LLM_BACKEND`) is a config change rather than a code change. Cloud backends only see redacted text; raw text goes to the local backend only.
- The deterministic pipeline cannot adapt to a document it was not built for. That is deliberate: an unrecognised document is classified `other` and sent to a person.
