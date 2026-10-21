# ADR-005: One adjuster agent over MCP

Status: accepted, Oct 21, 2026

## Context

Once a claim reaches `needs_review`, the adjuster has follow-up questions: "is glass covered for this claim?", "which rule is blocking it?", "is there an excess for theft?". Some are about the policy wording that applied on the incident date, some about what the pipeline found in the documents, and some need both. Unlike the pipeline, the order of lookups is not fixed in advance, so this is the one place where a model choosing tools earns its cost (ADR-001).

The risks are the usual ones for a tool-using model: it answers from general knowledge instead of the policy, cites a clause it never read, reads the wrong claim, or changes something it should only suggest. The default backend is a 7B model on a laptop (qwen2.5:7b-instruct through Ollama), which makes all four more likely.

## Decision

**One agent, one loop.** `claimcheck.agent.loop.answer_question` sends the question and the tool list to the gateway, runs the tool calls the model asks for, and stops at the first turn without tool calls or after 6 turns. The final message must be a JSON object `{answer, citations}`.

**Tools are an MCP server with an explicit allowlist.** `claimcheck.mcp_server.allowlist.ALLOWED_TOOLS` names seven tools: `get_claim_summary`, `list_claim_documents`, `get_extracted_fields`, `get_validation_results`, `search_policy_clauses`, `get_policy_version` and `flag_for_review`. `build_server` registers exactly those and fails at start-up if one has no implementation. The agent only reaches them through an MCP client session: in-process over the SDK's in-memory transport for the API and the evals, or `python -m claimcheck.mcp_server` on stdio for any other MCP client. Tool results are compact JSON with filenames and clause refs, never database ids.

**Read-only by default.** Six tools only read. The one write tool, `flag_for_review`, returns `confirmation_required` and changes nothing unless `human_confirmed` is true; when it is, it writes a `claim.flagged` audit row and still does not touch the claim's status, which only `transition()` changes.

**The loop decides what the model must not.** The agent answers about the claim it was opened on, so `claim_id` is removed from every schema the model sees and set by the loop on every call. `human_confirmed` is removed the same way and always sent as false, so the model can ask for a flag but never confirm one. `search_policy_clauses` takes no date: it always searches the version in force on the claim's incident date (ADR-002, ADR-004).

**Citations are checked, not trusted.** A cited clause ref counts only if `search_policy_clauses` returned it during the run. When the `citations` list holds no clause ref but the answer text names one ("excluded under clause 6.2"), the refs named in the text are taken as the citations and checked the same way. Anything else is dropped and the answer is marked `uncited`, as is an answer with no citation, one whose final message was not the JSON asked for, and one that ran out of turns ("I could not answer within the tool budget.").

**Everything is audited.** `/claims/{id}/ask` is adjuster-only and writes `agent.question` before the agent starts; every tool call the model attempts writes `agent.tool_call` with its arguments and outcome in its own transaction. The server does this where it dispatches calls, so a call to an unknown tool or with arguments that fail the tool's schema is recorded as `rejected` just like a call that succeeds. Calls the model proposes on its last turn are never run, since nothing would read their results; the API records them as `rejected: budget`. The HTTP reply carries the answer, the citations, `uncited` and the names of the tools used, not the tool payloads. For a cloud backend, the gateway is scoped to the claim's OCR text, so names that appear in tool results are redacted before they leave (ADR-006).

**Faithfulness is measured.** `evals/agent_questions.jsonl` holds 32 adjuster questions over 14 dev packets, plus 11 on holdout packets (see Consequences). On dev, 25 are about the policy, each with the clause that answers it, and 7 are about the claim's own data or about a claim with no version in force. The gated metric, `agent.faithfulness`, covers the policy questions only: an answer passes when it is not `uncited` and an LLM judge finds it faithful to the clauses it cites. The judge works statement by statement and must quote the clause words it relies on; the code then checks every statement of a policy answer, whatever label the judge gave it, for a quote that is really in the cited clauses and contains every amount the statement states. A quote may be loosely matched or may leave entries out of a list, but it may not skip over a sentence, and its amounts must be whole amounts in the clause text it matched. A policy answer with nothing to check is not faithful. When a policy tool reports that no version was in force, the judge is shown that note in place of clauses and may quote it. The claim-data questions get their own metric, `agent.claim_data_accuracy`, which is reported but not gated: the final message must be the JSON asked for, cite no clause that was not retrieved, and contain no policy statement the judge cannot support.

## Alternatives considered

- **Several agents** (a retriever, a claim reader, a writer that composes the answer). Each hand-off is another prompt that can lose a citation or a condition, and another set of model calls per question. The questions here need at most two or three tool calls, which one loop already handles in 2 to 3 turns. A second agent would add coordination failures without adding a capability.
- **Plain function calling without MCP.** Simpler by one dependency. MCP gives the allowlist a single place to live, lets the same tools serve the API, the evals and an external MCP client without a second implementation, and makes the agent's only access to data a protocol boundary that can be audited.
- **RAG only, no agent.** Clause questions alone could be one retrieval plus one generation call. But adjusters mix clause questions with claim questions ("why is this blocked, and does the policy cover it anyway?"), and routing those by hand is the tool choice the agent exists to make.
- **Letting the model pass the claim id and the confirmation flag.** It is what a generic MCP client does. Here it would mean a 7B model typing UUIDs and being able to confirm its own write. Pinning both in the loop costs nothing.
- **Trusting the model's citations.** The first run showed why not: two answers cited a rule id as a clause, and citation lists were left empty while the answer text named the clause.

## Consequences

Measured on dev with qwen2.5:7b-instruct for both the agent and the judge, from cassettes recorded against Ollama; runs 1 to 4 were scored over all 25 questions at the time and are recomputed here from each report's per-question details under the current definitions.

| Run | Change | faithfulness | citation hit | claim-data accuracy | Report |
| --- | --- | --- | --- | --- | --- |
| 1 | first prompt, yes/no judge | 0.44 | 0.83 | 0.14 | `evals/reports/2026-10-21-dev-7.json` |
| 2 | run 1 answers; stricter citation parsing, statement-level judge with quote check | 0.72 | 0.89 | 0.57 | `evals/reports/2026-10-21-dev-8.json` |
| 3 | run 1 answers; judge also checks the amounts in each statement | 0.61 | 0.89 | 0.57 | `evals/reports/2026-10-21-dev-9.json` |
| 4 | revised system prompt, answers recorded again; judge accepts a quote that starts with the clause header | 1.00 | 1.00 | 0.71 | `evals/reports/2026-10-21-dev-10.json` |
| 5 | run 4 answers; judge checks every statement of a policy answer whatever its label, judge calls recorded again | 1.00 | 1.00 | 0.71 | `evals/reports/2026-10-21-dev-11.json` |
| 5, replay | run 5 cassettes replayed (`LLM_BACKEND=replay`) on a freshly migrated and seeded database | 1.00 | 1.00 | 0.71 | `evals/reports/2026-10-21-dev-12.json` |
| 6 | 7 conditional policy questions added ("is it covered if ..."), 25 policy questions in all | 0.92 | 0.96 | 0.71 | `evals/reports/2026-10-21-dev-15.json` |
| 7 | run 6 answers; judge accepts a quote that leaves words out of a list, a loose match must keep its amounts, the no-version note is evidence; judge calls recorded again | 0.92 | 0.96 | 0.86 | `evals/reports/2026-10-21-dev-19.json` |
| 8 | run 7 code, prompt unchanged, agent and judge calls recorded again | 0.88 | 0.96 | 0.86 | `evals/reports/2026-10-21-dev-22.json` |
| 8, replay | run 8 cassettes replayed under the stricter quote check (no sentence skips, whole amounts) | 0.88 | 0.96 | 0.86 | `evals/reports/2026-10-21-dev-23.json` |

Most of the early gain came from the measuring side: a judge asked for a bare yes or no was not a usable instrument at this model size, so it now quotes its evidence and the code checks every quote and amount. The judge is the same 7B model as the agent and the dev questions were written while looking at these failures, so the dev numbers are optimistic; on the 8 holdout policy questions `agent.faithfulness` is 0.875 against the 0.90 gate (`evals/reports/2026-10-21-holdout-5.json`), and the one failure is an answer that turns a policy condition into a fact about the claim. [docs/failure-analysis.md](../failure-analysis.md) has the holdout diagnosis behind runs 6 to 8 and what it would take to close the gate.
