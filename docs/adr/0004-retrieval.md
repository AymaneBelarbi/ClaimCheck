# ADR-004: Retrieval

Status: accepted, Oct 21, 2026

## Context

An adjuster or the agent asks a question such as "is a cracked windscreen covered?" about a claim. The answer has to quote the clause from the policy version in force on the incident date (ADR-002). Policy versions differ in small ways: glass is an exclusion in one version and a cover clause with a limit in the next, so the wrong version's clause reads as a plausible answer.

The corpus is small: a few policies, a few versions each, 15 clauses per version, 105 clauses in the development data. Clause text and OCR text from claim documents are personal data, and I do not want them sent to a third party just to be embedded.

## Decision

**pgvector in the same Postgres.** Clauses live in `policy_clauses` with a 384-dimension `vector` column and an HNSW index on cosine distance. The embeddings are a derived index: `python -m db.seed --reembed` rebuilds them from the policy markdown.

**Local embeddings.** `BAAI/bge-small-en-v1.5` through fastembed (ONNX, CPU), cached under `var/models`. Nothing is sent out for embedding, and a model change is a re-embed rather than a new vendor. The text embedded for a clause is `"{title}\n{text}"`; the queries are embedded as they are.

**The version filter is the version id, not a metadata filter.** `retrieve_clauses` calls `policy_in_force(policy_id, incident_date)`, then selects from `policy_clauses` filtered by the version id that `policy_in_force` returns, ordered by `embedding <=> query`, limited to k = 5. The score returned is `1 - distance`. No version in force gives an empty list. The search never ranks a clause from another version, so it cannot cite one.

Chunking is one clause per `## <ref> <title>` section; clauses are short enough that splitting them further would only separate a rule from its limit.

## Alternatives considered

- **A separate vector database.** Better at very large collections, but it adds a service and a second copy of the policy data. Version dates would have to be mirrored into it as metadata and kept in step with `policy_versions`, which is where the exclusion constraint from ADR-002 lives. With a few hundred clauses there is no scale problem for it to solve.
- **Version dates as a metadata filter in the vector store.** That moves the "which version applied" logic out of the one tested function and into a second implementation. Filtering by the version id that `policy_in_force` returns keeps a single definition, including its handling of gaps and overlaps.
- **A hosted embedding API.** Slightly better models, but clause text would leave the machine, and the evals would depend on network and price. The local model is good enough here (see below).
- **Keyword search only.** Cheaper, but weak when the question uses different words from the clause ("hire car" against "replacement transport"). It stays an option, since retrieval is measured on its own.
- **Retrieve across all versions, then filter.** Rejected in ADR-002, and measured below.

## Consequences

Measured on the dev questions (35 of the 49 in `evals/clause_questions.jsonl`; the split is a hash of the question id, 70/30; holdout is not used for tuning; report `evals/reports/2026-10-21-dev-13.json`):

| Metric | Value |
| --- | --- |
| recall@5 | 1.00 |
| recall@1 | 1.00 |
| MRR | 1.00 |
| wrong-version hits | 0 |

Two checks sit behind those numbers. Prefixing queries with the bge instruction ("Represent this sentence for searching relevant passages: ") changed nothing, so queries are embedded as they are. And the retrieval eval runs every question a second time with the filter removed (all versions of the policy searched together, `details.retrieval.control` in the report): a clause from the wrong version comes first for 21 of the 35 questions (`control_wrong_version_top1`), and 95 of the 175 returned clauses are from the wrong version (`control_wrong_version_hits`). Recall@5 is still 1.00 without the filter (`control_recall_at_5`) only because the right clause fits among the five; what the filter buys is that the first citation is from the right year.

These numbers are not a hard test. A version has 15 clauses and k is 5, so a third of it comes back every time, and the questions are phrased close to the clause topics. In a one-off manual measurement on Oct 21, 2026, not kept in a report, the mean gap in cosine distance between the best and second clause was 0.071 (smallest 0.014), so some questions are close calls that a different embedding model or longer clauses could tip. The set should grow with the policies. The gates are recall@5 at 0.85 and the wrong-version count at 0, a `max` gate that CI checks as an invariant; recall@1 is reported next to them without a gate.

The HNSW index matters once the planner chooses it: an approximate index scan applies the version filter after it has fetched its candidates, so it can return fewer than k rows. I have not checked which plan Postgres picks at the current size. If the table grows to where the planner uses the index, pgvector's iterative scan (`hnsw.iterative_scan`) is the setting to turn on, and the eval's recall number will show whether it matters.

Ingesting a version deletes and reinserts its clauses in the caller's transaction. That needs DELETE on `policy_clauses`, which the application role does not have; seeding and re-embedding run as the owner.
