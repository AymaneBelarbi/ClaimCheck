# ClaimCheck

[![CI](https://github.com/AymaneBelarbi/ClaimCheck/actions/workflows/ci.yml/badge.svg)](https://github.com/AymaneBelarbi/ClaimCheck/actions/workflows/ci.yml)
[![Release gate](https://github.com/AymaneBelarbi/ClaimCheck/actions/workflows/evals.yml/badge.svg)](https://github.com/AymaneBelarbi/ClaimCheck/actions/workflows/evals.yml)
[![Python 3.12](https://img.shields.io/badge/python-3.12-3776ab.svg)](https://www.python.org/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

ClaimCheck checks an auto insurance claim packet (claim form, ID, invoice, incident report) against the policy version in force on the incident date and hands an adjuster a decision-ready file in which every verdict carries its evidence. It uses schema-constrained LLM extraction, version-filtered clause retrieval over pgvector and one adjuster agent over MCP tools.

![Architecture: upload, FastAPI, a Postgres job queue, the deterministic worker pipeline (OCR, classify, extract, validate), Postgres with pgvector, the LLM gateway, the adjuster agent behind POST /ask with its MCP tool server, the decision-ready file and the review UI](docs/architecture.png)

<!-- demo.gif: upload a packet, watch verdicts, ask a clause question -->

## Quick start

    docker compose up -d     # Postgres + pgvector, API, worker, Jaeger
    make demo                # files one packet, prints its review file, asks the agent a question

Needs Docker and [uv](https://docs.astral.sh/uv/). On a fresh clone `make demo` first generates the 50 synthetic packets (deterministic, from a fixed seed). No LLM or API key is needed: the stack starts with `LLM_BACKEND=replay`, which answers model calls from responses recorded with a local model (`evals/cassettes/`). The first run downloads the embedding model (`BAAI/bge-small-en-v1.5`, about 64 MB) from Hugging Face, once for the seed on the host and once into the containers' model volume.

<details>
<summary>What <code>make demo</code> prints</summary>

```
== uploading PKT-050: glass claim on AUT-104233, incident 2026-03-15, 3 documents
claim 63b36812-995e-40d0-9eb5-cd2c9e01248d queued

== waiting for the worker
received
processing
needs_review

== review file
policy version in force: v3 from 2026-01-01
  doc_1.pdf    claim_form
  doc_2.png    id_card
  doc_3.pdf    invoice
rule verdicts: {"found": 6, "conflicting": 1}
field verdicts: {"found": 14}
blocking rules:
  R3_INVOICE_WITHIN_LIMIT (conflicting): Invoice total 1,873.72 exceeds the limit of 1,500.00 by 373.72

== asking: What is the most the policy pays for broken glass on this claim?
answer: The policy pays up to $1,500 for broken glass, less the $100 glass excess.
citations: 4.2, 5.1
tools called: search_policy_clauses
```

All 14 fields were found, and one rule blocks the claim: the invoice is over the glass limit of the version in force. The answer cites the glass cover clause and the excess clause, both retrieved from that version.

</details>

## Measured quality

The release gate, run on the 15 held-out packets from the recorded model responses ([`evals/reports/2026-10-21-holdout-6.json`](evals/reports/2026-10-21-holdout-6.json)). Stages are scored separately, each against its own gate.

| Stage | Metric | Value | Gate | Result |
| --- | --- | ---: | ---: | --- |
| Classification | accuracy, 47 readable documents | 1.00 | ≥ 0.95 | pass |
| Extraction | field F1 (precision 0.944, recall 0.922) | 0.933 | ≥ 0.90 | pass |
| Retrieval | recall@5, 14 clause questions | 1.00 | ≥ 0.85 | pass |
| Retrieval | clauses returned from the wrong policy version | 0 | = 0 | pass |
| Validation | rule verdict accuracy on ground-truth fields, 93 rule checks | 1.00 | 1.00 | pass |
| Validation | rule verdict accuracy end to end, on the extracted fields, 81 rule checks | 1.00 | none | reported |
| Agent | faithfulness, 8 policy questions | 0.875 | ≥ 0.90 | **fail** |

Agent faithfulness is below its gate, so the `evals` workflow and `make gate` currently fail. On one holdout question the agent turned a policy condition into a fact about the claim, and two prompt changes tried on dev did not fix it; [docs/failure-analysis.md](docs/failure-analysis.md) has the investigation and what it would take to close it. Reported without a gate: retrieval MRR 1.00, agent citation hit 0.875, agent claim-data accuracy 0.667, and the share of unreadable scans that OCR flags for review, 1.00.

The gated validation number runs the rules on ground-truth fields, so it tests the rules against the dataset generator; the end-to-end row runs them on what extraction produced from the scans (real OCR and the recorded model output, documents typed by their ground truth; packets with a deliberately unreadable scan are left out, 2 of the 15 holdout packets and 3 of the 35 dev packets), and on dev it is 0.986, 209 of 212, where all three misses are invoice totals marked `low_confidence` and sent to a person ([`2026-10-21-dev-24.json`](evals/reports/2026-10-21-dev-24.json)).

### Cost and latency per packet

Generated with `python -m evals.cost_table evals/reports/2026-10-21-holdout.json`, the run that recorded the holdout cassettes against qwen2.5:7b-instruct on Ollama on a laptop:

| backend | model | avg cost / packet | p95 LLM latency / packet | p95 wall time / packet | avg LLM calls / packet |
| --- | --- | ---: | ---: | ---: | ---: |
| replay | qwen2.5:7b-instruct | $0.0000 | 271.8 s | 278.3 s | 3.3 |

The backend reads `replay` because recording passes each call through the replay layer to Ollama. Cost is zero because the model is local. LLM latency is the recorded model time per call, summed over the packet; wall time covers OCR and the extraction calls (classification fallbacks count in LLM latency only). Replayed from the cassettes, as CI does, the same packets take a p95 of 2.4 s each (`holdout-6`). No cloud backend has run with a real key yet, so there is no cloud cost row; the gateway prices every call (`gateway/pricing.py`).

## Nothing is ever silently guessed

Every extracted field and every rule ends in exactly one verdict:

- **found**: for a field, the code located the value in the OCR text with a confidence of at least 0.70; for a rule, the check passed on inputs that were all found.
- **missing**: the document does not show it, or a required document is absent.
- **conflicting**: documents disagree with each other or with the policy (a name mismatch, an invoice dated before the incident, a total over the limit).
- **low_confidence**: the value is there but cannot be trusted: a scan with mean OCR word confidence under 0.60, a field under 0.70, a date that reads either way round (`03/04/2026`), an extraction call that failed twice, a field the model could not find in a document too long to send it whole.

Each verdict carries its evidence: the document, the page, the character span in the OCR text and the source text. A field's confidence is the lower of the model's confidence and the OCR confidence under the value. If the code cannot find the value in the OCR text, the field is `low_confidence` whatever the model said.

Seven rules run, one file each in `pipeline/validate/`: the name on the ID matches the claimant (R1), a policy version was in force on the incident date (R2), the invoice total is within that version's limit (R3), the invoice is not dated before the incident (R4), the ID was valid on the incident date (R5), every document the claim type requires is there (R6), and the claim form, and the incident report if there is one, give the incident date the claim was filed with (R7). A rule only runs on inputs that were found; a missing or low-confidence input becomes the rule's verdict. Documents classified `other` are sent to a person and skip extraction. If no policy version was in force on the incident date, that is the verdict; the system does not fall back to the nearest version. Any rule that does not end `found` blocks the decision and is named in the review file's summary.

## Why each mechanism

Each capability gets the cheapest mechanism that does the job. One ADR per row, with the alternatives that were rejected.

| Capability | Mechanism | Why |
| --- | --- | --- |
| Packet intake and validation ([ADR-001](docs/adr/0001-architecture.md)) | Deterministic worker: OCR, classify, extract, validate in a fixed order; Postgres as the job queue; seven rules as plain classes | Reprocessing a packet gives the same review file, and the rules can be held to exactly 1.00 on ground-truth fields because no model is involved in them |
| Policy in force ([ADR-002](docs/adr/0002-policy-versioning.md)) | Immutable versions with half-open date ranges, an exclusion constraint against overlap, one `policy_in_force()` function | A claim is judged by the wording on the incident date, and the database cannot hold two versions for one day |
| Field extraction ([ADR-003](docs/adr/0003-extraction-and-confidence.md)) | One schema-constrained LLM call per document; the code locates each quoted value in the OCR text and caps confidence by OCR | Layouts vary too much for templates, and a fixed task gains nothing from an agent. The failure to prevent is a wrong value with high confidence |
| Clause questions ([ADR-004](docs/adr/0004-retrieval.md)) | pgvector in the same Postgres, local embeddings, filtered by the version id that `policy_in_force()` returns before ranking | With the filter removed, a clause from the wrong version ranks first for 8 of the 14 holdout questions |
| Adjuster follow-ups ([ADR-005](docs/adr/0005-single-agent-over-mcp.md)) | One tool-calling agent over an MCP allowlist; the loop pins the claim id and the confirmation flag; citations are checked against what was retrieved | Open-ended questions need tool choice. They need only a few tool calls, which one loop handles; more agents would mean more hand-offs to get wrong |
| Model backend ([ADR-006](docs/adr/0006-llm-gateway-and-redaction.md)) | One gateway interface; `LLM_BACKEND` picks anthropic, openai or ollama; redaction for cloud; recorded cassettes for CI | Cloud or on-prem is a deployment setting. CI measures the code against fixed model outputs, with no key and no network |

## Tracing

![Jaeger trace of one claim.process run against qwen2.5:7b-instruct on Ollama: three stage.ocr spans, stage.classify and stage.extract per document with an llm.call child under each extract, and stage.validate; the selected llm.call shows model, backend, input and output tokens, cost and latency](docs/images/trace.png)

Every packet is one trace. The API and the worker export OpenTelemetry spans to the bundled Jaeger (http://localhost:16686, service `claimcheck-worker`, operation `claim.process`); host commands such as the evals export only when `OTEL_ENDPOINT` is set. Each `llm.call` carries the model, token counts, cost, latency and purpose; prompts and document text are never put on a span. This trace is the demo packet processed live with `LLM_BACKEND=ollama` (qwen2.5:7b-instruct on a laptop), so the durations are real: the three extraction calls take nearly all of the time, OCR and classification under a second each. Under `replay` the same spans last milliseconds and `claimcheck.latency_ms` holds the recorded time. [docs/runbook.md](docs/runbook.md#reading-a-trace) explains every span.

## The review UI

![Review UI on PKT-050: not ready for decision, six of the seven rules found and R3_INVOICE_WITHIN_LIMIT conflicting with its evidence from doc_3.pdf open, and an answer to a glass limit question citing clauses 4.2 and 5.1](docs/images/review-ui.png)

One React page for adjusters: claims by status, the summary and blocking rules, the evidence behind every rule and field, questions to the agent (an uncited answer is flagged as such), and the approve or reject decision, which needs a written reason while rules are blocking. The screenshot leaves out the per-document field tables and the decision form.

    npm --prefix ui install      # once
    make token                   # an adjuster token for the compose stack
    make ui                      # http://localhost:5173

Paste the token into the header field; it is kept in `sessionStorage` for the tab.

## Repository layout

    api/            FastAPI routes, JWT auth, upload limits
    worker/         job queue consumer, pipeline orchestration, state machine, audit log
    pipeline/       ocr.py, classify.py, extract.py, schemas.py, validate/ (one file per rule)
    rag/            clause chunking, local embeddings, version-filtered retrieval
    agent/          the adjuster agent loop and the eval judge
    mcp_server/     MCP tool definitions and the allowlist
    gateway/        LLM backends (anthropic, openai, ollama), redaction, replay, tracing
    db/             migrations (schema, row-level security), policy_in_force.py, seed
    core/           settings and shared types, imported by every package
    evals/          dataset generator, one runner per stage, gates, redaction check, reports/, cassettes/
    ui/             the adjuster review page (React, Vite)
    scripts/        demo.py
    tests/          unit and database tests, mirroring the packages
    docs/           adr/, runbook.md, failure-analysis.md, architecture diagram
    RELEASE_CRITERIA.md
    docker-compose.yml

## Evaluation and release gates

- **Dataset.** `evals/generate_dataset.py` writes 50 synthetic packets with ground truth (expected fields and verdicts), including the failures the rules exist for: mismatched names, incidents outside coverage, expired IDs, missing documents, unreadable scans, totals over the limit. 35 packets are for development. The other 15 are held out for the gate and the final numbers; they were inspected once, for the diagnosis in [docs/failure-analysis.md](docs/failure-analysis.md), and their agent and judge cassettes were re-recorded once afterwards.
- **Per-stage metrics.** Classification accuracy per document type, extraction precision and recall per field, retrieval recall@5 and MRR within the version in force, rule verdict accuracy, and agent faithfulness. A judge model reviews each answer statement by statement and has to quote the clause it relies on; the code then checks each quote and every amount against the cited clause text, so every quoted clause and amount is verified in code. Whether a quote really supports its statement is still the judge's call.
- **Reports.** Every run writes a dated JSON report to [`evals/reports/`](evals/reports/) with per-type, per-field and per-question details. Extraction F1 on dev went from 0.850 to 0.921 ([ADR-003](docs/adr/0003-extraction-and-confidence.md)); agent faithfulness on dev is 0.88 on 25 policy questions (`2026-10-21-dev-23.json`).
- **Cassettes.** CI has no model, so it replays recorded model responses keyed by call; the report counts calls whose request has drifted from the recording, and a call that was never recorded fails the run ([ADR-006](docs/adr/0006-llm-gateway-and-redaction.md)).
- **CI.** Three jobs on every pull request: `checks` (ruff, ruff format, mypy strict, pytest against a pgvector Postgres), `evals` (every stage on holdout with `--gate --require-all-gates`, report uploaded as an artifact) and `ui` (vitest and a type-checked build). The gate fails on any metric below its gate in [RELEASE_CRITERIA.md](RELEASE_CRITERIA.md), and on a gated metric that is missing from the run. `make gate` runs the same thing locally.

## Switching the LLM backend

    LLM_BACKEND=ollama docker compose up -d
    LLM_BACKEND=anthropic ANTHROPIC_API_KEY=sk-ant-... docker compose up -d

`openai` works the same way with `OPENAI_API_KEY`. Models default to `claude-haiku-5-5`, `gpt-5-mini` and `qwen2.5:7b-instruct`; to change them, set `ANTHROPIC_MODEL`, `OPENAI_MODEL` or `OLLAMA_MODEL` in `.env`. Each backend requests its provider's structured-output mode, and every reply is validated against the Pydantic schema; a reply that does not fit is an error, and nothing is repaired. The anthropic and openai backends are covered by tests of their request bodies and response parsing, but have not yet run against a real key.

Cloud backends get redacted text. Before a request leaves, names and ID numbers are replaced with placeholders, and the reply is restored so evidence still matches the OCR text. Names are found after labels and across the documents of the same packet. Measured over the 50 generated packets (`python -m evals.redaction_check`, [`evals/reports/2026-10-21-redaction.json`](evals/reports/2026-10-21-redaction.json)), none of the 43 licence numbers leaked and a name got through in 16 of 143 document/name pairs; the residual and its causes are in [ADR-006](docs/adr/0006-llm-gateway-and-redaction.md). Ollama runs on the client's own machine and gets the raw text. `replay` is for CI, tests and the demo only.

## Security model

- **Roles.** HS256 JWTs carrying `sub`, `role` (`adjuster` or `customer`) and `exp`. A customer can file a claim on their own policy and read their own claims. The review file, the agent and decisions are adjuster-only.
- **Row-level security.** Forced on claims, documents, extracted fields and rule results. The API and the worker each log in with their own role holding only the grants they use, and each transaction states who is asking; a transaction that does not sees no rows, and a customer only ever sees claims created for them. The worker's `system` identity counts only on the worker's own login. `audit_log` and `jobs` have no row-level security: only the services read them, and the API returns neither.
- **Agent tools.** An explicit MCP allowlist of seven tools, six read-only. The one write tool, `flag_for_review`, does nothing without a human-confirmation flag, and the loop removes that flag and the claim id from what the model sees. The model can propose a flag but never confirm one, and it cannot reach another claim.
- **Audit log.** Append-only: the service roles can insert and read rows but cannot update or delete them. Every state change is written by one `transition()` function in the same transaction as the change; every agent question and every tool call the model attempts gets a row, including calls the tool server rejects and calls proposed on the last turn of the budget, which are never run and are recorded as `rejected: budget`.
- **Secrets.** From the environment only. `.env.example` is committed with placeholders; `.env` is ignored. The API does not start without `JWT_SECRET` and logs a warning when it is one of the published placeholders (the compose default or the `.env.example` value). Traces never hold prompts, document text or answers.

## Development

Host prerequisites beyond Docker and uv: Tesseract for the OCR tests and the evals (`brew install tesseract` or `apt install tesseract-ocr`), Ollama with `qwen2.5:7b-instruct` pulled for `make evals` and for recording cassettes, and Node 22 for the UI.

    make install                 # uv sync: dev tools and the dataset generator included
    make lint typecheck          # ruff check + format check, mypy --strict
    docker compose up -d db
    TEST_DATABASE_URL=postgresql+psycopg://claimcheck:claimcheck@localhost:5432/claimcheck make test
    npm --prefix ui test -- --run
    make evals                   # every stage on dev, against Ollama unless LLM_BACKEND says otherwise
    make gate                    # the CI release gate on holdout, from the cassettes
    uv run pre-commit install    # ruff and mypy on each commit

Without `TEST_DATABASE_URL` the database tests are skipped. `make record-holdout` and `make record-demo` re-record cassettes against Ollama; read the rules in [RELEASE_CRITERIA.md](RELEASE_CRITERIA.md) first.

## Further reading

- [docs/runbook.md](docs/runbook.md): start and stop, backup and restore, reprocessing a failed claim, rotating the token secret, reading a trace, refreshing cassettes.
- [docs/failure-analysis.md](docs/failure-analysis.md): the holdout agent failure and the extraction evidence rule, from symptom to fix.
- [docs/adr/](docs/adr/): the six decisions above, with the alternatives considered.
- [RELEASE_CRITERIA.md](RELEASE_CRITERIA.md): the gates and how they change.
- [CONTRIBUTING.md](CONTRIBUTING.md), [SECURITY.md](SECURITY.md) and [CHANGELOG.md](CHANGELOG.md).

## License

MIT. See [LICENSE](LICENSE).
