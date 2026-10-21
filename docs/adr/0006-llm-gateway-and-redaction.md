# ADR-006: One LLM gateway, the backend as configuration

Status: accepted, Oct 21, 2026

## Context

Four things call a model: extraction (one structured call per document), the classification fallback (structured), the adjuster agent (tool-calling chat) and, in the evals, the judge (structured). I want the backend to be a deployment choice: a cloud API where the client accepts that, a 7B model on their own hardware where they do not. Claim documents carry names and licence numbers, so whatever goes to a cloud provider has to be redacted first. And CI has to run the extraction and agent evals on every pull request with no model, no network and no API key.

## Decision

**One protocol, three backends, chosen by `LLM_BACKEND`.** `claimcheck.gateway.base.LLMGateway` has two methods: `structured(system, prompt, schema)`, which returns a validated Pydantic object with its usage (model, tokens, cost, latency), and `chat(system, messages, tools)`, which returns text or tool calls. Pipeline, agent and evals take an `LLMGateway` and never import a provider. `claimcheck.gateway.factory.get_gateway` builds `anthropic`, `openai` or `ollama` from settings; each is a thin client over httpx, not a vendor SDK. Failures come back as two exceptions: `GatewayError` (unreachable, error status, bad body) and `SchemaError` (the reply does not fit the schema, raw text attached). The callers decide what a failure means: extraction retries once and then marks every field `low_confidence`, classification routes the document to `other`.

Switching is one variable on the compose stack:

    LLM_BACKEND=ollama docker compose up -d
    LLM_BACKEND=anthropic ANTHROPIC_API_KEY=sk-ant-... docker compose up -d

**Structured output is enforced by the provider, then validated again.**
- Anthropic: `output_config.format` with a JSON schema. The schema goes through `strict_schema` (every object closed, every property required) and loses the keywords the format rejects (`minimum`, `maxLength` and the like); Pydantic still checks them on the reply. A forced `tool_choice` would do the same job but, per the API documentation (checked Oct 20, 2026), is a 400 on Sonnet 5.5, Opus 5.5 and Fable 5.1. `max_tokens` is 16,000 because thinking counts against it, and the request timeout is 180 s rather than the 60 s the other backends use; a reply that stops at the cap or is refused is a `SchemaError`, not a parse attempt.
- OpenAI: `response_format` of type `json_schema` with `strict: true`, the same `strict_schema` rewrite, and a refusal is a `SchemaError`. Strict mode cannot express a free-form `dict` field, so such a schema fails at request time naming the field; none of ours has one.
- Ollama: `format` set to the Pydantic JSON schema, which constrains decoding.
- In all three the text is validated with `model_validate_json`. Nothing is repaired.

**Sampling and thinking are set per model, not globally.** Per the API documentation (checked Oct 20, 2026), current Claude models answer 400 to `temperature=0`, so none is sent; the gpt-5 family accepts only its default temperature; older OpenAI models and Ollama get 0. Thinking is switched off where the API allows it (`disabled` on Haiku), set to `between_tools` on Sonnet 5.5, which rejects `disabled`, and left at the adaptive default on Opus 5.5 and Fable, which cannot turn it off. The gateway's `Message` has no place for thinking blocks, so they are never echoed back in a tool loop.

**Redaction for cloud backends only.** `get_gateway` wraps any backend whose `is_cloud` is true in `RedactingGateway`; Ollama runs on the client's machine and gets the raw text, which also gives it the best chance at extraction. Each request replaces names and ID numbers with placeholders and swaps them back in the reply, `source_text` quotes included, so evidence still matches the OCR text. Names are found after labels (Name, Claimant, Customer, Bill to, Officer and so on), and `scope_gateway(gateway, packet_texts)` adds every name labelled anywhere in the claim's packet, because a name labelled on the claim form is often a bare line on the licence. The worker, the API's `/ask` and the eval runners all scope per claim.

Measured by `python -m evals.redaction_check --split all` on the 50 generated packets (170 documents, OCR text, ground-truth names and licence numbers; report `evals/reports/2026-10-21-redaction.json`): 0 of 43 licence numbers leak. 16 of 143 document/name pairs leak (31 of 372 capitalized name-token occurrences): claim forms 5 of 53, licences 9 of 50, invoices 2 of 40. Without packet scope the same labels leave 44 of 143 pairs; with packet scope but only the six labels the redactor started with (Name, Full name, Insured, Claimant, Driver, Officer), 52 of 143. So the claim this system can make is "ID numbers, and every name that is labelled somewhere in the packet", not "all names". The known limits, which the docstring of `src/claimcheck/gateway/redaction.py` also lists:
- A name that never follows a label is not found, for example a third party named only in free text.
- Table-layout forms come out of OCR as a column of labels and then a column of values, so the name is not next to its label; it is caught only if another document labels it.
- A bare name line on a licence is caught only through packet scope. When the licence holder is not the claimant, the case a reviewer most needs to see, the holder's first name goes out.
- OCR errors defeat matching ("Iain" read as "lain"); tokens shorter than three letters are replaced only as part of the full name; names OCR returned in lowercase are not captured.
- Over-capture costs context, not safety: "Customer Receipt" hides "Receipt", common words used as names are redacted wherever they are capitalized.
- Addresses, phone numbers, plates and dates of birth are not redacted. Redaction covers names and ID numbers only.

**Replay cassettes for CI.** `LLM_BACKEND=replay` is not a production backend; it exists for CI, tests and the demo. `ReplayGateway` answers each call from a JSON file named after its cache key (`extract:PKT-003:doc_1.pdf`, `agent:AQ-026:turn2`). Recording (`REPLAY_RECORD=true REPLAY_INNER=ollama`) fills missing keys from the real backend, and records failures as well as answers: a `SchemaError` or `GatewayError` seen while recording is stored and raised again on replay, so a retry under the next key replays the same sequence. The price is that a transient timeout during recording is frozen into the cassette; that is what the run saw. A replayed key whose request no longer hashes to the recorded one is still served, and counted: every eval report carries `replay_drift_count`. A key that was never recorded raises `CassetteMissError` and fails the CI run with the key in the message. Redaction sits outside replay, so cassettes recorded from a cloud backend hold only redacted text. Tracing is outermost and puts model, tokens, cost and latency on an `llm.call` span, never prompt or output text.

## Alternatives considered

- **A provider SDK or a multi-provider library.** Less code, but the details that broke things here (which models reject `temperature`, which reject a forced tool choice, how thinking is switched off) are request fields I need to see and test. Three small httpx clients with request bodies asserted in tests are easier to keep honest, and the replay layer only has to wrap one interface.
- **Redacting for every backend.** Simpler to state, but it costs the local model context it does not need to lose, and the text never leaves the machine.
- **A live model in CI.** Measures the model rather than the change, needs a secret in pull request builds, costs money per push and is not repeatable. Mocking the model instead would measure nothing.
- **Recording at the HTTP level.** Keyed on bytes, so any header or field order change is a miss. Cache keys named after the packet, document and turn survive changes that do not change the request, and drift counting says when one did.

## Consequences

- CI measures the code against fixed model outputs. A prompt, schema or model change needs the holdout cassettes recorded again, which is a deliberate pull request (`RELEASE_CRITERIA.md`); a stale cassette shows up as drift, a missing one as a failed run.
- OCR differs between machines, so CI's prompts differ slightly from the recorded ones. The holdout cassettes were recorded on macOS (Tesseract 5.5.3); even the generated PNG scans differ pixel by pixel on Linux. Replaying the full holdout gate command in the project image (Debian, Tesseract 5.5.0; `evals/reports/2026-10-21-holdout-7.json`) served every call with no miss and counted 45 drifted calls. Every metric matched the same command on macOS (`2026-10-21-holdout-6.json`, 6 drifted agent turns whose tool results now include rule R7) except extraction F1, 0.9302 against 0.9327, because the code checks the recorded answers against different OCR text. A one-off manual run in an Ubuntu 24.04 container like the CI runner (Tesseract 5.3.4) on Oct 21, 2026, not kept as a report, also served every call. The holdout cassettes include the classification fallback for every document, so a shifted embedding margin cannot turn into a miss.
- The cloud backends are tested by asserting their request bodies and parsing documented response shapes (respx), not against live APIs: no key has been used in this project. The first run against a real account should be watched.
- Cost per packet comes from the gateway's usage records (`src/claimcheck/gateway/pricing.py`, checked Oct 20, 2026). Ollama costs nothing; a cloud backend's cost is reported per packet in every eval report.
- Opus and Fable keep thinking on, and their thinking blocks are dropped between agent turns. Haiku, the default, is not affected; using the others for the agent needs that checked first.
