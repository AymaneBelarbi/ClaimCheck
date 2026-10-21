# Changelog

All notable changes to this project are recorded here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses [Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-10-21

### Added

- Claim intake API (FastAPI) with JWT roles, idempotent uploads and a Postgres job queue with retries and a dead-letter state.
- Worker pipeline: Tesseract OCR with per-word confidence, embedding classifier with an LLM fallback, schema-constrained extraction with confidence and evidence, and seven cross-document validation rules.
- Policy versioning with a policy-in-force query and version-filtered clause retrieval over pgvector.
- Adjuster Q&A agent over an allowlisted MCP tool server, with every tool call audited.
- LLM gateway for Anthropic, OpenAI and Ollama, with name and ID redaction for cloud backends and replay cassettes for CI.
- Per-stage evals on a 50-packet synthetic dataset with a dev/holdout split and a release gate.
- Row-level security, OpenTelemetry tracing, a review UI, ADRs, a runbook and a failure analysis.

### Known issues

- Agent faithfulness on the holdout split is 0.875, below its 0.90 gate. See [docs/failure-analysis.md](docs/failure-analysis.md).

[0.1.0]: https://github.com/AymaneBelarbi/ClaimCheck/releases/tag/v0.1.0
