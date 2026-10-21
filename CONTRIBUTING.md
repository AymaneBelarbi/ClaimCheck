# Contributing

ClaimCheck is a solo project, but issues and pull requests are welcome. This page covers what a change needs before it can be merged.

## Setup

You need Python 3.12 with [uv](https://docs.astral.sh/uv/), Docker, Tesseract, and Node 22 for the review UI. Ollama with `qwen2.5:7b-instruct` is only needed to re-record model responses.

    make install                 # uv sync, including the dev and dataset groups
    uv run pre-commit install    # ruff and mypy on each commit
    docker compose up -d db      # Postgres with pgvector for the database tests

## Checks

    make lint typecheck
    TEST_DATABASE_URL=postgresql+psycopg://claimcheck:claimcheck@localhost:5432/claimcheck make test
    npm --prefix ui test -- --run

Without `TEST_DATABASE_URL` the database tests are skipped, so run them before opening a pull request. CI runs everything above in `.github/workflows/ci.yml`.

## Changes that move a metric

The eval gates live in [RELEASE_CRITERIA.md](RELEASE_CRITERIA.md) and run in `.github/workflows/evals.yml`. If your change affects classification, extraction, retrieval, validation or the agent:

- run the affected stages on the dev split (`make evals`) and attach the before and after report from `evals/reports/`;
- do not tune anything against the holdout split;
- a gate only changes in a pull request that says why, with the reports that justify it.

Changes to prompts or schemas invalidate recorded model responses. [docs/runbook.md](docs/runbook.md#refreshing-the-cassettes) explains how to refresh them.

## Commits and pull requests

Commit messages follow [Conventional Commits](https://www.conventionalcommits.org/) (`feat(pipeline): ...`, `fix(api): ...`). Keep a pull request to one change, describe what it does and how you checked it, and link the issue if there is one. Architectural decisions get a short record in `docs/adr/`.
