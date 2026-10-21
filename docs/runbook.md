# Runbook

Operating the docker compose stack: `db` (Postgres 16 with pgvector), `migrate` (runs once),
`api`, `worker`, `jaeger`, and an optional `ollama`. Commands are run from the repository root.
Host-side commands use `uv run` and the URLs in `.env` (see `.env.example`); without a `.env` the
defaults point at the compose database on `localhost:5432`.

## Start and stop

    make up                      # docker compose up -d --build
    docker compose ps -a         # migrate shows "Exited (0)" once it has applied the migrations
    curl -s localhost:8000/health
    make down                    # stops the stack; volumes (database, uploads, model) are kept

`make dataset && make demo` then files one generated packet and walks it through to an answer
(see `scripts/demo.py`). Logs: `docker compose logs -f worker` (one line per job attempt) and
`docker compose logs -f api`.

`docker compose down -v` also deletes the volumes. Take a backup first.

The services run as the unprivileged user `claimcheck` (uid 10001). Volumes created by an older
image that ran as root are owned by root, and the API then cannot store uploads. Hand them over
once:

    docker compose run --rm --no-deps --user root api chown -R claimcheck /data/storage /app/var/models
    docker compose restart api worker

### Which model answers

The services read `LLM_BACKEND` when they start; compose passes it through and defaults to
`replay`, which answers from the recorded cassettes in `evals/cassettes` (mounted read-only).
Replay knows only the calls that were recorded, so it is for the demo and CI. To switch:

    LLM_BACKEND=ollama docker compose up -d                                  # Ollama on the host
    LLM_BACKEND=anthropic ANTHROPIC_API_KEY=sk-ant-... docker compose up -d  # Claude, redacted

`up -d` recreates `api` and `worker` with the new value; check with
`docker compose logs worker | grep "LLM backend"`. For Ollama in a container instead of on the
host, add `--profile ollama` and `COMPOSE_OLLAMA_URL=http://ollama:11434`, and pull the model
once: `docker compose exec ollama ollama pull qwen2.5:7b-instruct`. Put the variables in `.env`
to make them stick. ADR-006 explains what each backend receives.

## Single worker, and what a lease means

The worker takes one job at a time with `SELECT ... FOR UPDATE SKIP LOCKED`, marks it `running`
and records the attempt. There is no heartbeat. A job left `running` for 15 minutes
(`claimcheck.worker.queue.LEASE_TIMEOUT`) is taken to belong to a worker that died, and the next lease picks
it up again: as `done` if its claim already reached `needs_review`, as `dead` if that was the
third attempt, otherwise as the next attempt.

That is safe with the one worker compose runs. With more than one, an attempt that is slow
rather than dead (a local model can take minutes per document) is leased a second time after 15
minutes and the packet is processed twice. The results are replaced, not duplicated, but the
work and the audit trail are doubled. Before scaling the worker out, add a heartbeat that
refreshes `jobs.updated_at` during an attempt, or raise the timeout above the slowest packet.

Stopping the worker with SIGTERM (`docker compose stop worker`) lets the job in hand finish.

## Backup

Two things hold state: the database and the `storage` volume with the uploaded files. Policy
embeddings are in the database but can be rebuilt (below); the embedding model volume is a
cache.

    mkdir -p backup
    docker compose exec -T db pg_dump -U claimcheck -d claimcheck --format=custom \
        > backup/claimcheck-$(date +%F).dump
    docker compose exec -T db pg_dumpall -U claimcheck --roles-only \
        > backup/roles-$(date +%F).sql
    docker compose run --rm --no-deps -v "$PWD/backup:/backup" --entrypoint tar worker \
        czf /backup/storage-$(date +%F).tgz -C /data/storage .

`backup/` is in `.gitignore`, but it is still inside the working copy. The dumps hold claim
data and the roles file holds password hashes, so on a real deployment write them to storage
outside the repository and the host, with access limited like the database itself.

The custom-format dump holds the schema, the data, the grants and the row-level security
policies. The roles file holds `claimcheck_app` and `claimcheck_worker` with their password
hashes; a new server needs those roles before the grants in the dump can be restored. Keep both
files with the storage archive: a claim row without its files can be read but not reprocessed.

## Restore

Restore into a new database, check it, then point the services at it. Restoring over the live
database is not needed and not recommended.

    docker compose exec -T db createdb -U claimcheck claimcheck_restore
    docker compose exec -T db pg_restore -U claimcheck -d claimcheck_restore --exit-on-error \
        < backup/claimcheck-2026-10-21.dump
    docker compose exec -T db psql -U claimcheck -d claimcheck_restore \
        -c "select version_num from alembic_version" \
        -c "select count(*) from policies" -c "select count(*) from claims"

On a fresh server, load the roles first:
`docker compose exec -T db psql -U claimcheck -d postgres < backup/roles-2026-10-21.sql` (the
line for the `claimcheck` superuser fails because it exists already; that is expected).

To make the restored database the live one, set `POSTGRES_DB=claimcheck_restore` in `.env` and
run `docker compose up -d`, or rename it in `psql` while the services are stopped
(`docker compose stop api worker`, then `ALTER DATABASE claimcheck RENAME TO claimcheck_old;`
and `ALTER DATABASE claimcheck_restore RENAME TO claimcheck;` from the `postgres` database). The
uploaded files go back with:

    docker compose run --rm --no-deps -v "$PWD/backup:/backup" --entrypoint tar worker \
        xzf /backup/storage-2026-10-21.tgz -C /data/storage

These steps were tried end to end. Restored next to the development database, the copy had the
same row counts, migration version 0002, the policy embeddings (the retrieval eval run against
it gave recall@5 1.00 and no wrong-version hits) and working row-level security for
`claimcheck_app`. On an empty pgvector server, `pg_restore` stops at the first `GRANT` until the
roles file is loaded, and restores cleanly after it. The rename swap and the storage restore
were done on a throwaway stack, after which the API listed the restored claims.

## Rebuilding the policy embeddings

The `policy_clauses.embedding` column is derived from the policy markdown. After changing the
embedding model or the text that is embedded, or if the vectors are suspect after a restore:

    uv run python -m evals.seed --data evals/data --reembed

It runs as the database owner (`DATABASE_URL`) in one transaction, so a failure leaves the old
vectors in place. Without `--reembed`, the seed only rewrites versions whose clauses differ from
the markdown; running it twice changes nothing. Check the result with
`uv run python -m evals.run --split dev --stages retrieval`.

## A claim that failed

A failed attempt sets the claim to `failed` and requeues the job with a backoff of 30 s, then
60 s. After the third failure the job is `dead` and the claim stays `failed` until an operator
acts; a claim whose attempts all failed before the pipeline could start it is moved to `failed`
then as well. Find them, with the reason (dead jobs are listed whatever their claim's status,
in case one predates that rule):

    docker compose exec -T db psql -U claimcheck -d claimcheck -c \
        "select c.id, c.packet_id, c.status, j.state, j.attempts, j.last_error
           from claims c join jobs j on j.claim_id = c.id
          where c.status = 'failed' or j.state = 'dead' order by j.updated_at"

`docker compose logs worker` has the traceback for each attempt, and the audit log has a
`job.failed` or `job.dead` row per attempt. Fix the cause first: a model backend that was down,
a missing API key, a cassette that was never recorded under `replay`. Then requeue it:

    docker compose exec worker python -m claimcheck.worker.reprocess <claim_id>

The job starts again from attempt 0 and a `job.requeued` audit row records who did it
(`--actor <name>`, default `operator`) and the previous error. It refuses a claim that is not
`failed` and a job that is still being retried.

## Rotating JWT_SECRET

Tokens are HS256 and signed with `JWT_SECRET`; the API holds no session state. Rotating the
secret invalidates every token at once.

1. Generate a new key:
   `python -c "import secrets; print(secrets.token_urlsafe(48))"`.
2. Put it in `.env` as `JWT_SECRET=...` (compose reads `.env`; the default in
   `docker-compose.yml` is for development only).
3. `docker compose up -d api`. Requests with an old token now get 401.
4. Issue new tokens: `make token ROLE=adjuster SUB=<user>`, which signs with the secret in
   `.env`. In a real deployment the identity provider issues them.

There is no overlap window with two valid keys. With 12-hour tokens, rotate at a quiet time.

## Reading a trace

The API, the worker and the evals export spans to Jaeger when `OTEL_ENDPOINT` is set; compose
sets it to the bundled Jaeger. Open http://localhost:16686, choose the service
`claimcheck-worker` and the operation `claim.process`, and search by tag, for example
`packet.id=PKT-050` (the upload's Idempotency-Key) or `claim.id=<uuid>`.

A processed claim is one trace:

- `claim.process`: the whole attempt, tagged `claim.id` and `packet.id`.
- `stage.ocr`, `stage.classify`, `stage.extract`: one each per document, in order;
  `stage.validate` once at the end.
- `llm.call`: a child of the stage that made it, with `gen_ai.request.model`, input and output
  tokens, `claimcheck.cost_usd`, `claimcheck.latency_ms`, `claimcheck.purpose` (`classify`,
  `extract`) and the cache key with the file name hashed. A `classify` call means the embeddings
  were unsure about that document.

A failed span has status error and `error.type`; the message itself is in the worker log, not in
the trace. A question to the agent is a trace of `claimcheck-api`: the HTTP request span
`POST /claims/{claim_id}/ask`, the `agent.ask` span under it, and inside that the MCP tool calls
and one `llm.call` per model turn. No prompt, document text or answer is put on a span. Under `replay` a span
lasts about a millisecond while `claimcheck.latency_ms` shows the recorded latency.

## Refreshing the cassettes

The cassettes in `evals/cassettes` are recorded model responses keyed by call: the dev and
holdout eval runs, and the demo's packet and question. They need recording again when a prompt,
an extraction schema, the agent's tools or the model changes; a non-zero `replay_drift_count`
in an eval report says some no longer match the request being sent.

- Holdout (what CI gates on): delete the holdout packets' and holdout questions' cassettes, then
  `make record-holdout` with Ollama running, then `make gate`. The procedure, and what to
  do when the new numbers miss a gate, are in `RELEASE_CRITERIA.md`.
- Dev: delete the affected files and run the stage with
  `LLM_BACKEND=replay REPLAY_RECORD=true REPLAY_INNER=ollama uv run python -m evals.run --split dev --stages <stage>`.
- Demo: `make record-demo` records the classification fallback for every document of the demo
  packet (`--record-all-fallbacks` over a split holding only that packet), so a different
  Tesseract build cannot cause a miss. For the `/ask` turns, delete the demo packet's `agent__`
  cassettes, start the stack recording, run `make demo`, then start it again with the defaults.
  `tests/scripts/test_demo_cassettes.py` fails if a call of the default demo is not recorded. Recording needs the cassette directory writable, so use
  an override file that is not committed, for example `record.yml`:

      services:
        api: &record
          environment: {LLM_BACKEND: replay, REPLAY_RECORD: "true", REPLAY_INNER: ollama}
          volumes: ["./evals/cassettes:/app/evals/cassettes"]
        worker: *record

  and `docker compose -f docker-compose.yml -f record.yml up -d`.

Recording only fills keys that are missing; a cassette that exists is replayed, so an
out-of-date one has to be deleted to be recorded again. A failed call is recorded as a failure
and fails again on replay.
