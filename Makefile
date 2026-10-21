.PHONY: install lint typecheck test up down migrate seed dataset evals gate record-holdout record-demo demo token ui

install:
	uv sync

lint:
	uv run ruff check .
	uv run ruff format --check .

typecheck:
	uv run mypy .

test:
	uv run pytest -q

up:
	docker compose up -d --build

down:
	docker compose down

# Applies migrations to the compose database from the host.
migrate:
	uv run alembic upgrade head

seed:
	uv run python -m db.seed --data evals/data

dataset:
	uv run python -m evals.generate_dataset --out evals/data --seed 20261020

STAGES := classification,extraction,retrieval,validation,agent

evals:
	uv run python -m evals.run --split dev --stages $(STAGES)

# What CI runs: the holdout split from the recorded cassettes, every gate required.
gate:
	LLM_BACKEND=replay uv run python -m evals.run --split holdout --stages $(STAGES) --gate --require-all-gates

# Records the holdout cassettes against local Ollama. Only for a deliberate refresh, in its own
# pull request (see RELEASE_CRITERIA.md); delete the old holdout cassettes first.
record-holdout:
	LLM_BACKEND=replay REPLAY_RECORD=true REPLAY_INNER=ollama uv run python -m evals.run \
		--split holdout --stages classification,extraction,agent --record-all-fallbacks

# evals/data is not committed; a fresh clone generates it once (deterministic, seeded).
# splits.json is written after the packets, so its presence means a complete dataset.
demo:
	@test -f evals/data/splits.json || $(MAKE) dataset
	uv run python scripts/demo.py

# Records the classification fallback for every document of the demo packet, so `make demo`
# under replay never misses when another Tesseract build tips an embedding margin. Uses a split
# file holding only that packet; the extraction and /ask cassettes are recorded by other runs.
DEMO_PACKET ?= PKT-050
record-demo:
	mkdir -p var/demo-data/packets
	ln -sfn "$(CURDIR)/evals/data/packets/$(DEMO_PACKET)" var/demo-data/packets/$(DEMO_PACKET)
	echo '{"dev": ["$(DEMO_PACKET)"], "holdout": []}' > var/demo-data/splits.json
	LLM_BACKEND=replay REPLAY_RECORD=true REPLAY_INNER=ollama uv run python -m evals.run \
		--split dev --stages classification --record-all-fallbacks \
		--data var/demo-data --out-dir var/demo-data/reports

# Prints a development JWT for the compose stack: signed with JWT_SECRET (environment or .env),
# or with compose's development default when it is unset, as the stack does.
# Override: make token ROLE=customer SUB=cust-1
ROLE ?= adjuster
SUB ?= adj-1
token:
	@uv run python -m api.tokens --role $(ROLE) --sub $(SUB) --compose-default

# The adjuster review UI on http://localhost:5173.
ui:
	npm --prefix ui run dev
