# Release criteria

A change may be released only if every metric below is within its gate on the held-out
split: at or above it for `min`, at or below it for `max`. Metrics are measured per stage and
never blended into one score.

| metric | gate | direction |
| --- | --- | --- |
| classification.accuracy | 0.95 | min |
| extraction.f1 | 0.90 | min |
| retrieval.recall_at_5 | 0.85 | min |
| retrieval.wrong_version_hits | 0 | max |
| validation.accuracy | 1.00 | min |
| agent.faithfulness | 0.90 | min |

`retrieval.wrong_version_hits` counts retrieved clauses that belong to a policy version other
than the one in force on the incident date. It is an invariant, not a target: the version
filter in SQL (ADR-002, ADR-004) makes it 0, and any other value is a bug.

`agent.faithfulness` counts only the agent questions that have an expected policy clause: an
answer passes when it is the JSON asked for, cites at least one clause, every clause it cites
was retrieved during the run, and the judge finds each of its statements in the cited clause
text. Questions about the claim's own data are reported as `agent.claim_data_accuracy`, which
has no gate.

`validation.accuracy` runs the rules on the ground-truth fields of each packet and compares
their verdicts with the ground-truth verdicts. No model is involved, so anything below 1.00 is a
bug in a rule or in the dataset generator; it does not measure the system's decisions.
`validation.end_to_end_accuracy` runs the same rules on the fields the extraction stage produced
in the same run (real OCR, the recorded model output, documents typed by their ground truth) and
has no gate; it is produced only when the extraction stage runs first, as it does in the CI
command below, and leaves out packets with a scan that is unreadable by construction.

`classification.accuracy` counts only documents whose ground truth is readable. Scans that
are unreadable by construction are tracked separately as `classification.unreadable_flagged_rate`
(the share that OCR flags `low_confidence` and so routes to review); it has no gate yet.

## How the gates are enforced

CI (`.github/workflows/evals.yml`) runs every stage on the holdout split against a
freshly migrated and seeded Postgres, with model calls answered from the cassettes:

    LLM_BACKEND=replay uv run python -m evals.run --split holdout \
        --stages classification,extraction,retrieval,validation,agent --gate --require-all-gates

`make gate` runs the same command locally. The run exits with code 1 and prints the failing
metrics if any metric misses its gate, and, because of `--require-all-gates`, also when a gated
metric is missing from the run: a dropped stage or a stage that produced nothing cannot pass.
A cassette miss stops the run with the missing key in the error. The report is uploaded as the
job's artifact. `evals.gates.load_gates` reads the table above, so this file is the only place
the numbers live.

Every run also writes a dated report to `evals/reports/`. Development runs use the `dev`
split; the holdout split is held out for the CI gate and for final numbers. It was inspected
once, for the diagnosis in `docs/failure-analysis.md`, and its agent and judge cassettes were
re-recorded once afterwards.

## Changing a gate

Open a pull request that changes the table and nothing else about the evaluated behaviour.
The description must say why the gate is moving and include the before and after reports
from `evals/reports/`. Lowering a gate needs a written reason that is not "the system
cannot reach it"; raising one needs a report showing the system clears it.

## Replay cassettes

CI has no model to call, so it runs with `LLM_BACKEND=replay` and answers LLM calls from
the responses recorded in `evals/cassettes/`, one JSON file per call, named after its cache key
(`classify:`, `extract:`, `agent:` and `judge:` keys carry the packet or question id). Failed
calls are recorded too and fail again on replay, so a run replays exactly what the recording
saw. A recorded response is served whenever its key matches, even if the prompt sent now is
different; the report then shows it in `replay_drift_count`, the number of calls whose request
no longer matches the one recorded. A non-zero count on an unchanged prompt usually means OCR
text differs from the recording machine (another Tesseract build); after a prompt change it
means the cassettes are stale.

The holdout cassettes were recorded once with qwen2.5:7b-instruct on Ollama:

    make record-holdout

which runs the classification, extraction and agent stages on the holdout split with
`LLM_BACKEND=replay REPLAY_RECORD=true REPLAY_INNER=ollama` and `--record-all-fallbacks`. That
flag records the classifier's model fallback for every holdout document, not only for those
the embeddings were unsure about, because Tesseract on Linux reads a few characters
differently and can push a document's embedding margin under the threshold in CI. The
recording's own report is the holdout result on the real model; replaying it must give the
same metrics.

Refresh the holdout cassettes only when a prompt, an extraction schema, the agent's tools or
the model changes, never to move a number. It is a pull request of its own: delete the
holdout cassettes (`classify__`, `extract__`, `agent__` and `judge__` files of the holdout
packets and questions), run `make record-holdout`, then `make gate`, and attach both reports.
If the new holdout numbers miss a gate, the fix is made and measured on dev and the holdout
cassettes are recorded again afterwards. That has happened once: the holdout was inspected for
the diagnosis in `docs/failure-analysis.md`, and its agent and judge cassettes were re-recorded
after the changes.
