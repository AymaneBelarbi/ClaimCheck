# Eval reports

Every `python -m evals.run` writes one dated report here (`<date>-<split>.json`, then `-2`, `-3`, ...
on the same day); `python -m evals.redaction_check` writes `<date>-redaction.json`. All of them
are kept, including runs nothing cites, so the history can be checked. Backend `replay` means
the model calls came from `evals/cassettes`; a recording run also says `replay`, because
recording passes each call through the replay layer to Ollama.

| Report | Run | Cited in |
| --- | --- | --- |
| `2026-10-20-dev.json` | classification on dev, first run | not cited |
| `2026-10-21-dev.json` | extraction on dev, baseline (F1 0.850) | ADR-003 table, failure-analysis |
| `2026-10-21-dev-2.json` | extraction on dev, value looked up when its quote joins label and value (F1 0.922) | failure-analysis, ADR-003 (as `dev*.json`) |
| `2026-10-21-dev-3.json` | extraction on dev, intermediate run between `dev-2` and `dev-4` | not cited |
| `2026-10-21-dev-4.json` | extraction on dev, the looser evidence variant that was not kept | failure-analysis |
| `2026-10-21-dev-5.json` | extraction on dev, value must sit inside the located quote (F1 0.921) | failure-analysis, ADR-003 (as `dev*.json`) |
| `2026-10-21-dev-6.json` | retrieval on dev, first run | not cited |
| `2026-10-21-dev-7.json` | agent on dev, run 1 | ADR-005, failure-analysis |
| `2026-10-21-dev-8.json` | agent on dev, run 2 (run 1 answers, new judge) | ADR-005, failure-analysis (in the range `dev-7` to `dev-12`) |
| `2026-10-21-dev-9.json` | agent on dev, run 3 (amount check) | ADR-005, failure-analysis (in the range `dev-7` to `dev-12`) |
| `2026-10-21-dev-10.json` | agent on dev, run 4 (revised prompt, recorded again) | ADR-005, failure-analysis (in the range `dev-7` to `dev-12`) |
| `2026-10-21-dev-11.json` | agent on dev, run 5 (every statement checked) | ADR-005, failure-analysis (in the range `dev-7` to `dev-12`) |
| `2026-10-21-dev-12.json` | agent on dev, run 5 replayed on a fresh database | ADR-005, failure-analysis |
| `2026-10-21-dev-13.json` | retrieval on dev, with the no-filter control | ADR-004 |
| `2026-10-21-dev-14.json` | agent on dev, the conditional questions first recorded | failure-analysis |
| `2026-10-21-dev-15.json` | agent on dev, run 6 | ADR-005, failure-analysis |
| `2026-10-21-dev-16.json` | agent on dev, no-version note in the judge's system prompt | failure-analysis |
| `2026-10-21-dev-17.json` | agent on dev, no-version note shown without an instruction | failure-analysis |
| `2026-10-21-dev-18.json` | agent on dev, between the `dev-17` and `dev-19` judge attempts | not cited |
| `2026-10-21-dev-19.json` | agent on dev, run 7 (judge changes) | ADR-005, failure-analysis |
| `2026-10-21-dev-20.json` | agent on dev, first prompt rule for conditional questions (reverted) | failure-analysis |
| `2026-10-21-dev-21.json` | agent on dev, second prompt rule (reverted) | failure-analysis |
| `2026-10-21-dev-22.json` | agent on dev, run 8 (original prompt recorded again) | ADR-005, failure-analysis |
| `2026-10-21-dev-23.json` | agent on dev, run 8 replayed under the stricter quote check | README, ADR-005, failure-analysis |
| `2026-10-21-dev-24.json` | extraction and validation on dev with rule R7, including `validation.end_to_end_accuracy` | README |
| `2026-10-21-holdout.json` | holdout recording run against Ollama (classification, extraction, agent) | README (cost table), failure-analysis |
| `2026-10-21-holdout-2.json` | first full holdout gate run | failure-analysis |
| `2026-10-21-holdout-3.json` | holdout agent and judge recorded again after the judge changes | failure-analysis |
| `2026-10-21-holdout-4.json` | holdout gate run after that recording | failure-analysis |
| `2026-10-21-holdout-5.json` | holdout gate run, stricter quote check, same cassettes | ADR-005, failure-analysis |
| `2026-10-21-holdout-6.json` | holdout gate run with rule R7 and the end-to-end validation score (macOS, Tesseract 5.5.3) | README, ADR-006 |
| `2026-10-21-holdout-7.json` | the same gate command inside the project image (Debian, Tesseract 5.5.0) | ADR-006 |
| `2026-10-21-redaction.json` | `python -m evals.redaction_check --split all`, all 50 packets | README, ADR-006, `gateway/redaction.py` |
