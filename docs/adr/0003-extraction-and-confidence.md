# ADR-003: Extraction and confidence

Status: accepted, Oct 21, 2026

## Context

Once a document is classified, its fields have to be read out of OCR text: names, dates, plate numbers, invoice lines and totals. The text is imperfect (Tesseract drops accents, reads "I" as "|", scores small labels at zero confidence) and the documents come in a few layouts, so the same field sits in different places. Every field ends as `found`, `missing`, `conflicting` or `low_confidence`, and an adjuster has to be able to check it against the page.

The failure I care about most is a wrong value delivered with confidence. A field left empty costs an adjuster a look; a wrong plate or total that nobody questions costs a claim decision.

## Decision

**One schema-constrained call per document.** Each doc type has a Pydantic schema (`pipeline/schemas.py`) and the model returns, for every field, `value`, `confidence` and `source_text`, a quote from the document. The prompt tells the model to copy values as printed, to answer `null` when the document does not show the field, and not to fix OCR errors in the quote. `other` documents never reach this stage.

**The code checks what the model says.**
- The quote is located in the OCR text for its page and character span: exact match, then a case- and whitespace-tolerant match, then a fuzzy match (rapidfuzz `partial_ratio_alignment`, at least 85). Fuzzy matching only finds where the quote is. The value has to appear exactly inside the located span (dates and amounts after normalisation), so an OCR slip in the quote can move the span but cannot change the value.
- Confidence is `min(llm_confidence, ocr_span_confidence)`, where the OCR confidence is the mean word confidence under the value, not under its label. If the quote or the value cannot be found, the confidence is multiplied by 0.5. Below 0.70 the field is `low_confidence` and the value goes to the reviewer with its evidence.
- Dates and amounts are normalised in code. A date that reads either way round (`03/04/2026`) or an amount with a decimal comma comes back `low_confidence` with no value. The code does not pick an order or a convention.
- Invoice lines are checked against the invoice's own total, but only when the total itself was confirmed; a mismatch demotes the line items and never rewrites them.

**Null over guessing.** A missing field is `missing`. A field the model could not support is `low_confidence`. There is no fallback that fills a value from another field or from context.

**Failure is a result, not an exception.** A schema error or gateway error is retried once; after the second failure every field of the document is `low_confidence` with `details="extraction_failed"`, and the pipeline carries on to review.

## Alternatives considered

- **Regex and templates.** Deterministic and free, and I would use them if the layouts were fixed. They are not: the same invoice has its amounts in a column block in one layout and next to the description in another, and incident descriptions are free text. Every new layout is a new pattern to maintain.
- **A tool-calling agent per document.** It could retry and look around, but the task has a fixed shape (given this text, return these fields), results would vary between runs, and each document would cost several calls. Nothing in the task needs the model to choose its next step.
- **Free text from the model, parsed afterwards.** Needs a repair path for every malformed answer. Constrained output turns "malformed" into a typed error that the retry rule handles.
- **Trusting the model's confidence alone.** A 7B model is confident about values that OCR got wrong. Capping it by the OCR confidence under the value brings in a second, independent signal.

## Consequences

Measured on the dev split (118 scored documents, local qwen2.5:7b-instruct, gate 0.90 on holdout):

| Run | Precision | Recall | F1 |
| --- | --- | --- | --- |
| Baseline | 0.922 | 0.789 | 0.850 |
| Look the value up when its quote joins label and value, OCR confidence under the value, line items against the total | 0.935 | 0.909 | 0.922 |
| Value must be inside the located quote, per-row line-item evidence, total check only on a confirmed total | 0.937 | 0.906 | 0.921 |

The first jump came without any prompt change, on the same recorded model outputs. The model's quotes often joined a label to a value that the OCR text does not keep adjacent, and the label's low OCR confidence (Tesseract scored a small `DL NO.` label at 0.00) dragged correct values below the threshold. The last step gave back a little recall on purpose. A looser variant scored 0.922 but let a wrong name through with high confidence, and I kept the rule that cannot do that. Reports are in `evals/reports/` (`2026-10-21-dev*.json`).

What remains is mostly OCR: lost accents in names and "|" for "I" in descriptions, at high Tesseract confidence. The extractor cannot see these, because `min(llm, ocr)` only catches what the OCR itself doubted. Fixing it belongs in the OCR stage. The name rule in validation compares accent-stripped names, which covers part of it.

Precision is counted over `found` values only. A `low_confidence` value costs recall and never counts as a wrong answer, because a person looks at it. The 0.70 threshold and the 0.5 penalty are fixed values; they were not moved to improve a number.

The prompt text is part of the recorded replay cassettes, so changing it means re-recording the extraction run against a real model (about 70 minutes locally).
