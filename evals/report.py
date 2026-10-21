"""Dated JSON reports of an eval run, kept in evals/reports/ so progress can be compared."""

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass
class StageResult:
    # Keys are qualified, e.g. "classification.accuracy", so they line up with the gates.
    metrics: dict[str, float]
    details: dict[str, Any] = field(default_factory=dict)


def write_report(
    results: dict[str, StageResult],
    split: str,
    out_dir: Path,
    *,
    info: dict[str, Any] | None = None,
    cost: dict[str, Any] | None = None,
    now: datetime | None = None,
) -> Path:
    """Write `<YYYY-MM-DD>-<split>.json`; a same-day rerun gets `-2`, `-3`, ... instead.

    `info` carries run metadata (backend, models, replay drift) and is merged at the top level.
    `cost` is the cost section of `evals.cost.summarize_cost`, left out when the run has none.
    """
    moment = now or datetime.now(UTC)
    report: dict[str, Any] = {
        "split": split,
        "stages": list(results),
        "generated_at": moment.isoformat(),
        **(info or {}),
        **({"cost": cost} if cost is not None else {}),
        "metrics": {name: value for r in results.values() for name, value in r.metrics.items()},
        "details": {stage: r.details for stage, r in results.items()},
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{moment.date().isoformat()}-{split}"
    attempt = 1
    while True:
        name = f"{stem}.json" if attempt == 1 else f"{stem}-{attempt}.json"
        try:
            # "x" fails instead of overwriting, so concurrent runs cannot clobber each other.
            with (out_dir / name).open("x", encoding="utf-8") as handle:
                handle.write(json.dumps(report, indent=2) + "\n")
        except FileExistsError:
            attempt += 1
            continue
        return out_dir / name
