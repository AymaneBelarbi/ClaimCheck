"""Print the cost table of one or more eval reports as Markdown.

    python -m evals.cost_table evals/reports/2026-10-21-dev.json [more.json ...]

One row per report, from its `cost` section (written when the run processed packets).
"""

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

HEADER = (
    "| backend | model | avg cost / packet | p95 LLM latency / packet "
    "| p95 wall time / packet | avg LLM calls / packet |"
)
SEPARATOR = "| --- | --- | ---: | ---: | ---: | ---: |"


def _seconds(milliseconds: float | None) -> str:
    return "n/a" if milliseconds is None else f"{milliseconds / 1000:.1f} s"


def row(cost: dict[str, Any]) -> str:
    return (
        f"| {cost['backend']} | {cost['model'] or 'n/a'} "
        f"| ${cost['avg_cost_usd_per_packet']:.4f} "
        f"| {_seconds(cost['p95_latency_ms_per_packet'])} "
        f"| {_seconds(cost.get('p95_wall_ms_per_packet'))} "
        f"| {cost['avg_llm_calls_per_packet']:.1f} |"
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Print a Markdown cost table from eval reports.")
    parser.add_argument("reports", nargs="+", type=Path)
    args = parser.parse_args(argv)

    rows: list[str] = []
    for path in args.reports:
        cost = json.loads(path.read_text(encoding="utf-8")).get("cost")
        if cost is None:
            print(f"{path}: no cost section (the run processed no packets)", file=sys.stderr)
            return 2
        rows.append(row(cost))
    print("\n".join([HEADER, SEPARATOR, *rows]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
