"""Release gates: parsed from the table in RELEASE_CRITERIA.md, compared against a run.

A gate row is `| metric | threshold | direction |`. `min` means the metric must be at or above
the threshold, `max` that it must be at or below it (a count that has to stay at zero). A row
without a direction is a minimum.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

Direction = Literal["min", "max"]

_ROW = re.compile(
    r"^\|\s*([a-z_]+\.[a-z0-9_]+)\s*\|\s*(\d+(?:\.\d+)?)\s*\|(?:\s*(min|max)\s*\|)?\s*$"
)
# A table row whose first cell names a metric: it has to parse as a gate, or the gate it was
# meant to be would silently go unchecked.
_METRIC_CELL = re.compile(r"^\|\s*[A-Za-z_]+\.[A-Za-z0-9_]+\s*\|")


@dataclass(frozen=True)
class Gate:
    threshold: float
    direction: Direction = "min"

    def passes(self, value: float) -> bool:
        if self.direction == "max":
            return value <= self.threshold
        return value >= self.threshold

    def __str__(self) -> str:
        return f"{'>=' if self.direction == 'min' else '<='} {self.threshold:.4f}"


@dataclass(frozen=True)
class GateRow:
    metric: str
    value: float
    gate: Gate

    @property
    def passed(self) -> bool:
        return self.gate.passes(self.value)


@dataclass(frozen=True)
class GateTable:
    rows: list[GateRow]
    # Gated metrics the run did not produce.
    missing: list[str] = field(default_factory=list)

    @property
    def failures(self) -> list[GateRow]:
        return [row for row in self.rows if not row.passed]

    def render_failures(self) -> str:
        width = max((len(r.metric) for r in self.failures), default=6)
        lines = [f"{'metric':<{width}}  {'value':>7}  {'gate':>9}"]
        lines += [f"{r.metric:<{width}}  {r.value:>7.4f}  {r.gate!s:>9}" for r in self.failures]
        return "\n".join(lines)

    def render_missing(self) -> str:
        return "\n".join(self.missing)


def load_gates(path: Path = Path("RELEASE_CRITERIA.md")) -> dict[str, Gate]:
    gates: dict[str, Gate] = {}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        match = _ROW.match(line.strip())
        if match:
            direction: Direction = "max" if match.group(3) == "max" else "min"
            gates[match.group(1)] = Gate(float(match.group(2)), direction)
        elif _METRIC_CELL.match(line.strip()):
            raise ValueError(
                f"{path}:{number}: cannot read gate row {line.strip()!r}; "
                "expected | metric | threshold | min or max |"
            )
    if not gates:
        raise ValueError(f"no gate table (| metric | gate | direction | rows) found in {path}")
    return gates


def check_gates(metrics: dict[str, float], gates: dict[str, Gate]) -> GateTable:
    """Compare the metrics a run produced against their gates.

    A gate for a metric the run did not produce is not a failure here, so a run of one stage
    does not fail for the stages it skipped; it is listed in `missing`, and a caller that
    needs every gate checked (CI) treats a non-empty `missing` as a failure.
    """
    return GateTable(
        rows=[GateRow(name, metrics[name], g) for name, g in gates.items() if name in metrics],
        missing=[name for name in gates if name not in metrics],
    )
