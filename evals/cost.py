"""Cost and latency per claim packet, from the usage a run collected.

A packet's numbers are sums over the model calls made while it was processed:

- cost is the sum of the calls' `cost_usd`;
- LLM latency is the sum of the calls' `latency_ms`, so it is time spent waiting on the model,
  not the time the packet took end to end;
- wall time is measured around the packet's whole processing (OCR included) by the eval stage.

Under `LLM_BACKEND=replay` the usage comes from the cassettes, which store what the recording
backend reported. The table then describes the backend and model that were recorded, not the
speed of the machine running the replay; the wall time is the only number measured live.
"""

import math
from collections.abc import Sequence
from typing import Any

from claimcheck.gateway.tracing import UsageLedger

# Nearest-rank percentile: the smallest value with at least p percent of the values at or below it.
_P95 = 95
_COST_DIGITS = 8


def p95(values: Sequence[float]) -> float:
    """Nearest-rank 95th percentile, 0.0 for no values."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = math.ceil(_P95 * len(ordered) / 100)
    return ordered[rank - 1]


def summarize_cost(
    ledger: UsageLedger, *, backend: str, model: str | None
) -> dict[str, Any] | None:
    """The report's `cost` section, or None when no packet was processed in the run."""
    if not ledger.packets:
        return None
    entries = list(ledger.packets.values())
    costs = [math.fsum(u.cost_usd for u in entry.usages) for entry in entries]
    latencies = [math.fsum(u.latency_ms for u in entry.usages) for entry in entries]
    calls = [len(entry.usages) for entry in entries]
    walls = [entry.wall_ms for entry in entries if entry.wall_ms is not None]
    packets = len(entries)
    return {
        "backend": backend,
        "model": model,
        "packets": packets,
        "avg_cost_usd_per_packet": round(math.fsum(costs) / packets, _COST_DIGITS),
        "p95_latency_ms_per_packet": round(p95(latencies), 1),
        "p95_wall_ms_per_packet": round(p95(walls), 1) if walls else None,
        "avg_llm_calls_per_packet": sum(calls) / packets,
    }
