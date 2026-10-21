import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from claimcheck.core.settings import get_settings
from claimcheck.core.types import DocType, Verdict
from claimcheck.gateway.base import LLMGateway, Usage
from claimcheck.gateway.tracing import PacketUsage, TracingGateway, UsageLedger
from claimcheck.pipeline.extract import FieldResult
from claimcheck.pipeline.types import OcrPage, OcrResult
from evals import cost_table, run
from evals.context import EvalContext
from evals.cost import p95
from evals.cost import summarize_cost as maybe_summarize
from evals.report import StageResult, write_report
from evals.runners import extraction
from tests.evals.test_extraction_metrics import _gt, _write_packet
from tests.fakes import FakeGateway, HashingEmbedder


def usage(cost: float, latency: float = 100.0) -> Usage:
    return Usage(model="m", input_tokens=10, output_tokens=5, cost_usd=cost, latency_ms=latency)


def summarize_cost(ledger: UsageLedger, *, backend: str, model: str | None) -> dict[str, Any]:
    cost = maybe_summarize(ledger, backend=backend, model=model)
    assert cost is not None
    return cost


def ledger_of(*packets: list[Usage], wall_ms: float | None = None) -> UsageLedger:
    ledger = UsageLedger()
    for number, usages in enumerate(packets, start=1):
        ledger.packets[f"PKT-{number:03d}"] = PacketUsage(list(usages), wall_ms)
    return ledger


def test_p95_is_nearest_rank() -> None:
    assert p95([float(n) for n in range(1, 21)]) == 19.0
    assert p95([float(n) for n in range(20, 0, -1)]) == 19.0  # order does not matter
    assert p95([5.0]) == 5.0
    assert p95([1.0, 2.0]) == 2.0
    assert p95([]) == 0.0


def test_average_cost_per_packet() -> None:
    ledger = ledger_of([usage(0.002)], [usage(0.004)])

    cost = summarize_cost(ledger, backend="anthropic", model="claude-haiku-5-5")

    assert cost["avg_cost_usd_per_packet"] == 0.003
    assert cost["backend"] == "anthropic"
    assert cost["model"] == "claude-haiku-5-5"


def test_a_packet_is_the_sum_of_its_calls() -> None:
    ledger = ledger_of(
        [usage(0.001, 100), usage(0.001, 300), usage(0.002, 200)],
        [usage(0.002, 50)],
    )

    cost = summarize_cost(ledger, backend="ollama", model="m")

    assert cost["avg_cost_usd_per_packet"] == 0.003
    assert cost["avg_llm_calls_per_packet"] == 2.0
    # per-packet latency is the sum of its calls' latency: 600 and 50
    assert cost["p95_latency_ms_per_packet"] == 600.0
    assert cost["packets"] == 2


def test_a_packet_without_llm_calls_counts_as_free_and_instant() -> None:
    ledger = ledger_of([usage(0.004, 400)], [])

    cost = summarize_cost(ledger, backend="replay", model="m")

    assert cost["avg_cost_usd_per_packet"] == 0.002
    assert cost["avg_llm_calls_per_packet"] == 0.5


def test_wall_time_p95_is_taken_over_the_timed_packets() -> None:
    ledger = UsageLedger()
    for number in range(1, 21):
        ledger.packets[f"p{number}"] = PacketUsage([], float(number))
    ledger.packets["untimed"] = PacketUsage()

    cost = summarize_cost(ledger, backend="replay", model="m")

    assert cost["p95_wall_ms_per_packet"] == 19.0


def test_no_packets_gives_no_cost_section() -> None:
    assert maybe_summarize(UsageLedger(), backend="replay", model="m") is None


def test_report_carries_the_cost_section(tmp_path: Path) -> None:
    cost = summarize_cost(ledger_of([usage(0.002)]), backend="replay", model="m")

    path = write_report(
        {"extraction": StageResult(metrics={"extraction.f1": 0.9})}, "dev", tmp_path, cost=cost
    )

    report = json.loads(path.read_text())
    assert report["cost"]["avg_cost_usd_per_packet"] == 0.002
    assert set(report["cost"]) >= {
        "backend",
        "model",
        "avg_cost_usd_per_packet",
        "p95_latency_ms_per_packet",
        "p95_wall_ms_per_packet",
        "avg_llm_calls_per_packet",
    }


def test_report_without_cost_has_no_cost_key(tmp_path: Path) -> None:
    path = write_report({"x": StageResult(metrics={})}, "dev", tmp_path)

    assert "cost" not in json.loads(path.read_text())


def test_cost_table_is_markdown(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    cost = summarize_cost(
        ledger_of([usage(0.0123, 1500.0), usage(0.0, 0.0)], wall_ms=2300.0),
        backend="anthropic",
        model="claude-haiku-5-5",
    )
    path = write_report({"x": StageResult(metrics={})}, "dev", tmp_path, cost=cost)

    assert cost_table.main([str(path)]) == 0

    lines = capsys.readouterr().out.strip().splitlines()
    assert lines[0] == (
        "| backend | model | avg cost / packet | p95 LLM latency / packet "
        "| p95 wall time / packet | avg LLM calls / packet |"
    )
    assert lines[1].startswith("| --- |")
    assert lines[2] == "| anthropic | claude-haiku-5-5 | $0.0123 | 1.5 s | 2.3 s | 2.0 |"


def test_cost_table_rejects_a_report_without_cost(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = write_report({"x": StageResult(metrics={})}, "dev", tmp_path)

    assert cost_table.main([str(path)]) == 2

    assert "no cost section" in capsys.readouterr().err


class _Out(BaseModel):
    text: str


def test_extraction_stage_fills_the_ledger_per_packet(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for packet_id in ("PKT-001", "PKT-002"):
        docs = [
            _gt("a.pdf", DocType.INVOICE, {"vendor": "V"}, Verdict.FOUND),
            _gt("b.pdf", DocType.ID_CARD, {"full_name": "N"}, Verdict.FOUND),
        ]
        _write_packet(tmp_path, packet_id, docs)
    (tmp_path / "splits.json").write_text(
        json.dumps({"dev": ["PKT-001", "PKT-002"], "holdout": []})
    )
    monkeypatch.setattr(
        extraction,
        "run_ocr",
        lambda data, filename: OcrResult(
            [OcrPage(1, filename)], [], filename, mean_confidence=0.9, low_confidence=False
        ),
    )

    def fake_extract(
        doc_type: DocType,
        ocr: OcrResult,
        document_id: str,
        llm: LLMGateway,
        *,
        cache_key: str | None = None,
    ) -> list[FieldResult]:
        llm.structured(
            system="s", prompt="p", schema=_Out, purpose="extract.x", cache_key=cache_key
        )
        return []

    monkeypatch.setattr(extraction, "extract", fake_extract)

    class Priced(FakeGateway):
        def structured(self, **kwargs):  # type: ignore[no-untyped-def]
            result = super().structured(**kwargs)
            return result.model_copy(update={"usage": usage(0.001, 40.0)})

    gateway = TracingGateway(Priced(structured_outputs=[_Out(text="x")]))
    ctx = EvalContext(tmp_path, get_settings(), gateway=gateway, embedder=HashingEmbedder())

    extraction.run("dev", ctx)

    assert sorted(ctx.ledger.packets) == ["PKT-001", "PKT-002"]
    for entry in ctx.ledger.packets.values():
        assert len(entry.usages) == 2
        assert entry.wall_ms is not None and entry.wall_ms > 0
    cost = ctx.cost_summary()
    assert cost is not None
    assert cost["avg_llm_calls_per_packet"] == 2.0
    assert cost["avg_cost_usd_per_packet"] == 0.002
    assert cost["p95_latency_ms_per_packet"] == 80.0


def test_eval_run_writes_the_cost_section(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def runner(split: str, ctx: EvalContext) -> StageResult:
        with ctx.ledger.packet("PKT-001"):
            pass
        return StageResult(metrics={"classification.accuracy": 1.0})

    monkeypatch.setitem(run.STAGES, "classification", runner)
    gates = tmp_path / "gates.md"
    gates.write_text("| metric | gate |\n| --- | --- |\n")

    run.main(
        [
            "--split",
            "dev",
            "--stages",
            "classification",
            "--gates",
            str(gates),
            "--out-dir",
            str(tmp_path / "reports"),
        ]
    )

    (path,) = (tmp_path / "reports").glob("*-dev.json")
    cost = json.loads(path.read_text())["cost"]
    assert cost["backend"] == "ollama"
    assert cost["avg_llm_calls_per_packet"] == 0.0
