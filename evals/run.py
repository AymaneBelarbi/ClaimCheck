"""Run eval stages and write a report.

    python -m evals.run --split dev --stages classification [--gate] [--require-all-gates]

Each stage is a runner `(split, ctx) -> StageResult` registered in STAGES. With --gate the
exit code is 1 if any metric the run produced misses its gate in RELEASE_CRITERIA.md. With
--require-all-gates (what CI uses) a gated metric the run did not produce is a failure too,
so dropping a stage, or a stage that silently produces nothing, cannot pass the gate.

--record-all-fallbacks is for recording cassettes only (LLM_BACKEND=replay REPLAY_RECORD=true):
the classification stage then also records the model fallback for documents the embeddings
classified on their own, so a replay whose OCR text differs slightly (another Tesseract build)
still finds a cassette when its embedding margin tips the other way. The metric is unchanged.
"""

import argparse
import sys
from pathlib import Path

from core.settings import get_settings
from core.tracing import setup_tracing
from evals.context import EvalContext, StageRunner, StageUnavailable
from evals.gates import check_gates, load_gates
from evals.report import StageResult, write_report
from gateway.base import CassetteMissError

KNOWN_STAGES = ("classification", "extraction", "retrieval", "validation", "agent")


STAGES: dict[str, StageRunner] = {}


def _register_builtin_stages() -> None:
    from evals.runners import agent, classification, extraction, retrieval, validation

    STAGES["classification"] = classification.run
    STAGES["extraction"] = extraction.run
    STAGES["retrieval"] = retrieval.run
    STAGES["validation"] = validation.run
    STAGES["agent"] = agent.run


_register_builtin_stages()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run eval stages and write a dated report.")
    parser.add_argument("--split", choices=["dev", "holdout"], required=True)
    parser.add_argument(
        "--stages", required=True, help=f"comma-separated: {', '.join(KNOWN_STAGES)}"
    )
    parser.add_argument("--gate", action="store_true", help="exit 1 if a metric misses its gate")
    parser.add_argument(
        "--require-all-gates",
        action="store_true",
        help="implies --gate; also exit 1 if a gated metric was not produced",
    )
    parser.add_argument(
        "--record-all-fallbacks",
        action="store_true",
        help="while recording cassettes, record the classification fallback for every document",
    )
    parser.add_argument("--data", type=Path, default=Path("evals/data"))
    parser.add_argument("--out-dir", type=Path, default=Path("evals/reports"))
    parser.add_argument("--gates", type=Path, default=Path("RELEASE_CRITERIA.md"))
    args = parser.parse_args(argv)

    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    for stage in stages:
        if stage not in KNOWN_STAGES:
            print(
                f"unknown stage {stage!r}; known stages: {', '.join(KNOWN_STAGES)}", file=sys.stderr
            )
            return 2
        if stage not in STAGES:
            print(
                f"stage {stage!r} has no runner registered; available: {', '.join(sorted(STAGES))}",
                file=sys.stderr,
            )
            return 2

    settings = get_settings()
    if args.record_all_fallbacks and not (
        settings.llm_backend == "replay" and settings.replay_record
    ):
        print(
            "--record-all-fallbacks only records cassettes: "
            "set LLM_BACKEND=replay REPLAY_RECORD=true REPLAY_INNER=<backend>",
            file=sys.stderr,
        )
        return 2
    setup_tracing("claimcheck-evals", settings.otel_endpoint)
    ctx = EvalContext(args.data, settings, record_all_fallbacks=args.record_all_fallbacks)
    results: dict[str, StageResult] = {}
    for stage in stages:
        try:
            results[stage] = STAGES[stage](args.split, ctx)
        except StageUnavailable as exc:
            print(f"stage {stage!r} cannot run: {exc}", file=sys.stderr)
            return 2
        except CassetteMissError as exc:
            print(
                f"stage {stage!r} stopped: {exc}. The cassettes do not cover this run; "
                "see RELEASE_CRITERIA.md for recording them.",
                file=sys.stderr,
            )
            return 1

    path = write_report(
        results, args.split, args.out_dir, info=ctx.run_info(), cost=ctx.cost_summary()
    )
    metrics = {name: value for r in results.values() for name, value in r.metrics.items()}
    for name, value in metrics.items():
        print(f"{name:<40} {value:.4f}")
    print(f"report: {path}")

    if args.gate or args.require_all_gates:
        table = check_gates(metrics, load_gates(args.gates))
        failed = False
        if table.failures:
            print("\nbelow gate:\n" + table.render_failures())
            failed = True
        if args.require_all_gates and table.missing:
            print("\ngated but not produced by this run:\n" + table.render_missing())
            failed = True
        if failed:
            return 1
        print(f"\ngate: {len(table.rows)} metric(s) checked, all within their gates")
    return 0


if __name__ == "__main__":
    sys.exit(main())
