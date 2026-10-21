"""The gate CI runs on the holdout split, exercised with stub stages against RELEASE_CRITERIA.md."""

from pathlib import Path

import pytest

from evals import run
from evals.context import EvalContext, StageRunner
from evals.gates import load_gates
from evals.report import StageResult
from evals.run import main
from gateway.base import CassetteMissError

ROOT = Path(__file__).parents[2]
STAGES = "classification,extraction,retrieval,validation,agent"

# Every gated metric at a value that clears its gate, by stage.
PASSING: dict[str, dict[str, float]] = {
    "classification": {"classification.accuracy": 0.97},
    "extraction": {"extraction.f1": 0.93},
    "retrieval": {"retrieval.recall_at_5": 1.0, "retrieval.wrong_version_hits": 0.0},
    "validation": {"validation.accuracy": 1.0},
    "agent": {"agent.faithfulness": 0.95},
}


def _stub(metrics: dict[str, float]) -> StageRunner:
    def runner(split: str, ctx: EvalContext) -> StageResult:
        return StageResult(metrics=dict(metrics), details={"split": split})

    return runner


def _install(monkeypatch: pytest.MonkeyPatch, **overrides: dict[str, float]) -> None:
    for stage, metrics in PASSING.items():
        monkeypatch.setitem(run.STAGES, stage, _stub(overrides.get(stage, metrics)))


def _ci_argv(tmp_path: Path, *extra: str) -> list[str]:
    # The gates come from the real RELEASE_CRITERIA.md, as in CI.
    return [
        "--split",
        "holdout",
        "--stages",
        STAGES,
        "--gate",
        "--require-all-gates",
        "--gates",
        str(ROOT / "RELEASE_CRITERIA.md"),
        "--out-dir",
        str(tmp_path / "reports"),
        *extra,
    ]


def test_degraded_metric_fails_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _install(monkeypatch, classification={"classification.accuracy": 0.80})

    assert main(_ci_argv(tmp_path)) == 1

    out = capsys.readouterr().out
    failures = out.split("below gate:", 1)[1]
    assert "classification.accuracy" in failures
    assert "0.8000" in failures
    assert "extraction.f1" not in failures


def test_all_gates_met_passes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _install(monkeypatch)

    assert main(_ci_argv(tmp_path)) == 0

    assert "6 metric(s) checked" in capsys.readouterr().out


def test_a_wrong_version_hit_fails_the_max_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _install(
        monkeypatch,
        retrieval={"retrieval.recall_at_5": 1.0, "retrieval.wrong_version_hits": 1.0},
    )

    assert main(_ci_argv(tmp_path)) == 1

    failures = capsys.readouterr().out.split("below gate:", 1)[1]
    assert "retrieval.wrong_version_hits" in failures
    assert "<=" in failures


def test_a_skipped_stage_fails_when_all_gates_are_required(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _install(monkeypatch)
    argv = _ci_argv(tmp_path)
    argv[argv.index(STAGES)] = "classification,extraction,retrieval,validation"

    assert main(argv) == 1

    out = capsys.readouterr().out
    assert "not produced by this run" in out
    assert "agent.faithfulness" in out


def test_a_stage_that_produces_no_metric_fails_when_all_gates_are_required(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # The agent stage returns no metrics when the split has no questions.
    _install(monkeypatch, agent={})

    assert main(_ci_argv(tmp_path)) == 1

    assert "agent.faithfulness" in capsys.readouterr().out


def test_a_skipped_stage_is_not_checked_without_require_all_gates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install(monkeypatch)
    argv = _ci_argv(tmp_path)
    argv.remove("--require-all-gates")
    argv[argv.index(STAGES)] = "classification"

    assert main(argv) == 0


def test_release_criteria_gates_wrong_version_hits_at_zero() -> None:
    gates = load_gates(ROOT / "RELEASE_CRITERIA.md")

    hits = gates["retrieval.wrong_version_hits"]
    assert (hits.threshold, hits.direction) == (0.0, "max")
    assert all(
        g.direction == "min" for name, g in gates.items() if name != "retrieval.wrong_version_hits"
    )


def test_ci_workflow_runs_the_holdout_gate() -> None:
    workflow = (ROOT / ".github" / "workflows" / "evals.yml").read_text(encoding="utf-8")

    command = next(line for line in workflow.splitlines() if "evals.run" in line)
    assert "LLM_BACKEND=replay" in command
    assert "--split holdout" in command
    assert f"--stages {STAGES}" in command
    assert "--gate" in command.split()
    assert "--require-all-gates" in command.split()


def test_a_cassette_miss_fails_with_the_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def missing(split: str, ctx: EvalContext) -> StageResult:
        raise CassetteMissError("extract:PKT-003:doc_1.pdf")

    _install(monkeypatch)
    monkeypatch.setitem(run.STAGES, "extraction", missing)

    assert main(_ci_argv(tmp_path)) == 1

    err = capsys.readouterr().err
    assert "extract:PKT-003:doc_1.pdf" in err
    assert "extraction" in err
