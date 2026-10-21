import json
from pathlib import Path

import pytest

from claimcheck.core.settings import Settings, get_settings
from evals import run
from evals.context import EvalContext, StageRunner
from evals.report import StageResult
from evals.run import main


def _stub(value: float) -> StageRunner:
    def runner(split: str, ctx: EvalContext) -> StageResult:
        return StageResult(metrics={"classification.accuracy": value}, details={"split": split})

    return runner


def _argv(tmp_path: Path, *extra: str) -> list[str]:
    gates = tmp_path / "gates.md"
    gates.write_text("| metric | gate |\n| --- | --- |\n| classification.accuracy | 0.95 |\n")
    return [
        "--split",
        "dev",
        "--stages",
        "classification",
        "--gates",
        str(gates),
        "--out-dir",
        str(tmp_path / "reports"),
        *extra,
    ]


def test_gate_failure_exit_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setitem(run.STAGES, "classification", _stub(0.5))

    assert main(_argv(tmp_path, "--gate")) == 1

    assert "classification.accuracy" in capsys.readouterr().out


def test_gate_pass_exit_code(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(run.STAGES, "classification", _stub(0.96))

    assert main(_argv(tmp_path, "--gate")) == 0


def test_without_gate_flag_a_low_metric_does_not_fail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(run.STAGES, "classification", _stub(0.5))

    assert main(_argv(tmp_path)) == 0


def test_run_writes_a_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(run.STAGES, "classification", _stub(0.96))
    # The default backend, whatever a developer's .env says.
    monkeypatch.setattr(run, "get_settings", lambda: Settings(_env_file=None))

    main(_argv(tmp_path))

    (path,) = (tmp_path / "reports").glob("*-dev.json")
    report = json.loads(path.read_text())
    assert report["metrics"] == {"classification.accuracy": 0.96}
    assert report["stages"] == ["classification"]
    assert report["llm_backend"] == "ollama"


def test_unregistered_stage_is_a_clear_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(run, "KNOWN_STAGES", (*run.KNOWN_STAGES, "nonexistent"))
    argv = _argv(tmp_path)
    argv[argv.index("classification")] = "nonexistent"

    assert main(argv) == 2

    assert "nonexistent has no runner" in capsys.readouterr().err.replace("'", "")


def test_every_known_stage_has_a_runner() -> None:
    assert set(run.STAGES) == set(run.KNOWN_STAGES)


def test_unknown_stage_is_a_clear_error(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    argv = _argv(tmp_path)
    argv[argv.index("classification")] = "nonsense"

    assert main(argv) == 2

    assert "nonsense" in capsys.readouterr().err


def test_record_all_fallbacks_needs_a_recording_replay_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setitem(run.STAGES, "classification", _stub(0.96))
    monkeypatch.setenv("LLM_BACKEND", "replay")
    monkeypatch.setenv("REPLAY_RECORD", "false")
    get_settings.cache_clear()

    try:
        assert main(_argv(tmp_path, "--record-all-fallbacks")) == 2
    finally:
        get_settings.cache_clear()

    assert "REPLAY_RECORD=true" in capsys.readouterr().err
    assert not (tmp_path / "reports").exists()


def test_record_all_fallbacks_reaches_the_runner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[bool] = []

    def runner(split: str, ctx: EvalContext) -> StageResult:
        seen.append(ctx.record_all_fallbacks)
        return StageResult(metrics={"classification.accuracy": 1.0})

    monkeypatch.setitem(run.STAGES, "classification", runner)
    monkeypatch.setenv("LLM_BACKEND", "replay")
    monkeypatch.setenv("REPLAY_RECORD", "true")
    monkeypatch.setenv("REPLAY_INNER", "ollama")
    get_settings.cache_clear()

    try:
        assert main(_argv(tmp_path, "--record-all-fallbacks")) == 0
        assert main(_argv(tmp_path)) == 0
    finally:
        get_settings.cache_clear()

    assert seen == [True, False]
