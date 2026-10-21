import json
from datetime import UTC, datetime
from pathlib import Path

from evals.report import StageResult, write_report

NOW = datetime(2026, 10, 20, 9, 30, tzinfo=UTC)


def _results() -> dict[str, StageResult]:
    return {
        "classification": StageResult(
            metrics={"classification.accuracy": 0.97},
            details={"per_doc_type": {"invoice": {"accuracy": 1.0}}},
        )
    }


def test_report_filename_increments(tmp_path: Path) -> None:
    first = write_report(_results(), "dev", tmp_path, now=NOW)
    second = write_report(_results(), "dev", tmp_path, now=NOW)
    third = write_report(_results(), "dev", tmp_path, now=NOW)
    holdout = write_report(_results(), "holdout", tmp_path, now=NOW)

    assert [p.name for p in (first, second, third, holdout)] == [
        "2026-10-20-dev.json",
        "2026-10-20-dev-2.json",
        "2026-10-20-dev-3.json",
        "2026-10-20-holdout.json",
    ]


def test_report_contents(tmp_path: Path) -> None:
    path = write_report(
        _results(),
        "dev",
        tmp_path,
        now=NOW,
        info={
            "llm_backend": "replay",
            "llm_model": "qwen2.5:7b-instruct",
            "embedding_model": "BAAI/bge-small-en-v1.5",
            "replay_drift_count": 0,
        },
    )

    report = json.loads(path.read_text())

    assert report == {
        "split": "dev",
        "stages": ["classification"],
        "generated_at": "2026-10-20T09:30:00+00:00",
        "llm_backend": "replay",
        "llm_model": "qwen2.5:7b-instruct",
        "embedding_model": "BAAI/bge-small-en-v1.5",
        "replay_drift_count": 0,
        "metrics": {"classification.accuracy": 0.97},
        "details": {"classification": {"per_doc_type": {"invoice": {"accuracy": 1.0}}}},
    }


def test_report_creates_missing_directory(tmp_path: Path) -> None:
    path = write_report(_results(), "dev", tmp_path / "nested" / "reports", now=NOW)

    assert path.exists()


def test_report_has_no_git_fields(tmp_path: Path) -> None:
    text = write_report(_results(), "dev", tmp_path, now=NOW).read_text()

    assert "commit" not in text and "git" not in text
