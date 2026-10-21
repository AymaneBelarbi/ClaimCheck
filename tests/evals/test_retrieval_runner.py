import json
from collections.abc import Callable
from pathlib import Path

import pytest
from alembic import command
from pydantic import SecretStr
from sqlalchemy.engine import URL
from sqlalchemy.orm import Session

from claimcheck.core.settings import Settings
from claimcheck.db.session import engine_for
from claimcheck.rag.retrieve import RetrievedClause
from evals import run
from evals.context import EvalContext, StageUnavailable
from evals.report import StageResult
from evals.runners import retrieval
from evals.runners.retrieval import split_of
from evals.seed import seed_database
from tests.db.conftest import alembic_config
from tests.fakes import HashingEmbedder

QUESTIONS_FILE = Path(__file__).parents[2] / "evals" / "clause_questions.jsonl"


def _policy_md(version: int, tag: str) -> str:
    sections = [f"## 1.{n} Clause {n}\n\nFacts about {tag}{n}." for n in range(1, 8)]
    return f"# Policy, version {version}\n\n" + "\n\n".join(sections) + "\n"


def _write_data(root: Path) -> Path:
    data = root / "data"
    (data / "policies" / "AUT-000001").mkdir(parents=True)
    # Version 2 uses different words, so a query can only match the era it names.
    (data / "policies" / "AUT-000001" / "v1.md").write_text(_policy_md(1, "old"))
    (data / "policies" / "AUT-000001" / "v2.md").write_text(_policy_md(2, "new"))
    index = {
        "policies": [
            {
                "policy_number": "AUT-000001",
                "holder_id": "cust-1",
                "versions": [
                    {
                        "version_no": 1,
                        "effective_from": "2025-01-01",
                        "effective_to": "2026-01-01",
                        "file": "AUT-000001/v1.md",
                    },
                    {
                        "version_no": 2,
                        "effective_from": "2026-01-01",
                        "effective_to": None,
                        "file": "AUT-000001/v2.md",
                    },
                ],
            }
        ]
    }
    (data / "policies" / "index.json").write_text(json.dumps(index))
    return data


def _ids(split: str, count: int) -> list[str]:
    found: list[str] = []
    n = 0
    while len(found) < count:
        n += 1
        if split_of(f"T-{n:03d}") == split:
            found.append(f"T-{n:03d}")
    return found


def _write_questions(root: Path, rows: list[tuple[str, str, str, int]]) -> None:
    """rows: (question text, incident date, expected clause ref, expected version)"""
    ids = _ids("dev", len(rows)) + _ids("holdout", 1)
    lines = [
        {
            "id": ids[i],
            "policy_number": "AUT-000001",
            "incident_date": day,
            "question": text,
            "expected_clause_ref": ref,
            "expected_version_no": version,
        }
        for i, (text, day, ref, version) in enumerate(rows)
    ]
    # One holdout question that would be a miss, to show it is not counted for dev.
    lines.append(
        {
            "id": ids[-1],
            "policy_number": "AUT-000001",
            "incident_date": "2025-03-01",
            "question": "old1",
            "expected_clause_ref": "1.7",
            "expected_version_no": 1,
        }
    )
    (root / "clause_questions.jsonl").write_text("\n".join(json.dumps(x) for x in lines))


@pytest.fixture
def seeded(make_database: Callable[[], URL], tmp_path: Path) -> tuple[EvalContext, Path]:
    url = make_database()
    command.upgrade(alembic_config(url), "head")
    data = _write_data(tmp_path)
    engine = engine_for(url.render_as_string(hide_password=False))
    with Session(engine) as session, session.begin():
        seed_database(session, data, HashingEmbedder())
    settings = Settings(
        jwt_secret=SecretStr("x" * 40), database_url=url.render_as_string(hide_password=False)
    )
    return EvalContext(data, settings, embedder=HashingEmbedder()), tmp_path


pytestmark = pytest.mark.db


def test_metrics_for_known_ranks(seeded: tuple[EvalContext, Path]) -> None:
    ctx, root = seeded
    _write_questions(
        root,
        [
            ("old3", "2025-03-01", "1.3", 1),  # rank 1
            ("new5", "2026-02-01", "1.5", 2),  # rank 1, version 2
            ("old1 old2", "2025-03-01", "1.2", 1),  # tie on score, ref order: rank 2
            ("old1", "2025-03-01", "1.7", 1),  # nothing matches: ref order puts it 7th
        ],
    )

    result = retrieval.run("dev", ctx)

    assert result.metrics["retrieval.recall_at_5"] == pytest.approx(3 / 4)
    assert result.metrics["retrieval.recall_at_1"] == pytest.approx(2 / 4)
    assert result.metrics["retrieval.mrr"] == pytest.approx((1 + 1 + 0.5 + 0) / 4)
    assert result.metrics["retrieval.wrong_version_hits"] == 0
    assert result.details["questions"] == 4
    assert [m["expected_clause_ref"] for m in result.details["misses"]] == ["1.7"]
    assert [q["rank"] for q in result.details["per_question"]] == [1, 1, 2, None]


def test_holdout_questions_are_not_run_for_dev(seeded: tuple[EvalContext, Path]) -> None:
    ctx, root = seeded
    _write_questions(root, [("old3", "2025-03-01", "1.3", 1)])

    assert retrieval.run("dev", ctx).details["questions"] == 1
    assert retrieval.run("holdout", ctx).details["questions"] == 1


def test_clauses_from_another_version_are_counted(
    seeded: tuple[EvalContext, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    ctx, root = seeded
    _write_questions(root, [("old3", "2025-03-01", "1.3", 1)])

    def wrong_era(**_: object) -> list[RetrievedClause]:
        return [
            RetrievedClause("1.3", "Clause 3", "t", 2, 0.9),
            RetrievedClause("1.3", "Clause 3", "t", 1, 0.8),
            RetrievedClause("1.4", "Clause 4", "t", 2, 0.7),
        ]

    monkeypatch.setattr(retrieval, "retrieve_clauses", lambda session, **kw: wrong_era())

    result = retrieval.run("dev", ctx)

    assert result.metrics["retrieval.wrong_version_hits"] == 2
    # The right ref of the wrong version does not count as a hit; the right version does.
    assert result.details["per_question"][0]["rank"] == 2


def test_the_control_searches_every_version_of_the_policy(
    seeded: tuple[EvalContext, Path],
) -> None:
    ctx, root = seeded
    _write_questions(
        root,
        [
            ("old3", "2025-03-01", "1.3", 1),  # only version 1 has "old3": right either way
            ("new5", "2025-03-01", "1.5", 1),  # version 2 has "new5": wrong first without filter
        ],
    )

    result = retrieval.run("dev", ctx)

    control = result.details["control"]
    assert control["control_wrong_version_top1"] == {"count": 1, "total": 2}
    assert control["control_wrong_version_hits"]["returned"] == 10
    assert control["control_wrong_version_hits"]["count"] >= 1
    # With the filter nothing from version 2 comes back.
    assert result.metrics["retrieval.wrong_version_hits"] == 0


def test_an_empty_database_asks_for_seeding(
    make_database: Callable[[], URL], tmp_path: Path
) -> None:
    url = make_database()
    command.upgrade(alembic_config(url), "head")
    data = _write_data(tmp_path)
    _write_questions(tmp_path, [("old3", "2025-03-01", "1.3", 1)])
    settings = Settings(
        jwt_secret=SecretStr("x" * 40), database_url=url.render_as_string(hide_password=False)
    )
    ctx = EvalContext(data, settings, embedder=HashingEmbedder())

    with pytest.raises(StageUnavailable, match="make seed"):
        retrieval.run("dev", ctx)


def test_the_runner_reports_a_missing_policy_as_unseeded(
    seeded: tuple[EvalContext, Path],
) -> None:
    ctx, root = seeded
    _write_questions(root, [("old3", "2025-03-01", "1.3", 1)])
    rows = [json.loads(line) for line in (root / "clause_questions.jsonl").read_text().splitlines()]
    for row in rows:
        row["policy_number"] = "AUT-999999"
    (root / "clause_questions.jsonl").write_text("\n".join(json.dumps(r) for r in rows))

    with pytest.raises(StageUnavailable, match="AUT-999999"):
        retrieval.run("dev", ctx)


def test_run_cli_reports_an_unseeded_database_without_a_traceback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def unavailable(split: str, ctx: EvalContext) -> StageResult:
        raise StageUnavailable("no policies in the database; run make seed first")

    monkeypatch.setitem(run.STAGES, "retrieval", unavailable)

    code = run.main(
        ["--split", "dev", "--stages", "retrieval", "--out-dir", str(tmp_path / "reports")]
    )

    assert code == 2
    assert "make seed" in capsys.readouterr().err
    assert not (tmp_path / "reports").exists()


# --- the split --------------------------------------------------------------------------


def test_split_is_deterministic_and_covers_both_sides() -> None:
    assert split_of("CQ-001") == split_of("CQ-001")
    assert {split_of(f"CQ-{n:03d}") for n in range(1, 50)} == {"dev", "holdout"}


def test_question_file_splits_roughly_70_30() -> None:
    ids = [json.loads(line)["id"] for line in QUESTIONS_FILE.read_text().splitlines()]
    dev = sum(split_of(i) == "dev" for i in ids)

    assert 0.6 <= dev / len(ids) <= 0.8
    assert len(ids) - dev >= 10
