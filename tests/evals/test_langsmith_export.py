import uuid
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal

import pytest

from claimcheck.core.settings import get_settings
from evals import run
from evals.langsmith_export import collect_cases, publish
from evals.report import StageResult

NOW = datetime(2026, 11, 13, 9, 30, tzinfo=UTC)


def _answer(
    case_id: str, *, passed: bool, faithful: bool | None, hit: bool | None
) -> dict[str, Any]:
    return {
        "id": case_id,
        "packet_id": "PKT-001",
        "question": "What is the collision limit that applies to this claim?",
        "expected_clause_ref": "4.1",
        "answer": "The collision limit is $15,000 per claim.",
        "citations": ["4.1"],
        "retrieved": ["4.1", "5.1"],
        "stop_reason": "answered",
        "failure": None if passed else "unfaithful",
        "passed": passed,
        "faithful": faithful,
        "citation_hit": hit,
    }


def _results() -> dict[str, StageResult]:
    return {
        "retrieval": StageResult(
            metrics={"retrieval.recall_at_5": 0.5, "retrieval.mrr": 0.4},
            details={"per_question": [{"id": "CQ-001", "rank": 2}, {"id": "CQ-002", "rank": None}]},
        ),
        "agent": StageResult(
            metrics={"agent.faithfulness": 0.5},
            details={
                "answers": [
                    _answer("AQ-001", passed=True, faithful=True, hit=True),
                    _answer("AQ-002", passed=False, faithful=False, hit=None),
                ]
            },
        ),
        "validation": StageResult(metrics={"validation.accuracy": 1.0}, details={}),
    }


class FakeLangSmith:
    def __init__(self, existing_examples: list[uuid.UUID] | None = None) -> None:
        self.datasets: dict[str, SimpleNamespace] = {}
        self.examples: list[dict[str, Any]] = []
        self.projects: dict[str, dict[str, Any]] = {}
        self.runs: list[dict[str, Any]] = []
        self.feedback: list[tuple[uuid.UUID, str, float]] = []
        self.flushed = 0
        self._existing = existing_examples or []

    def has_dataset(self, *, dataset_name: str) -> bool:
        return dataset_name in self.datasets

    def read_dataset(self, *, dataset_name: str) -> SimpleNamespace:
        return self.datasets[dataset_name]

    def create_dataset(self, dataset_name: str, *, description: str) -> SimpleNamespace:
        dataset = SimpleNamespace(id=uuid.uuid4(), name=dataset_name)
        self.datasets[dataset_name] = dataset
        return dataset

    def list_examples(self, *, dataset_id: uuid.UUID) -> list[SimpleNamespace]:
        ids = self._existing + [e["id"] for e in self.examples]
        return [SimpleNamespace(id=i) for i in ids]

    def create_examples(self, *, dataset_id: uuid.UUID, examples: list[dict[str, Any]]) -> None:
        self.examples.extend(examples)

    def create_project(
        self, project_name: str, *, reference_dataset_id: uuid.UUID, metadata: dict[str, Any]
    ) -> None:
        self.projects[project_name] = {
            "reference_dataset_id": reference_dataset_id,
            "metadata": metadata,
        }

    def create_run(
        self,
        name: str,
        inputs: dict[str, Any],
        run_type: Literal["chain"],
        *,
        project_name: str,
        **kwargs: Any,
    ) -> None:
        self.runs.append(
            {"name": name, "inputs": inputs, "run_type": run_type, "project_name": project_name}
            | kwargs
        )

    def create_feedback(self, run_id: uuid.UUID, key: str, *, score: float) -> None:
        assert any(r["id"] == run_id for r in self.runs), "feedback before its run"
        self.feedback.append((run_id, key, score))

    def flush(self) -> None:
        self.flushed += 1


def test_cases_come_from_stages_with_per_case_results() -> None:
    cases = collect_cases(_results())
    assert [(c.stage, c.case_id) for c in cases] == [
        ("retrieval", "CQ-001"),
        ("retrieval", "CQ-002"),
        ("agent", "AQ-001"),
        ("agent", "AQ-002"),
    ]
    assert cases[0].scores == {
        "retrieval.hit_at_1": False,
        "retrieval.hit_at_5": True,
        "retrieval.reciprocal_rank": 0.5,
    }
    assert cases[1].scores["retrieval.reciprocal_rank"] == 0.0


def test_example_ids_are_stable_across_runs() -> None:
    first = [c.example_id for c in collect_cases(_results())]
    second = [c.example_id for c in collect_cases(_results())]
    assert first == second
    assert len(set(first)) == len(first)


def test_publish_creates_dataset_experiment_runs_and_feedback() -> None:
    client = FakeLangSmith()

    name = publish(_results(), "dev", client, info={"llm_backend": "replay"}, now=NOW)

    assert name == "claimcheck-dev-20261113-093000"
    dataset = client.datasets["claimcheck-dev"]
    project = client.projects[name]
    assert project["reference_dataset_id"] == dataset.id
    assert project["metadata"]["llm_backend"] == "replay"
    assert project["metadata"]["metrics"]["validation.accuracy"] == 1.0

    assert len(client.examples) == 4
    agent_example = next(e for e in client.examples if e["metadata"]["case_id"] == "AQ-001")
    assert agent_example["outputs"] == {"clause_ref": "4.1"}

    assert {r["reference_example_id"] for r in client.runs} == {e["id"] for e in client.examples}
    assert all(r["project_name"] == name for r in client.runs)

    agent_run = next(r for r in client.runs if r["inputs"].get("question") and r["id"])
    keys = {key: score for run_id, key, score in client.feedback if run_id == agent_run["id"]}
    assert keys == {"agent.passed": 1.0, "agent.faithful": 1.0, "agent.citation_hit": 1.0}

    unknown_hit = [k for _, k, _ in client.feedback if k == "agent.citation_hit"]
    assert len(unknown_hit) == 1
    assert client.flushed == 2


def test_a_rerun_reuses_the_examples_it_already_has() -> None:
    known = [c.example_id for c in collect_cases(_results())[:2]]
    client = FakeLangSmith(existing_examples=known)
    client.create_dataset("claimcheck-dev", description="")

    publish(_results(), "dev", client, now=NOW)

    assert [e["metadata"]["case_id"] for e in client.examples] == ["AQ-001", "AQ-002"]
    assert len(client.runs) == 4


def test_run_refuses_langsmith_without_an_api_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    get_settings.cache_clear()
    monkeypatch.setitem(
        run.STAGES,
        "classification",
        lambda split, ctx: StageResult(metrics={"classification.accuracy": 1.0}),
    )

    argv = ["--split", "dev", "--stages", "classification", "--out-dir", str(tmp_path)]
    assert run.main([*argv, "--langsmith"]) == 2
    assert "LANGSMITH_API_KEY" in capsys.readouterr().err
    get_settings.cache_clear()


def test_run_publishes_when_asked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2_test")
    get_settings.cache_clear()
    published: list[tuple[str, dict[str, Any]]] = []

    def fake_publish(results: dict[str, StageResult], split: str, **kwargs: Any) -> str:
        published.append((split, kwargs))
        return "claimcheck-dev-x"

    monkeypatch.setattr(run, "_publish", fake_publish)
    monkeypatch.setattr(run, "setup_tracing", lambda *args: None)
    monkeypatch.setitem(
        run.STAGES,
        "classification",
        lambda split, ctx: StageResult(metrics={"classification.accuracy": 1.0}),
    )

    argv = ["--split", "dev", "--stages", "classification", "--out-dir", str(tmp_path)]
    assert run.main([*argv, "--langsmith"]) == 0
    assert published[0][0] == "dev"
    assert published[0][1]["api_key"] == "lsv2_test"
    assert isinstance(published[0][1]["info"], dict)
    get_settings.cache_clear()
