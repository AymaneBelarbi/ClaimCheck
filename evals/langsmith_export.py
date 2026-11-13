"""Publish an eval run to LangSmith as an experiment.

    python -m evals.run --split dev --stages retrieval,agent --langsmith

The cases of a split are the examples of one dataset, `claimcheck-<split>`. An example's id is
derived from its stage and case id, so every run reuses the same examples and LangSmith can
compare experiments case by case. Each case in the run becomes one run in the experiment, with the
stage's per-case checks as feedback; the stage metrics and the run info (backend, models, replay
drift) go on the experiment's metadata.

Only stages with per-case results are published case by case (retrieval, agent); every stage's
metrics are on the metadata. Eval packets are synthetic (evals/dataset/), which is why questions
and answers may leave the machine here when they never do through tracing.
"""

import uuid
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal, Protocol

from evals.report import StageResult

_CASE_NAMESPACE = uuid.UUID("6f2b8f4e-58a1-4c8e-9a51-4b7e1c3d2a90")
_DATASET_PREFIX = "claimcheck"

Score = float | bool | None


@dataclass(frozen=True)
class Case:
    stage: str
    case_id: str
    inputs: dict[str, Any]
    reference: dict[str, Any]
    outputs: dict[str, Any]
    scores: dict[str, Score] = field(default_factory=dict)

    @property
    def example_id(self) -> uuid.UUID:
        return uuid.uuid5(_CASE_NAMESPACE, f"{self.stage}:{self.case_id}")


class ExperimentClient(Protocol):
    """The part of `langsmith.Client` used here."""

    def has_dataset(self, *, dataset_name: str) -> bool: ...
    def read_dataset(self, *, dataset_name: str) -> Any: ...
    def create_dataset(self, dataset_name: str, *, description: str) -> Any: ...
    def list_examples(self, *, dataset_id: uuid.UUID) -> Iterable[Any]: ...
    def create_examples(self, *, dataset_id: uuid.UUID, examples: list[dict[str, Any]]) -> Any: ...
    def create_project(
        self, project_name: str, *, reference_dataset_id: uuid.UUID, metadata: dict[str, Any]
    ) -> Any: ...
    def create_run(
        self,
        name: str,
        inputs: dict[str, Any],
        run_type: Literal["chain"],
        *,
        project_name: str,
        **kwargs: Any,
    ) -> Any: ...
    def create_feedback(self, run_id: uuid.UUID, key: str, *, score: float) -> Any: ...
    def flush(self) -> None: ...


def _agent_cases(details: dict[str, Any]) -> list[Case]:
    return [
        Case(
            stage="agent",
            case_id=a["id"],
            inputs={"packet_id": a["packet_id"], "question": a["question"]},
            reference={"clause_ref": a["expected_clause_ref"]},
            outputs={
                "answer": a["answer"],
                "citations": a["citations"],
                "retrieved": a["retrieved"],
                "stop_reason": a["stop_reason"],
                "failure": a["failure"],
            },
            scores={
                "agent.passed": a["passed"],
                "agent.faithful": a["faithful"],
                "agent.citation_hit": a["citation_hit"],
            },
        )
        for a in details.get("answers", [])
    ]


def _retrieval_cases(details: dict[str, Any]) -> list[Case]:
    cases = []
    for q in details.get("per_question", []):
        rank = q["rank"]
        cases.append(
            Case(
                stage="retrieval",
                case_id=q["id"],
                inputs={"question_id": q["id"]},
                reference={},
                outputs={"rank": rank},
                scores={
                    "retrieval.hit_at_1": rank == 1,
                    "retrieval.hit_at_5": rank is not None and rank <= 5,
                    "retrieval.reciprocal_rank": 1 / rank if rank else 0.0,
                },
            )
        )
    return cases


CASE_BUILDERS: dict[str, Callable[[dict[str, Any]], list[Case]]] = {
    "agent": _agent_cases,
    "retrieval": _retrieval_cases,
}


def collect_cases(results: dict[str, StageResult]) -> list[Case]:
    return [
        case
        for stage, result in results.items()
        if stage in CASE_BUILDERS
        for case in CASE_BUILDERS[stage](result.details)
    ]


def _dataset(client: ExperimentClient, name: str, split: str) -> Any:
    if client.has_dataset(dataset_name=name):
        return client.read_dataset(dataset_name=name)
    return client.create_dataset(
        name, description=f"ClaimCheck eval cases, {split} split (synthetic packets)."
    )


def publish(
    results: dict[str, StageResult],
    split: str,
    client: ExperimentClient,
    *,
    info: dict[str, Any] | None = None,
    now: datetime | None = None,
) -> str:
    """Publish `results` as a new experiment and return its name."""
    moment = now or datetime.now(UTC)
    cases = collect_cases(results)
    dataset = _dataset(client, f"{_DATASET_PREFIX}-{split}", split)

    known = {example.id for example in client.list_examples(dataset_id=dataset.id)}
    missing = [case for case in cases if case.example_id not in known]
    if missing:
        client.create_examples(
            dataset_id=dataset.id,
            examples=[
                {
                    "id": case.example_id,
                    "inputs": case.inputs,
                    "outputs": case.reference,
                    "metadata": {"stage": case.stage, "case_id": case.case_id},
                }
                for case in missing
            ],
        )

    experiment = f"{_DATASET_PREFIX}-{split}-{moment:%Y%m%d-%H%M%S}"
    client.create_project(
        experiment,
        reference_dataset_id=dataset.id,
        metadata={
            "split": split,
            "stages": list(results),
            **(info or {}),
            "metrics": {name: v for r in results.values() for name, v in r.metrics.items()},
        },
    )

    run_ids: list[tuple[uuid.UUID, Case]] = []
    for case in cases:
        run_id = uuid.uuid4()
        client.create_run(
            case.stage,
            case.inputs,
            "chain",
            id=run_id,
            outputs=case.outputs,
            project_name=experiment,
            reference_example_id=case.example_id,
            start_time=moment,
            end_time=moment,
        )
        run_ids.append((run_id, case))
    client.flush()

    for run_id, case in run_ids:
        for key, score in case.scores.items():
            if score is not None:
                client.create_feedback(run_id, key, score=float(score))
    client.flush()
    return experiment
