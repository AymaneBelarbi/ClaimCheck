"""One packet is one trace: a root span, the stages under it, and the model calls under those."""

import uuid
from collections import Counter
from pathlib import Path
from typing import Any

import pytest
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from db.session import SessionFactory
from evals.generate_dataset import generate
from gateway.base import LLMResult
from gateway.tracing import TracingGateway
from tests.fakes import FakeGateway, HashingEmbedder
from tests.worker.factories import (
    add_dataset_policy,
    add_packet,
    forced_classifier,
    gold_gateway,
    ground_truth,
)
from worker.pipeline_runner import PipelineDeps, process_claim
from worker.storage import Storage

pytestmark = [pytest.mark.db, pytest.mark.ocr]

PACKET = "PKT-001"  # four documents


class PricedGateway:
    """The gold extraction model, reporting tokens, cost and latency on every call."""

    def __init__(self, inner: FakeGateway) -> None:
        self._inner = inner

    @property
    def is_cloud(self) -> bool:
        return self._inner.is_cloud

    @property
    def model(self) -> str:
        return "gold-model"

    def structured(self, **kwargs: Any) -> LLMResult[Any]:
        result: LLMResult[Any] = self._inner.structured(**kwargs)
        usage = result.usage.model_copy(
            update={
                "model": "gold-model",
                "input_tokens": 800,
                "output_tokens": 90,
                "cost_usd": 0.0004,
                "latency_ms": 120.0,
            }
        )
        return result.model_copy(update={"usage": usage})

    def chat(self, **kwargs: Any) -> Any:
        return self._inner.chat(**kwargs)


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("dataset")
    generate(out=root / "data", seed=20261020, questions_path=root / "questions.jsonl")
    return root / "data"


def _by_name(spans: list[ReadableSpan]) -> dict[str, list[ReadableSpan]]:
    grouped: dict[str, list[ReadableSpan]] = {}
    for span in spans:
        grouped.setdefault(span.name, []).append(span)
    return grouped


def test_a_packet_is_one_trace_with_a_span_per_stage_and_per_model_call(
    sessions: SessionFactory,
    worker_sessions: SessionFactory,
    data_dir: Path,
    tmp_path: Path,
    spans: InMemorySpanExporter,
) -> None:
    gateway = TracingGateway(PricedGateway(gold_gateway(data_dir)), system="replay")
    deps = PipelineDeps(
        storage=Storage(tmp_path / "storage"),
        embedder=HashingEmbedder(),
        llm=gateway,
        classifier=forced_classifier(data_dir),
    )
    truth = ground_truth(data_dir, PACKET)
    with sessions() as session:
        policy = add_dataset_policy(session, data_dir, truth.policy_number, HashingEmbedder())
        claim_id: uuid.UUID = add_packet(session, data_dir, PACKET, policy, deps.storage).id
    documents = len(truth.documents)
    spans.clear()

    process_claim(worker_sessions, claim_id, deps)

    finished = list(spans.get_finished_spans())
    assert len({s.context.trace_id for s in finished}) == 1
    grouped = _by_name(finished)
    assert Counter({name: len(found) for name, found in grouped.items()}) == Counter(
        {
            "claim.process": 1,
            "stage.ocr": documents,
            "stage.classify": documents,
            "stage.extract": documents,
            "stage.validate": 1,
            "llm.call": documents,
        }
    )

    (root,) = grouped["claim.process"]
    assert root.parent is None
    assert dict(root.attributes or {}) == {"claim.id": str(claim_id), "packet.id": PACKET}

    for name in ("stage.ocr", "stage.classify", "stage.extract", "stage.validate"):
        for span in grouped[name]:
            assert span.parent is not None and span.parent.span_id == root.context.span_id
    for span in grouped["stage.extract"]:
        attributes = span.attributes or {}
        assert attributes["document.id"]
        assert attributes["doc_type"] in {"claim_form", "id_card", "invoice", "incident_report"}

    extract_ids = {s.context.span_id for s in grouped["stage.extract"]}
    for call in grouped["llm.call"]:
        assert call.parent is not None and call.parent.span_id in extract_ids
        attributes = call.attributes or {}
        assert attributes["gen_ai.request.model"] == "gold-model"
        assert attributes["gen_ai.usage.input_tokens"] == 800
        assert attributes["gen_ai.usage.output_tokens"] == 90
        assert attributes["claimcheck.cost_usd"] == 0.0004
        assert attributes["claimcheck.latency_ms"] == 120.0
        assert str(attributes["claimcheck.purpose"]).startswith("extract")
        assert str(attributes["claimcheck.cache_key"]).startswith(f"extract:{PACKET}:")
        assert ".pdf" not in str(attributes["claimcheck.cache_key"])
