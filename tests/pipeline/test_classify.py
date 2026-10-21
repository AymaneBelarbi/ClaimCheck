import numpy as np
import pytest

from core.types import DocType
from gateway.base import GatewayError, SchemaError
from pipeline.classify import ClassificationResult, DocTypeChoice, classify
from pipeline.prototypes import PROTOTYPES
from pipeline.types import OcrPage, OcrResult
from tests.fakes import FakeGateway, HashingEmbedder

_TYPES: list[DocType] = [t for t in DocType if t is not DocType.OTHER]
# one spare axis that no prototype uses, to build vectors that are far from every type
_DIM = len(_TYPES) + 1
_SPARE = len(_TYPES)


def _unit(spare: float = 0.0, **weights: float) -> list[float]:
    vec = np.zeros(_DIM)
    vec[_SPARE] = spare
    for name, w in weights.items():
        vec[_TYPES.index(DocType(name))] = w
    return [float(x) for x in vec / np.linalg.norm(vec)]


class AxisEmbedder:
    """Prototype texts map to their type's axis; other texts to a vector chosen by the test."""

    dim = _DIM
    name = "axis"

    def __init__(self, documents: dict[str, list[float]]) -> None:
        self._documents = documents
        self._prototypes = {text: t for t, texts in PROTOTYPES.items() for text in texts}
        self.calls = 0

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        return [
            _unit(**{self._prototypes[t].value: 1.0})
            if t in self._prototypes
            else self._documents[t]
            for t in texts
        ]


def _ocr(text: str, *, mean: float = 0.9, low: bool = False) -> OcrResult:
    return OcrResult(
        pages=[OcrPage(page_no=1, text=text)],
        words=[],
        text=text,
        mean_confidence=mean,
        low_confidence=low,
    )


def _llm(doc_type: str) -> FakeGateway:
    return FakeGateway([DocTypeChoice(doc_type=doc_type, confidence=0.8)])


def test_classifies_obvious_invoice() -> None:
    embedder = AxisEmbedder({"garage bill": _unit(invoice=1.0, claim_form=0.2)})
    llm = FakeGateway([GatewayError("must not be called", provider="x")])

    result = classify(_ocr("garage bill"), embedder, llm)

    assert result.doc_type is DocType.INVOICE
    assert result.method == "embedding"
    assert result.confidence > 0.9
    assert llm.calls == []


def test_ambiguous_uses_llm_fallback() -> None:
    embedder = AxisEmbedder({"unclear": _unit(claim_form=1.0, incident_report=1.0)})
    llm = _llm("incident_report")

    result = classify(_ocr("unclear"), embedder, llm, cache_key="classify:PKT-1:doc_1.pdf")

    assert result == ClassificationResult(DocType.INCIDENT_REPORT, 0.8, "llm")
    (call,) = llm.calls
    assert call.purpose == "classify"
    assert call.cache_key == "classify:PKT-1:doc_1.pdf"
    assert call.schema is DocTypeChoice
    assert call.prompt is not None and "unclear" in call.prompt


def test_low_top_similarity_uses_llm_fallback() -> None:
    # Clear winner (margin 0.5), but its cosine of 0.5 is under the 0.55 floor.
    embedder = AxisEmbedder({"vague": _unit(invoice=0.5, spare=0.866)})
    llm = _llm("other")

    result = classify(_ocr("vague"), embedder, llm)

    assert result.method == "llm"
    assert result.doc_type is DocType.OTHER


@pytest.mark.parametrize(
    "error", [GatewayError("down", provider="ollama"), SchemaError("bad", raw="{")]
)
def test_fallback_failure_returns_other(error: Exception) -> None:
    embedder = AxisEmbedder({"unclear": _unit(claim_form=1.0, incident_report=1.0)})

    result = classify(_ocr("unclear"), embedder, FakeGateway([error]))

    assert result.doc_type is DocType.OTHER
    assert result.confidence == 0.0


def test_ambiguous_without_llm_returns_other() -> None:
    embedder = AxisEmbedder({"unclear": _unit(claim_form=1.0, incident_report=1.0)})

    result = classify(_ocr("unclear"), embedder, None)

    assert result.doc_type is DocType.OTHER
    assert result.method == "embedding"


def test_low_confidence_ocr_skips_embedder_and_llm() -> None:
    embedder = AxisEmbedder({})
    llm = _llm("invoice")

    result = classify(_ocr("garbled", mean=0.41, low=True), embedder, llm)

    assert result == ClassificationResult(DocType.OTHER, 0.41, "embedding")
    assert embedder.calls == 0
    assert llm.calls == []


def test_only_first_two_thousand_characters_are_embedded() -> None:
    head = "x" * 2000
    embedder = AxisEmbedder({head: _unit(invoice=1.0)})

    classify(_ocr(head + "tail that is never seen"), embedder, None)


def test_centroids_are_computed_once_per_embedder() -> None:
    embedder = AxisEmbedder({"a": _unit(invoice=1.0), "b": _unit(invoice=1.0)})

    classify(_ocr("a"), embedder, None)
    classify(_ocr("b"), embedder, None)

    # one call for the prototypes, then one per document
    assert embedder.calls == 3


def test_prototypes_cover_every_type_but_other() -> None:
    assert set(PROTOTYPES) == set(_TYPES)
    assert all(len(texts) >= 3 for texts in PROTOTYPES.values())


def test_hashing_embedder_is_deterministic_and_normalised() -> None:
    e = HashingEmbedder()
    a, b = e.embed(["repair invoice total"]), e.embed(["repair invoice total"])
    assert a == b
    assert abs(float(np.linalg.norm(a[0])) - 1.0) < 1e-9


@pytest.mark.slow
def test_fastembed_prototypes_classify_a_plain_invoice() -> None:
    from rag.embedder import FastEmbedEmbedder

    text = (
        "Riverside Auto Body\nTax invoice 4471\nCustomer: J. Smith\nFront bumper replace 420.00\n"
        "Labour 6 hours 540.00\nTotal due 960.00"
    )

    result = classify(_ocr(text), FastEmbedEmbedder(), None)

    assert result.doc_type is DocType.INVOICE
