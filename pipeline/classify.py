"""Document classification: embedding similarity first, one LLM call only when it is unsure."""

import logging
from dataclasses import dataclass
from typing import Any, Literal
from weakref import WeakKeyDictionary

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel

from core.types import DocType
from gateway.base import GatewayError, LLMGateway, SchemaError
from pipeline.prototypes import PROTOTYPES
from pipeline.types import OcrResult
from rag.embedder import Embedder

logger = logging.getLogger(__name__)

MAX_CHARS = 2000
MIN_SIMILARITY = 0.55
MIN_MARGIN = 0.05

_CENTROIDS: WeakKeyDictionary[Any, dict[DocType, NDArray[np.float64]]] = WeakKeyDictionary()

_SYSTEM = (
    "You classify one document from an auto insurance claim packet, given its OCR text. "
    "Answer with exactly one doc_type:\n"
    "- claim_form: the policyholder's claim form\n"
    "- id_card: a driver licence or identity card\n"
    "- invoice: a repair or glass-replacement invoice from a garage or workshop\n"
    "- incident_report: a police or incident report\n"
    "- other: anything else, or text too broken to tell\n"
    "confidence is your certainty from 0 to 1."
)


class DocTypeChoice(BaseModel):
    doc_type: Literal["claim_form", "id_card", "invoice", "incident_report", "other"]
    confidence: float


@dataclass(frozen=True)
class ClassificationResult:
    doc_type: DocType
    confidence: float
    # "llm" means the fallback was consulted, whether or not it produced an answer.
    method: Literal["embedding", "llm"]


def classify(
    ocr: OcrResult,
    embedder: Embedder,
    llm: LLMGateway | None,
    *,
    cache_key: str | None = None,
) -> ClassificationResult:
    if ocr.low_confidence:
        # Poor OCR is flagged for a person, not guessed at.
        return ClassificationResult(DocType.OTHER, ocr.mean_confidence, "embedding")

    text = ocr.text[:MAX_CHARS]
    centroids = _centroids(embedder)
    doc = _unit(np.asarray(embedder.embed([text])[0], dtype=np.float64))
    scores = sorted(
        ((float(doc @ centroid), doc_type) for doc_type, centroid in centroids.items()),
        key=lambda pair: pair[0],
        reverse=True,
    )
    (top, best), (second, _) = scores[0], scores[1]

    if top >= MIN_SIMILARITY and top - second >= MIN_MARGIN:
        return ClassificationResult(best, top, "embedding")
    if llm is None:
        return ClassificationResult(DocType.OTHER, top, "embedding")
    return llm_fallback(ocr, llm, cache_key=cache_key)


def llm_fallback(
    ocr: OcrResult, llm: LLMGateway, *, cache_key: str | None = None
) -> ClassificationResult:
    """The model's answer for a document, as `classify` asks for it when it is unsure.

    Public so cassette recording can ask it for documents the embeddings were sure about.
    """
    try:
        result = llm.structured(
            system=_SYSTEM,
            prompt=f"Document text:\n<<<\n{ocr.text[:MAX_CHARS]}\n>>>",
            schema=DocTypeChoice,
            purpose="classify",
            cache_key=cache_key,
        )
    except (GatewayError, SchemaError) as exc:
        logger.warning("classification fallback failed (%s), routing to other", exc)
        return ClassificationResult(DocType.OTHER, 0.0, "llm")
    choice = result.output
    return ClassificationResult(
        DocType(choice.doc_type), min(max(choice.confidence, 0.0), 1.0), "llm"
    )


def _centroids(embedder: Embedder) -> dict[DocType, NDArray[np.float64]]:
    cached = _CENTROIDS.get(embedder)
    if cached is not None:
        return cached
    owners: list[DocType] = []
    texts: list[str] = []
    for doc_type, exemplars in PROTOTYPES.items():
        owners.extend([doc_type] * len(exemplars))
        texts.extend(exemplars)
    vectors = np.asarray(embedder.embed(texts), dtype=np.float64)
    centroids = {
        doc_type: _unit(vectors[[i for i, owner in enumerate(owners) if owner == doc_type]].mean(0))
        for doc_type in PROTOTYPES
    }
    _CENTROIDS[embedder] = centroids
    return centroids


def _unit(vec: NDArray[np.float64]) -> NDArray[np.float64]:
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm else vec
