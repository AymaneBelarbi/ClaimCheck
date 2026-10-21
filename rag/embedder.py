"""Local text embeddings. Always local: document text never goes to a cloud embedding API."""

import os
from pathlib import Path
from typing import Any, Protocol

import numpy as np

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = 384
DEFAULT_CACHE_DIR = "var/models"


class Embedder(Protocol):
    """Maps texts to unit-length vectors of a fixed size."""

    dim: int
    name: str

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class FastEmbedEmbedder:
    """bge-small-en-v1.5 through fastembed (ONNX, CPU).

    The model is downloaded on first use into `cache_dir` (env FASTEMBED_CACHE, default
    `var/models`) and loaded lazily, so constructing the object costs nothing.
    """

    dim = EMBEDDING_DIM
    name = EMBEDDING_MODEL

    def __init__(self, cache_dir: str | Path | None = None) -> None:
        self._cache_dir = str(cache_dir or os.environ.get("FASTEMBED_CACHE") or DEFAULT_CACHE_DIR)
        self._model: Any = None

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if self._model is None:
            # Tesseract subprocesses fork after the tokenizer has run; keep it single-threaded.
            os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
            from fastembed import TextEmbedding

            self._model = TextEmbedding(model_name=EMBEDDING_MODEL, cache_dir=self._cache_dir)
        vectors = np.asarray(list(self._model.embed(texts)), dtype=np.float64)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        # fastembed already normalises bge output; doing it again keeps the contract explicit.
        unit = vectors / np.where(norms == 0, 1.0, norms)
        return [[float(x) for x in row] for row in unit]
