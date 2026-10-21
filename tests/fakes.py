"""Test doubles shared across packages."""

import hashlib
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
from pydantic import BaseModel

from gateway.base import ChatTurn, LLMResult, Message, ToolSpec, Usage


class HashingEmbedder:
    """Deterministic bag-of-words embedder: each word hashes to one of `dim` buckets.

    Unit-length output, no model download. Texts that share words get a high cosine, which
    is enough for tests that only need "similar" and "different".
    """

    name = "hashing"

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for text in texts:
            vec = np.zeros(self.dim)
            for word in re.findall(r"[a-z0-9]+", text.lower()):
                digest = hashlib.sha256(word.encode()).digest()
                vec[int.from_bytes(digest[:4], "big") % self.dim] += 1.0
            norm = float(np.linalg.norm(vec))
            out.append([float(x) for x in (vec / norm if norm else vec)])
        return out


@dataclass
class FakeCall:
    method: str  # "structured" or "chat"
    system: str
    prompt: str | None = None
    schema: type[BaseModel] | None = None
    purpose: str | None = None
    messages: list[Message] = field(default_factory=list)
    tools: list[ToolSpec] = field(default_factory=list)
    cache_key: str | None = None


_ZERO_USAGE = Usage(model="fake-model", input_tokens=0, output_tokens=0, cost_usd=0.0, latency_ms=0)


class FakeGateway:
    """Scripted LLMGateway.

    `structured_outputs` and `chat_turns` are consumed in order; each item is either the
    value to return or an exception to raise. The last item repeats once the list runs out,
    so a single-element list means "always this". `structured_handler` overrides the list
    when the answer depends on the request. Every call is appended to `calls`.
    """

    def __init__(
        self,
        structured_outputs: Sequence[BaseModel | Exception] = (),
        chat_turns: Sequence[ChatTurn | Exception] = (),
        *,
        structured_handler: Callable[[FakeCall], BaseModel | Exception] | None = None,
        model: str = "fake-model",
        is_cloud: bool = False,
    ) -> None:
        self._structured = list(structured_outputs)
        self._chat = list(chat_turns)
        self._handler = structured_handler
        self._model = model
        self._is_cloud = is_cloud
        self.calls: list[FakeCall] = []

    @property
    def is_cloud(self) -> bool:
        return self._is_cloud

    @property
    def model(self) -> str:
        return self._model

    def structured[T: BaseModel](
        self,
        *,
        system: str,
        prompt: str,
        schema: type[T],
        purpose: str,
        cache_key: str | None = None,
    ) -> LLMResult[T]:
        call = FakeCall(
            "structured",
            system,
            prompt=prompt,
            schema=schema,
            purpose=purpose,
            cache_key=cache_key,
        )
        self.calls.append(call)
        if self._handler is not None:
            item = self._handler(call)
        else:
            item = self._next(
                self._structured, len([c for c in self.calls if c.method == "structured"])
            )
        if isinstance(item, Exception):
            raise item
        return LLMResult[T](output=schema.model_validate(item.model_dump()), usage=_ZERO_USAGE)

    def chat(
        self,
        *,
        system: str,
        messages: list[Message],
        tools: list[ToolSpec],
        cache_key: str | None = None,
    ) -> ChatTurn:
        self.calls.append(
            FakeCall(
                "chat", system, messages=list(messages), tools=list(tools), cache_key=cache_key
            )
        )
        item = self._next(self._chat, len([c for c in self.calls if c.method == "chat"]))
        if isinstance(item, Exception):
            raise item
        return item

    @staticmethod
    def _next[V](items: list[V], call_no: int) -> V:
        if not items:
            raise AssertionError("FakeGateway has no scripted response for this call")
        return items[min(call_no - 1, len(items) - 1)]
