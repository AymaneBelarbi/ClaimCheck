"""Trace every model call, and collect what the calls cost for one claim packet.

`TracingGateway` is the outermost wrapper `get_gateway` applies. Each `structured` or `chat`
call is one `llm.call` span, a child of whatever span is open (the pipeline's `stage.extract`,
the API's `agent.ask`). The span holds the backend, the model, token counts, cost, latency, the
purpose and the cache key. It never holds the prompt, the system text or the output.

`UsageLedger` is for eval runs: inside `ledger.packet(key)`, the `Usage` of every successful call
made through any `TracingGateway` is added to that packet's entry, however deep in the call stack
(or in a worker thread) the call happens.
"""

import time
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field

from opentelemetry.trace import Span
from opentelemetry.util.types import AttributeValue
from pydantic import BaseModel

from claimcheck.core.tracing import span as start_span
from claimcheck.core.tracing import span_cache_key
from claimcheck.gateway.base import (
    ChatTurn,
    GatewayError,
    LLMGateway,
    LLMResult,
    Message,
    ToolSpec,
    Usage,
)
from claimcheck.gateway.redaction import RedactingGateway, scope_gateway

CHAT_PURPOSE = "agent.chat"


@dataclass
class PacketUsage:
    """What one packet used: each model call's usage, and the wall time around its processing
    (None when the packet was only a recording target and nothing timed it)."""

    usages: list[Usage] = field(default_factory=list)
    wall_ms: float | None = None


@dataclass(frozen=True)
class _Active:
    ledger: "UsageLedger"
    key: str


_active: ContextVar[_Active | None] = ContextVar("claimcheck_usage_ledger", default=None)


class UsageLedger:
    """Usage per packet. Entries are created on entering `packet()`, so a packet that made no
    model call still counts (as zero calls)."""

    def __init__(self) -> None:
        self.packets: dict[str, PacketUsage] = {}

    @contextmanager
    def packet(self, key: str, *, timed: bool = True) -> Iterator[PacketUsage]:
        """Record the usage of model calls made inside the block under `key`.

        Entering the same key again adds to the entry. With `timed`, the wall time of the
        block is added to the entry's `wall_ms`; pass `timed=False` for a stage that is only
        one slice of the packet's processing and whose duration is measured elsewhere.
        """
        entry = self.packets.setdefault(key, PacketUsage())
        token = _active.set(_Active(self, key))
        started = time.perf_counter()
        try:
            yield entry
        finally:
            _active.reset(token)
            if timed:
                elapsed = (time.perf_counter() - started) * 1000
                entry.wall_ms = (entry.wall_ms or 0.0) + elapsed

    def add(self, key: str, usage: Usage) -> None:
        self.packets.setdefault(key, PacketUsage()).usages.append(usage)


def _record(usage: Usage) -> None:
    active = _active.get()
    if active is not None:
        active.ledger.add(active.key, usage)


class TracingGateway:
    """Wraps a gateway so each call is an `llm.call` span.

    `system` is the provider name for `gen_ai.system` (the configured backend).
    """

    def __init__(self, inner: LLMGateway, system: str = "unknown") -> None:
        self._inner = inner
        self._system = system

    @property
    def inner(self) -> LLMGateway:
        return self._inner

    @property
    def is_cloud(self) -> bool:
        return self._inner.is_cloud

    @property
    def model(self) -> str:
        return self._inner.model

    @property
    def drift_count(self) -> int:
        """Replay drift of the gateway underneath, 0 when it is not a replay gateway."""
        return int(getattr(self._inner, "drift_count", 0))

    @property
    def drifted_keys(self) -> list[str]:
        return list(getattr(self._inner, "drifted_keys", []))

    def scoped(self, texts: Iterable[str]) -> "TracingGateway":
        """The same tracing around `scope_gateway(inner, texts)`."""
        return TracingGateway(scope_gateway(self._inner, texts), self._system)

    def structured[T: BaseModel](
        self,
        *,
        system: str,
        prompt: str,
        schema: type[T],
        purpose: str,
        cache_key: str | None = None,
    ) -> LLMResult[T]:
        with self._span(purpose, cache_key) as span:
            result = self._inner.structured(
                system=system,
                prompt=prompt,
                schema=schema,
                purpose=purpose,
                cache_key=cache_key,
            )
            _finish(span, result.usage)
        _record(result.usage)
        return result

    def chat(
        self,
        *,
        system: str,
        messages: list[Message],
        tools: list[ToolSpec],
        cache_key: str | None = None,
    ) -> ChatTurn:
        with self._span(CHAT_PURPOSE, cache_key) as span:
            turn = self._inner.chat(
                system=system, messages=messages, tools=tools, cache_key=cache_key
            )
            _finish(span, turn.usage)
        _record(turn.usage)
        return turn

    @contextmanager
    def _span(self, purpose: str, cache_key: str | None) -> Iterator[Span]:
        attributes: dict[str, AttributeValue] = {
            "gen_ai.system": self._system,
            "gen_ai.request.model": self._inner.model,
            "claimcheck.purpose": purpose,
        }
        if cache_key is not None:
            attributes["claimcheck.cache_key"] = span_cache_key(cache_key)
        with start_span("llm.call", **attributes) as current:
            try:
                yield current
            except GatewayError as exc:
                # The status code says what went wrong; the message may hold the provider's body.
                if exc.status_code is not None:
                    current.set_attribute("http.response.status_code", exc.status_code)
                raise


def _finish(span: Span, usage: Usage) -> None:
    span.set_attribute("gen_ai.usage.input_tokens", usage.input_tokens)
    span.set_attribute("gen_ai.usage.output_tokens", usage.output_tokens)
    span.set_attribute("claimcheck.cost_usd", usage.cost_usd)
    span.set_attribute("claimcheck.latency_ms", usage.latency_ms)


def unwrap(gateway: LLMGateway) -> LLMGateway:
    """The backend (or replay) gateway under any tracing and redaction wrappers."""
    while isinstance(gateway, TracingGateway | RedactingGateway):
        gateway = gateway.inner
    return gateway
