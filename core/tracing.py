"""OpenTelemetry setup shared by the API, the worker, the tool server and the evals.

Spans are only exported when an OTLP/HTTP endpoint is configured (OTEL_ENDPOINT, the bundled
Jaeger listens on :4318). Without one the tracer is OpenTelemetry's no-op, so instrumented code
costs a function call and nothing else. Instrumentation is written by hand: the spans of
interest (a claim, its stages, each model call) are ours, and no library knows them.

Spans never carry prompts, model output or document text, only identifiers, counts and
timings. A claim packet is personal data. That includes failures: exception messages and
tracebacks (SQL parameters, provider error bodies, validation errors that quote the model's
output) can hold that text, so a failing span gets the exception's type name and nothing else.
Every span is started through `span()` below for that reason.
"""

import hashlib
import logging
import threading
from collections.abc import Iterator
from contextlib import contextmanager

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import Span, Status, StatusCode
from opentelemetry.util.types import AttributeValue

logger = logging.getLogger(__name__)

# Resolved against whichever provider is installed when a span starts, so importing this
# module before setup_tracing() runs is fine.
tracer = trace.get_tracer("claimcheck")

_TRACES_PATH = "/v1/traces"
_lock = threading.Lock()
_configured = False


@contextmanager
def span(name: str, **attributes: AttributeValue) -> Iterator[Span]:
    """Start `name` as the current span. If the block raises, the span gets `error.type` and
    an ERROR status described by the type name only; the exception is re-raised untouched and
    is never recorded on the span."""
    with tracer.start_as_current_span(
        name,
        attributes=attributes,
        record_exception=False,
        set_status_on_exception=False,
    ) as current:
        try:
            yield current
        except Exception as exc:
            kind = type(exc).__name__
            current.set_attribute("error.type", kind)
            current.set_status(Status(StatusCode.ERROR, kind))
            raise


def span_cache_key(key: str) -> str:
    """A cassette cache key that is safe to put on a span.

    `extract:{packet}:{filename}` and `classify:{packet}:{filename}` embed the name of an
    uploaded file, which can be a person's name; the filename becomes a short hash. The
    keys the cassettes use are not changed.
    """
    kind, _, rest = key.partition(":")
    packet, separator, filename = rest.partition(":")
    if kind in ("extract", "classify") and separator:
        return f"{kind}:{packet}:{hashlib.sha256(filename.encode()).hexdigest()[:8]}"
    return key


def otlp_traces_url(endpoint: str) -> str:
    """The OTLP/HTTP traces URL for a base endpoint such as http://jaeger:4318."""
    base = endpoint.rstrip("/")
    return base if base.endswith(_TRACES_PATH) else base + _TRACES_PATH


def build_provider(service_name: str, endpoint: str) -> TracerProvider:
    provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
    exporter = OTLPSpanExporter(endpoint=otlp_traces_url(endpoint))
    provider.add_span_processor(BatchSpanProcessor(exporter))
    return provider


def setup_tracing(service_name: str, endpoint: str | None) -> None:
    """Export spans to `endpoint` as `service_name`; does nothing without an endpoint.

    Safe to call more than once: only the first call that has an endpoint installs a provider.
    """
    global _configured
    if not endpoint:
        return
    with _lock:
        if _configured:
            return
        trace.set_tracer_provider(build_provider(service_name, endpoint))
        _configured = True
    logger.info("tracing %s to %s", service_name, otlp_traces_url(endpoint))
