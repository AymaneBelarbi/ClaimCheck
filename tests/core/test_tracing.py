import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pydantic import BaseModel

from core import tracing
from core.tracing import (
    build_provider,
    otlp_traces_url,
    setup_tracing,
    span_cache_key,
    tracer,
)
from core.tracing import span as start_span
from gateway.base import ChatTurn, GatewayError, Message, SchemaError, Usage
from gateway.redaction import RedactingGateway, scope_gateway
from gateway.replay import ReplayGateway
from gateway.tracing import TracingGateway, UsageLedger
from tests.fakes import FakeGateway

SECRET_PROMPT = "Claimant: Jane Doe, licence A12345678, rear bumper"
SECRET_SYSTEM = "You read insurance forms for Jane Doe"


class Answer(BaseModel):
    text: str


def usage(**changes: object) -> Usage:
    base = {
        "model": "m-1",
        "input_tokens": 120,
        "output_tokens": 30,
        "cost_usd": 0.002,
        "latency_ms": 250.0,
    }
    return Usage.model_validate({**base, **changes})


class PricedGateway(FakeGateway):
    """A fake whose calls report real-looking usage."""

    def __init__(self, *, call_usage: Usage, **kwargs: object) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self._usage = call_usage

    def structured(self, **kwargs):  # type: ignore[no-untyped-def]
        result = super().structured(**kwargs)
        return result.model_copy(update={"usage": self._usage})

    def chat(self, **kwargs):  # type: ignore[no-untyped-def]
        turn = super().chat(**kwargs)
        return turn.model_copy(update={"usage": self._usage})


def ask(gateway: TracingGateway | RedactingGateway, key: str | None = "extract:p:1.pdf") -> Answer:
    return gateway.structured(
        system=SECRET_SYSTEM,
        prompt=SECRET_PROMPT,
        schema=Answer,
        purpose="extract.invoice",
        cache_key=key,
    ).output


def llm_spans(exporter: InMemorySpanExporter) -> list[ReadableSpan]:
    return [s for s in exporter.get_finished_spans() if s.name == "llm.call"]


def test_traces_url_is_built_from_the_endpoint() -> None:
    assert otlp_traces_url("http://jaeger:4318") == "http://jaeger:4318/v1/traces"
    assert otlp_traces_url("http://jaeger:4318/") == "http://jaeger:4318/v1/traces"
    assert otlp_traces_url("http://jaeger:4318/v1/traces") == "http://jaeger:4318/v1/traces"


def test_provider_carries_the_service_name() -> None:
    provider = build_provider("claimcheck-worker", "http://localhost:4318")
    assert provider.resource.attributes["service.name"] == "claimcheck-worker"
    provider.shutdown()


def test_setup_without_an_endpoint_changes_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    installed: list[TracerProvider] = []
    monkeypatch.setattr(tracing, "_configured", False)
    monkeypatch.setattr(trace, "set_tracer_provider", installed.append)

    setup_tracing("claimcheck-api", None)

    assert installed == []


def test_setup_installs_one_provider_however_often_it_is_called(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    installed: list[TracerProvider] = []
    monkeypatch.setattr(tracing, "_configured", False)
    monkeypatch.setattr(trace, "set_tracer_provider", installed.append)

    setup_tracing("claimcheck-worker", "http://localhost:4318")
    setup_tracing("claimcheck-worker", "http://localhost:4318")

    assert len(installed) == 1
    installed[0].shutdown()


def test_llm_call_span_carries_model_tokens_cost_and_latency(
    spans: InMemorySpanExporter,
) -> None:
    gateway = TracingGateway(
        PricedGateway(call_usage=usage(), structured_outputs=[Answer(text="ok")], model="m-1"),
        system="anthropic",
    )

    ask(gateway)

    (span,) = llm_spans(spans)
    assert dict(span.attributes or {}) == {
        "gen_ai.system": "anthropic",
        "gen_ai.request.model": "m-1",
        "gen_ai.usage.input_tokens": 120,
        "gen_ai.usage.output_tokens": 30,
        "claimcheck.cost_usd": 0.002,
        "claimcheck.latency_ms": 250.0,
        "claimcheck.purpose": "extract.invoice",
        "claimcheck.cache_key": span_cache_key("extract:p:1.pdf"),
    }
    assert span.status.is_ok or span.status.status_code.name == "UNSET"


def test_cache_key_is_left_off_when_not_given(spans: InMemorySpanExporter) -> None:
    gateway = TracingGateway(
        PricedGateway(call_usage=usage(), structured_outputs=[Answer(text="x")])
    )

    ask(gateway, key=None)

    (span,) = llm_spans(spans)
    assert "claimcheck.cache_key" not in (span.attributes or {})


def test_chat_calls_are_spans_too(spans: InMemorySpanExporter) -> None:
    turn = ChatTurn(text="hi", tool_calls=[], usage=usage())
    gateway = TracingGateway(PricedGateway(call_usage=usage(), chat_turns=[turn]), system="openai")

    gateway.chat(
        system=SECRET_SYSTEM,
        messages=[Message(role="user", content=SECRET_PROMPT)],
        tools=[],
        cache_key="agent:c:turn1",
    )

    (span,) = llm_spans(spans)
    attributes = span.attributes or {}
    assert attributes["claimcheck.purpose"] == "agent.chat"
    assert attributes["gen_ai.usage.output_tokens"] == 30
    assert attributes["claimcheck.cache_key"] == "agent:c:turn1"


def test_a_failed_call_records_the_error_and_reraises(spans: InMemorySpanExporter) -> None:
    failure = SchemaError("output does not match Answer: 1 validation error(s)", raw=SECRET_PROMPT)
    gateway = TracingGateway(FakeGateway(structured_outputs=[failure]))

    with pytest.raises(SchemaError):
        ask(gateway)

    (span,) = llm_spans(spans)
    assert span.status.status_code.name == "ERROR"
    assert (span.attributes or {})["error.type"] == "SchemaError"
    assert span.status.description == "SchemaError"
    # The exception is not recorded: its text and chain can quote the model's output.
    assert span.events == ()
    assert "gen_ai.usage.input_tokens" not in (span.attributes or {})


def test_no_prompt_or_output_text_is_on_any_span(spans: InMemorySpanExporter) -> None:
    ok = TracingGateway(PricedGateway(call_usage=usage(), structured_outputs=[Answer(text="Jane")]))
    ask(ok)
    bad = TracingGateway(
        FakeGateway(structured_outputs=[SchemaError("bad output", raw=SECRET_PROMPT)])
    )
    with pytest.raises(SchemaError):
        ask(bad)

    for span in spans.get_finished_spans():
        recorded = repr(dict(span.attributes or {})) + repr(
            [(event.name, dict(event.attributes or {})) for event in span.events]
        )
        for secret in ("Jane", "A12345678", "rear bumper", SECRET_SYSTEM):
            assert secret not in recorded


def test_llm_calls_are_children_of_the_open_span(spans: InMemorySpanExporter) -> None:
    gateway = TracingGateway(
        PricedGateway(call_usage=usage(), structured_outputs=[Answer(text="x")])
    )

    with tracer.start_as_current_span("stage.extract"):
        ask(gateway)

    parent = next(s for s in spans.get_finished_spans() if s.name == "stage.extract")
    (call,) = llm_spans(spans)
    assert call.parent is not None and call.parent.span_id == parent.context.span_id
    assert call.context.trace_id == parent.context.trace_id


def test_a_scoped_tracing_gateway_still_traces_and_still_redacts(
    spans: InMemorySpanExporter,
) -> None:
    inner = FakeGateway(structured_outputs=[Answer(text="[PERSON_1]")], is_cloud=True)
    gateway = TracingGateway(RedactingGateway(inner), system="anthropic")

    scoped = scope_gateway(gateway, ["Claimant name: Jane Doe\nPolicy: AUT-1"])

    assert isinstance(scoped, TracingGateway)
    assert scoped is not gateway
    result = scoped.structured(
        system="s", prompt="the driver Jane wrote", schema=Answer, purpose="extract.x"
    )
    assert "Jane" not in (inner.calls[0].prompt or "")
    assert result.output.text == "Jane"
    assert len(llm_spans(spans)) == 1


def test_scoping_a_local_traced_gateway_is_harmless(spans: InMemorySpanExporter) -> None:
    gateway = TracingGateway(FakeGateway(structured_outputs=[Answer(text="x")]))

    scoped = scope_gateway(gateway, ["Claimant name: Jane Doe"])

    ask(scoped)  # type: ignore[arg-type]
    assert len(llm_spans(spans)) == 1


def test_the_wrapper_answers_for_the_gateway_inside(tmp_path: object) -> None:
    replay = ReplayGateway(None, tmp_path, record=False)  # type: ignore[arg-type]
    wrapped = TracingGateway(RedactingGateway(replay))

    assert wrapped.drift_count == 0
    assert wrapped.drifted_keys == []
    assert isinstance(wrapped.inner, RedactingGateway)
    assert wrapped.model == replay.model
    assert wrapped.is_cloud == replay.is_cloud


def test_ledger_collects_usage_per_packet_only_inside_a_packet() -> None:
    ledger = UsageLedger()
    gateway = TracingGateway(
        PricedGateway(call_usage=usage(cost_usd=0.002), structured_outputs=[Answer(text="x")])
    )

    ask(gateway)  # outside any packet: not counted
    with ledger.packet("p1"):
        ask(gateway)
        ask(gateway)
    with ledger.packet("p2"):
        pass

    assert [u.cost_usd for u in ledger.packets["p1"].usages] == [0.002, 0.002]
    assert ledger.packets["p2"].usages == []
    assert ledger.packets["p1"].wall_ms is not None and ledger.packets["p1"].wall_ms >= 0


def test_ledger_ignores_failed_calls() -> None:
    ledger = UsageLedger()
    gateway = TracingGateway(FakeGateway(structured_outputs=[SchemaError("bad", raw="r")]))

    with ledger.packet("p1"), pytest.raises(SchemaError):
        ask(gateway)

    assert ledger.packets["p1"].usages == []


def everything_on(exporter: InMemorySpanExporter) -> str:
    """All text a finished span carries: attributes, events and the status description."""
    parts: list[str] = []
    for finished in exporter.get_finished_spans():
        parts.append(repr(dict(finished.attributes or {})))
        parts.append(repr(finished.status.description))
        parts.extend(repr((e.name, dict(e.attributes or {}))) for e in finished.events)
    return "\n".join(parts)


SECRET = "Jane Doe A12345678"


def test_a_chained_validation_error_does_not_leak_the_model_output(
    spans: InMemorySpanExporter,
) -> None:
    try:
        Answer.model_validate({"text": [SECRET]})
    except ValueError as cause:
        failure = SchemaError("output does not match Answer", raw=SECRET)
        failure.__cause__ = cause
    assert SECRET in str(failure.__cause__)  # the chain really does quote it
    gateway = TracingGateway(FakeGateway(structured_outputs=[failure]))

    with pytest.raises(SchemaError):
        ask(gateway)

    assert SECRET not in everything_on(spans)
    assert "Jane" not in everything_on(spans)


def test_a_gateway_error_keeps_its_status_code_and_not_its_message(
    spans: InMemorySpanExporter,
) -> None:
    failure = GatewayError(
        f"openai returned HTTP 400: {SECRET}", provider="openai", status_code=400
    )
    gateway = TracingGateway(FakeGateway(structured_outputs=[failure]))

    with pytest.raises(GatewayError):
        ask(gateway)

    (call,) = llm_spans(spans)
    assert (call.attributes or {})["http.response.status_code"] == 400
    assert (call.attributes or {})["error.type"] == "GatewayError"
    assert SECRET not in everything_on(spans)


def test_a_failing_stage_span_records_only_the_exception_type(
    spans: InMemorySpanExporter,
) -> None:
    with pytest.raises(RuntimeError), start_span("claim.process"), start_span("stage.ocr"):
        raise RuntimeError(f"[parameters: ('{SECRET}',)]")

    finished = {s.name: s for s in spans.get_finished_spans()}
    assert set(finished) == {"claim.process", "stage.ocr"}
    for one in finished.values():
        assert (one.attributes or {})["error.type"] == "RuntimeError"
        assert one.status.status_code.name == "ERROR"
        assert one.status.description == "RuntimeError"
        assert one.events == ()
    assert SECRET not in everything_on(spans)


def test_a_filename_in_a_cache_key_is_hashed_on_the_span(spans: InMemorySpanExporter) -> None:
    gateway = TracingGateway(
        PricedGateway(call_usage=usage(), structured_outputs=[Answer(text="x")])
    )

    ask(gateway, key=f"extract:PKT-001:{SECRET}.pdf")

    (call,) = llm_spans(spans)
    recorded = str((call.attributes or {})["claimcheck.cache_key"])
    assert recorded.startswith("extract:PKT-001:") and len(recorded) == len("extract:PKT-001:") + 8
    assert "Jane" not in everything_on(spans)


def test_span_cache_key() -> None:
    assert span_cache_key("extract:PKT-1:a.pdf") == span_cache_key("extract:PKT-1:a.pdf")
    assert span_cache_key("extract:PKT-1:a.pdf") != span_cache_key("extract:PKT-1:b.pdf")
    assert span_cache_key("classify:PKT-1:a:b.pdf").startswith("classify:PKT-1:")
    assert "a.pdf" not in span_cache_key("classify:PKT-1:a.pdf")
    # Keys without a filename are kept as they are.
    assert span_cache_key("judge:AQ-001") == "judge:AQ-001"
    assert span_cache_key("agent:3f2a:abc123:turn1") == "agent:3f2a:abc123:turn1"
    assert span_cache_key("structured:deadbeef") == "structured:deadbeef"
