import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from gateway.base import (
    CassetteMissError,
    ChatTurn,
    GatewayError,
    LLMResult,
    Message,
    SchemaError,
    ToolCall,
    ToolSpec,
    Usage,
)
from gateway.replay import ReplayGateway, cassette_filename


class Invoice(BaseModel):
    vendor: str
    total: float


class Receipt(BaseModel):
    shop: str


USAGE = Usage(model="fake-1", input_tokens=10, output_tokens=5, cost_usd=0.001, latency_ms=12.0)
TOOL = ToolSpec(name="t", description="d", input_schema={"type": "object"})


class FakeInner:
    """Counts calls so tests can prove a replay never reached the backend."""

    is_cloud = True
    model = "fake-1"

    def __init__(self) -> None:
        self.calls = 0

    def structured[T: BaseModel](
        self,
        *,
        system: str,
        prompt: str,
        schema: type[T],
        purpose: str,
        cache_key: str | None = None,
    ) -> LLMResult[T]:
        self.calls += 1
        output = schema.model_validate({"vendor": "Garage Lu", "total": 9.5, "shop": "Kiosk"})
        return LLMResult[T](output=output, usage=USAGE)

    def chat(
        self,
        *,
        system: str,
        messages: list[Message],
        tools: list[ToolSpec],
        cache_key: str | None = None,
    ) -> ChatTurn:
        self.calls += 1
        return ChatTurn(
            text="ok",
            tool_calls=[ToolCall(id="c1", name="t", arguments={"a": 1})],
            usage=USAGE,
        )


def ask(gw: ReplayGateway, *, prompt: str = "P", key: str | None = "extract:pkt-1:inv.pdf") -> Any:
    return gw.structured(
        system="S", prompt=prompt, schema=Invoice, purpose="extract", cache_key=key
    )


def test_cassette_filename_sanitises_key() -> None:
    readable = cassette_filename("extract:pkt-001:invoice.pdf")
    assert readable.startswith("extract__pkt-001__invoice.pdf-")
    assert readable.endswith(".json")
    assert "/" not in cassette_filename("a/b\\c d")
    assert not cassette_filename("../../etc/passwd").startswith(".")
    assert len(cassette_filename("k" * 500)) < 120


@pytest.mark.parametrize(
    ("first", "second"),
    [("a:b", "a__b"), ("a/b", "a_b"), (".x", "x"), ("p q", "p_q")],
)
def test_keys_that_sanitise_alike_get_distinct_files(first: str, second: str) -> None:
    assert cassette_filename(first) != cassette_filename(second)


def test_colliding_keys_keep_separate_cassettes(tmp_path: Path) -> None:
    class Named(BaseModel):
        vendor: str
        total: float

    class Namer(FakeInner):
        def structured[T: BaseModel](
            self,
            *,
            system: str,
            prompt: str,
            schema: type[T],
            purpose: str,
            cache_key: str | None = None,
        ) -> LLMResult[T]:
            self.calls += 1
            output = schema.model_validate({"vendor": cache_key, "total": 1.0})
            return LLMResult[T](output=output, usage=USAGE)

    inner = Namer()
    recorder = ReplayGateway(inner, tmp_path, record=True)
    keys = ["a:b", "a__b", "a/b", "a_b", ".x", "x"]
    for key in keys:
        ask(recorder, key=key)
    assert inner.calls == len(keys)
    assert len(list(tmp_path.glob("*.json"))) == len(keys)

    replayer = ReplayGateway(None, tmp_path, record=False)
    assert [ask(replayer, key=key).output.vendor for key in keys] == keys
    assert inner.calls == len(keys)


def test_cassette_holding_another_key_is_not_served_or_overwritten(tmp_path: Path) -> None:
    ask(ReplayGateway(FakeInner(), tmp_path, record=True), key="a:b")
    path = tmp_path / cassette_filename("a:b")
    # Simulate a hand-edited or renamed file: same name, different key inside.
    stored = json.loads(path.read_text())
    stored["key"] = "other"
    path.write_text(json.dumps(stored))
    with pytest.raises(CassetteMissError):
        ask(ReplayGateway(None, tmp_path, record=False), key="a:b")
    with pytest.raises(GatewayError, match="belongs to key"):
        ask(ReplayGateway(FakeInner(), tmp_path, record=True), key="a:b")


def test_replay_records_then_replays(tmp_path: Path) -> None:
    inner = FakeInner()
    recorder = ReplayGateway(inner, tmp_path, record=True)
    first = ask(recorder)
    assert inner.calls == 1
    assert (tmp_path / cassette_filename("extract:pkt-1:inv.pdf")).exists()

    # A fresh gateway with no inner serves the cassette and never needs a backend.
    replayer = ReplayGateway(None, tmp_path, record=False)
    second = ask(replayer)
    assert second.output == first.output == Invoice(vendor="Garage Lu", total=9.5)
    assert second.usage == USAGE
    assert inner.calls == 1

    # Same gateway, same key: also served from disk while recording.
    ask(recorder)
    assert inner.calls == 1


def test_cassette_file_layout(tmp_path: Path) -> None:
    gw = ReplayGateway(FakeInner(), tmp_path, record=True)
    ask(gw)
    stored = json.loads((tmp_path / cassette_filename("extract:pkt-1:inv.pdf")).read_text())
    assert stored["key"] == "extract:pkt-1:inv.pdf"
    assert stored["kind"] == "structured"
    assert len(stored["request_sha256"]) == 64
    assert stored["response"] == {"vendor": "Garage Lu", "total": 9.5}
    assert stored["usage"]["input_tokens"] == 10
    assert stored["model"] == "fake-1"


def test_replay_miss_raises(tmp_path: Path) -> None:
    gw = ReplayGateway(None, tmp_path, record=False)
    with pytest.raises(CassetteMissError) as err:
        ask(gw, key="extract:pkt-9:missing.pdf")
    assert "extract:pkt-9:missing.pdf" in str(err.value)


def test_miss_without_record_does_not_call_inner(tmp_path: Path) -> None:
    inner = FakeInner()
    gw = ReplayGateway(inner, tmp_path, record=False)
    with pytest.raises(CassetteMissError):
        ask(gw)
    assert inner.calls == 0


def test_record_without_inner_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="inner"):
        ReplayGateway(None, tmp_path, record=True)


def test_key_defaults_to_request_hash(tmp_path: Path) -> None:
    inner = FakeInner()
    gw = ReplayGateway(inner, tmp_path, record=True)
    ask(gw, key=None)
    ask(gw, key=None)
    assert inner.calls == 1
    ask(gw, key=None, prompt="a different prompt")
    assert inner.calls == 2
    assert len(list(tmp_path.glob("*.json"))) == 2


def test_replay_validates_stored_response_against_schema(tmp_path: Path) -> None:
    ask(ReplayGateway(FakeInner(), tmp_path, record=True))
    replayer = ReplayGateway(None, tmp_path, record=False)
    # Same key, different schema: the stored response does not fit it.
    with pytest.raises(SchemaError):
        replayer.structured(
            system="S",
            prompt="P",
            schema=Receipt,
            purpose="extract",
            cache_key="extract:pkt-1:inv.pdf",
        )


def test_drift_is_counted_but_still_served(tmp_path: Path) -> None:
    ask(ReplayGateway(FakeInner(), tmp_path, record=True))
    replayer = ReplayGateway(None, tmp_path, record=False)
    assert replayer.drift_count == 0
    result = ask(replayer)  # identical request
    assert replayer.drift_count == 0
    assert result.output.vendor == "Garage Lu"
    changed = ask(replayer, prompt="P, but the prompt template changed")
    assert changed.output.vendor == "Garage Lu"
    assert replayer.drift_count == 1
    assert replayer.drifted_keys == ["extract:pkt-1:inv.pdf"]


def test_chat_round_trip(tmp_path: Path) -> None:
    inner = FakeInner()
    msgs = [Message(role="user", content="hi")]
    first = ReplayGateway(inner, tmp_path, record=True).chat(
        system="S", messages=msgs, tools=[TOOL], cache_key="agent:q1:turn0"
    )
    again = ReplayGateway(None, tmp_path, record=False).chat(
        system="S", messages=msgs, tools=[TOOL], cache_key="agent:q1:turn0"
    )
    assert again == first
    assert again.tool_calls[0].arguments == {"a": 1}
    assert inner.calls == 1


def test_kind_mismatch_is_a_miss(tmp_path: Path) -> None:
    ask(ReplayGateway(FakeInner(), tmp_path, record=True))
    with pytest.raises(CassetteMissError):
        ReplayGateway(None, tmp_path, record=False).chat(
            system="S",
            messages=[Message(role="user", content="hi")],
            tools=[],
            cache_key="extract:pkt-1:inv.pdf",
        )


def test_model_and_cloud_flag_come_from_the_cassette(tmp_path: Path) -> None:
    ask(ReplayGateway(FakeInner(), tmp_path, record=True))
    replayer = ReplayGateway(None, tmp_path, record=False)
    assert replayer.model == "fake-1"
    assert replayer.is_cloud is True


def test_model_delegates_to_inner_when_present(tmp_path: Path) -> None:
    gw = ReplayGateway(FakeInner(), tmp_path, record=True)
    assert gw.model == "fake-1"
    assert gw.is_cloud is True


class Failing(FakeInner):
    """Raises whatever it is given, so tests can record failures."""

    def __init__(self, error: Exception) -> None:
        super().__init__()
        self.error = error

    def structured[T: BaseModel](
        self,
        *,
        system: str,
        prompt: str,
        schema: type[T],
        purpose: str,
        cache_key: str | None = None,
    ) -> LLMResult[T]:
        self.calls += 1
        raise self.error

    def chat(
        self,
        *,
        system: str,
        messages: list[Message],
        tools: list[ToolSpec],
        cache_key: str | None = None,
    ) -> ChatTurn:
        self.calls += 1
        raise self.error


def test_structured_schema_error_is_recorded_and_replayed(tmp_path: Path) -> None:
    inner = Failing(SchemaError("output does not match Invoice", raw='{"vendor": 3}'))
    recorder = ReplayGateway(inner, tmp_path, record=True)
    with pytest.raises(SchemaError):
        ask(recorder)
    stored = json.loads((tmp_path / cassette_filename("extract:pkt-1:inv.pdf")).read_text())
    assert stored["kind"] == "structured"
    assert stored["error"] == {
        "type": "SchemaError",
        "message": "output does not match Invoice",
        "raw": '{"vendor": 3}',
    }
    assert "response" not in stored

    replayer = ReplayGateway(None, tmp_path, record=False)
    with pytest.raises(SchemaError) as err:
        ask(replayer)
    assert err.value.raw == '{"vendor": 3}'
    assert str(err.value) == "output does not match Invoice"
    # The recorder serves the recorded failure too instead of calling the backend again.
    with pytest.raises(SchemaError):
        ask(recorder)
    assert inner.calls == 1


def test_chat_schema_error_is_recorded_and_replayed(tmp_path: Path) -> None:
    inner = Failing(SchemaError("tool call 't' has unparsable arguments", raw="{oops"))
    msgs = [Message(role="user", content="hi")]
    with pytest.raises(SchemaError):
        ReplayGateway(inner, tmp_path, record=True).chat(
            system="S", messages=msgs, tools=[TOOL], cache_key="agent:q1:turn0"
        )
    with pytest.raises(SchemaError) as err:
        ReplayGateway(None, tmp_path, record=False).chat(
            system="S", messages=msgs, tools=[TOOL], cache_key="agent:q1:turn0"
        )
    assert err.value.raw == "{oops"


def test_transport_errors_are_recorded_and_replayed(tmp_path: Path) -> None:
    # A caller that retries under a new key after a timeout must replay the same sequence,
    # so the timeout itself is part of the recording.
    inner = Failing(GatewayError("timed out", provider="anthropic", status_code=504))
    recorder = ReplayGateway(inner, tmp_path, record=True)
    with pytest.raises(GatewayError):
        ask(recorder)
    stored = json.loads((tmp_path / cassette_filename("extract:pkt-1:inv.pdf")).read_text())
    assert stored["error"] == {
        "type": "GatewayError",
        "message": "timed out",
        "provider": "anthropic",
        "status_code": 504,
    }
    with pytest.raises(GatewayError):
        ask(recorder)
    assert inner.calls == 1

    with pytest.raises(GatewayError) as err:
        ask(ReplayGateway(None, tmp_path, record=False))
    assert str(err.value) == "timed out"
    assert err.value.provider == "anthropic"
    assert err.value.status_code == 504


def test_chat_transport_error_is_recorded_and_replayed(tmp_path: Path) -> None:
    inner = Failing(GatewayError("connection refused", provider="ollama"))
    msgs = [Message(role="user", content="hi")]
    with pytest.raises(GatewayError):
        ReplayGateway(inner, tmp_path, record=True).chat(
            system="S", messages=msgs, tools=[TOOL], cache_key="agent:q1:turn0"
        )
    with pytest.raises(GatewayError) as err:
        ReplayGateway(None, tmp_path, record=False).chat(
            system="S", messages=msgs, tools=[TOOL], cache_key="agent:q1:turn0"
        )
    assert err.value.provider == "ollama"
    assert err.value.status_code is None
