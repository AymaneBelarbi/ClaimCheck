"""Backend tests run against respx-mocked HTTP; nothing here needs a key or a network."""

import json
from typing import Any

import httpx
import pytest
import respx
from pydantic import BaseModel, Field

from gateway.anthropic import AnthropicGateway
from gateway.base import GatewayError, LLMGateway, Message, SchemaError, ToolCall, ToolSpec
from gateway.ollama import OllamaGateway
from gateway.openai import OpenAIGateway
from gateway.pricing import cost

ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
OPENAI_URL = "https://api.openai.com/v1/chat/completions"
OLLAMA_URL = "http://localhost:11434/api/chat"


class Invoice(BaseModel):
    vendor: str
    total: float
    note: str | None = None


LOOKUP = ToolSpec(
    name="lookup_clause",
    description="Find the policy clause covering a peril.",
    input_schema={
        "type": "object",
        "properties": {"peril": {"type": "string"}},
        "required": ["peril"],
    },
)

CHAT = [
    Message(role="user", content="Is glass covered?"),
    Message(
        role="assistant",
        content="",
        tool_calls=[ToolCall(id="call_1", name="lookup_clause", arguments={"peril": "glass"})],
    ),
    Message(role="tool", content="Clause 4.2: glass is covered.", tool_call_id="call_1"),
]


def last_request_body() -> dict[str, Any]:
    body: dict[str, Any] = json.loads(respx.calls.last.request.content)
    return body


# --- Anthropic ---------------------------------------------------------------


def anthropic_gateway() -> AnthropicGateway:
    return AnthropicGateway(api_key="sk-test", model="claude-haiku-5-5")


def anthropic_text(text: str, *, stop_reason: str = "end_turn") -> dict[str, Any]:
    return {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "claude-haiku-5-5",
        "content": [{"type": "text", "text": text}],
        "stop_reason": stop_reason,
        "usage": {"input_tokens": 120, "output_tokens": 30},
    }


@respx.mock
def test_anthropic_structured_parses_schema() -> None:
    respx.post(ANTHROPIC_URL).respond(
        json=anthropic_text(json.dumps({"vendor": "Garage Lu", "total": 412.5, "note": None}))
    )
    result = anthropic_gateway().structured(
        system="Extract.", prompt="Invoice text", schema=Invoice, purpose="extract"
    )
    assert result.output == Invoice(vendor="Garage Lu", total=412.5)
    assert result.usage.input_tokens == 120
    assert result.usage.output_tokens == 30
    assert result.usage.model == "claude-haiku-5-5"
    assert result.usage.cost_usd == pytest.approx(cost("claude-haiku-5-5", 120, 30))
    assert result.usage.cost_usd > 0
    assert result.usage.latency_ms >= 0


@respx.mock
def test_anthropic_structured_ignores_thinking_blocks() -> None:
    body = anthropic_text('{"vendor": "x", "total": 1, "note": null}')
    body["content"].insert(0, {"type": "thinking", "thinking": "", "signature": "sig"})
    respx.post(ANTHROPIC_URL).respond(json=body)
    result = anthropic_gateway().structured(
        system="S", prompt="P", schema=Invoice, purpose="extract"
    )
    assert result.output.vendor == "x"


@pytest.mark.parametrize(
    ("model", "thinking"),
    [
        ("claude-haiku-5-5", {"type": "disabled"}),
        ("claude-sonnet-5-5", {"type": "between_tools"}),
        ("claude-opus-5-5", None),
    ],
)
@respx.mock
def test_anthropic_structured_request_body_is_valid_per_model(
    model: str, thinking: dict[str, str] | None
) -> None:
    respx.post(ANTHROPIC_URL).respond(json=anthropic_text('{"vendor":"x","total":1,"note":null}'))
    AnthropicGateway(api_key="sk-test", model=model).structured(
        system="S", prompt="P", schema=Invoice, purpose="extract"
    )
    request = respx.calls.last.request
    body = last_request_body()
    assert request.headers["x-api-key"] == "sk-test"
    assert request.headers["anthropic-version"]
    assert body["model"] == model
    assert body["system"] == "S"
    assert body["messages"] == [{"role": "user", "content": "P"}]
    # Forced tool_choice is a 400 on Sonnet 5.5 / Opus 5.5 / Fable 5.1: use output_config.
    assert "tools" not in body
    assert "tool_choice" not in body
    fmt = body["output_config"]["format"]
    assert fmt["type"] == "json_schema"
    assert fmt["schema"]["additionalProperties"] is False
    assert sorted(fmt["schema"]["required"]) == ["note", "total", "vendor"]
    # Models that cannot take `disabled` get their own off-switch or no field at all.
    assert body.get("thinking") == thinking
    assert "temperature" not in body
    assert body["max_tokens"] >= 16_000


@respx.mock
def test_anthropic_structured_strips_keywords_the_format_rejects() -> None:
    class Bounded(BaseModel):
        count: int = Field(ge=0, le=10)
        tags: list[str] = Field(max_length=3)
        minimum: str

    respx.post(ANTHROPIC_URL).respond(json=anthropic_text('{"count":1,"tags":[],"minimum":"a"}'))
    anthropic_gateway().structured(system="S", prompt="P", schema=Bounded, purpose="extract")
    schema = last_request_body()["output_config"]["format"]["schema"]
    assert schema["properties"]["count"] == {"title": "Count", "type": "integer"}
    assert "maxItems" not in schema["properties"]["tags"]
    assert "minimum" in schema["properties"]  # a field name, not a keyword


@respx.mock
def test_anthropic_invalid_json_raises_schema_error() -> None:
    # total is not a number
    respx.post(ANTHROPIC_URL).respond(json=anthropic_text('{"vendor": "x", "total": "lots"}'))
    with pytest.raises(SchemaError) as err:
        anthropic_gateway().structured(system="S", prompt="P", schema=Invoice, purpose="extract")
    assert "lots" in err.value.raw


@respx.mock
def test_anthropic_max_tokens_raises_schema_error() -> None:
    respx.post(ANTHROPIC_URL).respond(
        json=anthropic_text('{"vendor": "Gara', stop_reason="max_tokens")
    )
    with pytest.raises(SchemaError, match="max_tokens") as err:
        anthropic_gateway().structured(system="S", prompt="P", schema=Invoice, purpose="extract")
    assert err.value.raw == '{"vendor": "Gara'


@respx.mock
def test_anthropic_refusal_raises_schema_error() -> None:
    body = anthropic_text("", stop_reason="refusal")
    body["content"] = []
    respx.post(ANTHROPIC_URL).respond(json=body)
    with pytest.raises(SchemaError, match="declined"):
        anthropic_gateway().structured(system="S", prompt="P", schema=Invoice, purpose="extract")


@pytest.mark.parametrize(
    ("model", "thinking"),
    [
        ("claude-haiku-5-5", {"type": "disabled"}),
        ("claude-sonnet-5-5", {"type": "between_tools"}),
    ],
)
@respx.mock
def test_anthropic_chat_request_body_is_valid_per_model(
    model: str, thinking: dict[str, str]
) -> None:
    respx.post(ANTHROPIC_URL).respond(json=anthropic_text("ok"))
    AnthropicGateway(api_key="sk-test", model=model).chat(
        system="S", messages=CHAT[:1], tools=[LOOKUP]
    )
    body = last_request_body()
    assert body["model"] == model
    assert body["thinking"] == thinking
    assert body["tool_choice"] == {"type": "auto"}
    assert body["max_tokens"] >= 16_000
    assert "temperature" not in body


@respx.mock
def test_anthropic_chat_returns_tool_calls() -> None:
    respx.post(ANTHROPIC_URL).respond(
        json={
            "id": "msg_2",
            "type": "message",
            "role": "assistant",
            "model": "claude-haiku-5-5",
            "content": [
                {"type": "text", "text": "Checking."},
                {
                    "type": "tool_use",
                    "id": "toolu_9",
                    "name": "lookup_clause",
                    "input": {"peril": "theft"},
                },
            ],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 80, "output_tokens": 20},
        }
    )
    turn = anthropic_gateway().chat(system="S", messages=CHAT[:1], tools=[LOOKUP])
    assert turn.text == "Checking."
    assert turn.tool_calls == [
        ToolCall(id="toolu_9", name="lookup_clause", arguments={"peril": "theft"})
    ]
    assert turn.usage.input_tokens == 80
    assert last_request_body()["tools"][0]["name"] == "lookup_clause"


@respx.mock
def test_anthropic_chat_sends_tool_results_as_user_blocks() -> None:
    respx.post(ANTHROPIC_URL).respond(
        json={
            "content": [{"type": "text", "text": "Yes, clause 4.2."}],
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }
    )
    two_results = [
        *CHAT[:2],
        Message(role="tool", content="one", tool_call_id="call_1"),
        Message(role="tool", content="two", tool_call_id="call_2"),
    ]
    turn = anthropic_gateway().chat(system="S", messages=two_results, tools=[LOOKUP])
    assert turn.text == "Yes, clause 4.2."
    assert turn.tool_calls == []
    messages = last_request_body()["messages"]
    assert messages[1]["role"] == "assistant"
    assert messages[1]["content"][0] == {
        "type": "tool_use",
        "id": "call_1",
        "name": "lookup_clause",
        "input": {"peril": "glass"},
    }
    # Both results go back in one user message.
    assert len(messages) == 3
    assert messages[2]["role"] == "user"
    assert [b["tool_use_id"] for b in messages[2]["content"]] == ["call_1", "call_2"]
    assert messages[2]["content"][0]["type"] == "tool_result"


# --- OpenAI ------------------------------------------------------------------


def openai_gateway() -> OpenAIGateway:
    return OpenAIGateway(api_key="sk-test", model="gpt-5-mini")


def openai_reply(message: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": "chatcmpl-1",
        "model": "gpt-5-mini",
        "choices": [{"index": 0, "message": {"role": "assistant", **message}}],
        "usage": {"prompt_tokens": 200, "completion_tokens": 50},
    }


@respx.mock
def test_openai_structured_parses_schema() -> None:
    respx.post(OPENAI_URL).respond(
        json=openai_reply({"content": json.dumps({"vendor": "Garage Lu", "total": 412.5})})
    )
    result = openai_gateway().structured(
        system="Extract.", prompt="Invoice text", schema=Invoice, purpose="extract"
    )
    assert result.output.vendor == "Garage Lu"
    assert result.output.note is None
    assert result.usage.input_tokens == 200
    assert result.usage.output_tokens == 50
    assert result.usage.cost_usd == pytest.approx(cost("gpt-5-mini", 200, 50))
    assert result.usage.cost_usd > 0


@respx.mock
def test_openai_structured_requests_strict_json_schema() -> None:
    respx.post(OPENAI_URL).respond(json=openai_reply({"content": '{"vendor":"x","total":1}'}))
    openai_gateway().structured(system="S", prompt="P", schema=Invoice, purpose="extract")
    request = respx.calls.last.request
    body = last_request_body()
    assert request.headers["authorization"] == "Bearer sk-test"
    fmt = body["response_format"]
    assert fmt["type"] == "json_schema"
    assert fmt["json_schema"]["strict"] is True
    schema = fmt["json_schema"]["schema"]
    # Strict mode needs every property listed as required and no extra keys.
    assert schema["additionalProperties"] is False
    assert sorted(schema["required"]) == ["note", "total", "vendor"]
    assert body["messages"][0] == {"role": "system", "content": "S"}
    assert body["messages"][1] == {"role": "user", "content": "P"}


@respx.mock
def test_openai_invalid_json_raises_schema_error() -> None:
    respx.post(OPENAI_URL).respond(json=openai_reply({"content": "{not json"}))
    with pytest.raises(SchemaError) as err:
        openai_gateway().structured(system="S", prompt="P", schema=Invoice, purpose="extract")
    assert err.value.raw == "{not json"


@respx.mock
def test_openai_refusal_raises_schema_error() -> None:
    respx.post(OPENAI_URL).respond(json=openai_reply({"content": None, "refusal": "No."}))
    with pytest.raises(SchemaError) as err:
        openai_gateway().structured(system="S", prompt="P", schema=Invoice, purpose="extract")
    assert err.value.raw == "No."


@respx.mock
def test_openai_chat_returns_tool_calls() -> None:
    respx.post(OPENAI_URL).respond(
        json=openai_reply(
            {
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_7",
                        "type": "function",
                        "function": {
                            "name": "lookup_clause",
                            "arguments": json.dumps({"peril": "theft"}),
                        },
                    }
                ],
            }
        )
    )
    turn = openai_gateway().chat(system="S", messages=CHAT, tools=[LOOKUP])
    assert turn.text is None
    assert turn.tool_calls == [
        ToolCall(id="call_7", name="lookup_clause", arguments={"peril": "theft"})
    ]
    body = last_request_body()
    assert body["tools"][0]["function"]["name"] == "lookup_clause"
    assistant = body["messages"][2]
    assert assistant["tool_calls"][0]["function"]["arguments"] == '{"peril": "glass"}'
    assert body["messages"][3] == {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": "Clause 4.2: glass is covered.",
    }


@respx.mock
def test_openai_chat_bad_tool_arguments_raise_schema_error() -> None:
    respx.post(OPENAI_URL).respond(
        json=openai_reply(
            {
                "content": None,
                "tool_calls": [
                    {"id": "c", "type": "function", "function": {"name": "t", "arguments": "{oops"}}
                ],
            }
        )
    )
    with pytest.raises(SchemaError):
        openai_gateway().chat(system="S", messages=CHAT[:1], tools=[LOOKUP])


# --- Ollama ------------------------------------------------------------------


def ollama_gateway() -> OllamaGateway:
    return OllamaGateway(base_url="http://localhost:11434", model="qwen2.5:7b-instruct")


def ollama_reply(message: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": "qwen2.5:7b-instruct",
        "message": {"role": "assistant", **message},
        "done": True,
        "prompt_eval_count": 90,
        "eval_count": 25,
    }


@respx.mock
def test_ollama_structured_parses_schema() -> None:
    respx.post(OLLAMA_URL).respond(
        json=ollama_reply({"content": json.dumps({"vendor": "Garage Lu", "total": 412.5})})
    )
    result = ollama_gateway().structured(
        system="Extract.", prompt="Invoice text", schema=Invoice, purpose="extract"
    )
    assert result.output == Invoice(vendor="Garage Lu", total=412.5)
    assert result.usage.input_tokens == 90
    assert result.usage.output_tokens == 25
    assert result.usage.cost_usd == 0.0
    assert result.usage.model == "qwen2.5:7b-instruct"
    assert result.usage.latency_ms >= 0


@respx.mock
def test_ollama_structured_request_shape() -> None:
    respx.post(OLLAMA_URL).respond(json=ollama_reply({"content": '{"vendor":"x","total":1}'}))
    ollama_gateway().structured(system="S", prompt="P", schema=Invoice, purpose="extract")
    body = last_request_body()
    assert body["stream"] is False
    assert body["format"] == Invoice.model_json_schema()
    assert body["options"] == {"temperature": 0, "num_ctx": 8192}
    assert body["messages"] == [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "P"},
    ]


@respx.mock
def test_ollama_invalid_json_raises_schema_error() -> None:
    respx.post(OLLAMA_URL).respond(json=ollama_reply({"content": "sorry, here you go: {"}))
    with pytest.raises(SchemaError) as err:
        ollama_gateway().structured(system="S", prompt="P", schema=Invoice, purpose="extract")
    assert err.value.raw.startswith("sorry")


@respx.mock
def test_ollama_chat_returns_tool_calls() -> None:
    respx.post(OLLAMA_URL).respond(
        json=ollama_reply(
            {
                "content": "",
                "tool_calls": [
                    {"function": {"name": "lookup_clause", "arguments": {"peril": "theft"}}}
                ],
            }
        )
    )
    turn = ollama_gateway().chat(system="S", messages=CHAT, tools=[LOOKUP])
    assert turn.text is None
    assert len(turn.tool_calls) == 1
    call = turn.tool_calls[0]
    assert (call.name, call.arguments) == ("lookup_clause", {"peril": "theft"})
    assert call.id  # Ollama may not supply one; the gateway must.
    body = last_request_body()
    assert body["tools"][0]["function"]["parameters"] == LOOKUP.input_schema
    assert body["messages"][2]["tool_calls"][0]["function"] == {
        "name": "lookup_clause",
        "arguments": {"peril": "glass"},
    }
    assert body["messages"][3] == {
        "role": "tool",
        "content": "Clause 4.2: glass is covered.",
        "tool_name": "lookup_clause",
    }
    assert body["options"] == {"temperature": 0, "num_ctx": 8192}


# --- Shared transport behaviour ---------------------------------------------


@respx.mock
def test_timeout_maps_to_gateway_error() -> None:
    respx.post(OLLAMA_URL).mock(side_effect=httpx.ReadTimeout("slow"))
    with pytest.raises(GatewayError) as err:
        ollama_gateway().structured(system="S", prompt="P", schema=Invoice, purpose="extract")
    assert "ollama" in str(err.value)


@respx.mock
def test_connection_error_maps_to_gateway_error() -> None:
    respx.post(OPENAI_URL).mock(side_effect=httpx.ConnectError("refused"))
    with pytest.raises(GatewayError) as err:
        openai_gateway().structured(system="S", prompt="P", schema=Invoice, purpose="extract")
    assert "openai" in str(err.value)


@respx.mock
def test_http_error_status_maps_to_gateway_error() -> None:
    respx.post(ANTHROPIC_URL).respond(status_code=529, json={"error": {"type": "overloaded"}})
    with pytest.raises(GatewayError) as err:
        anthropic_gateway().structured(system="S", prompt="P", schema=Invoice, purpose="extract")
    assert "529" in str(err.value)
    assert "anthropic" in str(err.value)
    assert err.value.status_code == 529


def test_cloud_flags_and_models() -> None:
    gateways: list[LLMGateway] = [anthropic_gateway(), openai_gateway(), ollama_gateway()]
    assert [g.is_cloud for g in gateways] == [True, True, False]
    assert [g.model for g in gateways] == ["claude-haiku-5-5", "gpt-5-mini", "qwen2.5:7b-instruct"]


def test_anthropic_waits_longer_than_the_shared_timeout() -> None:
    """A reply of up to MAX_TOKENS, thinking included, can take minutes."""
    from gateway.anthropic import TIMEOUT_SECONDS as ANTHROPIC_TIMEOUT
    from gateway.base import TIMEOUT_SECONDS

    gateway = AnthropicGateway(api_key="sk-test", model="claude-haiku-5-5")

    assert ANTHROPIC_TIMEOUT > TIMEOUT_SECONDS
    assert gateway._client.timeout.read == ANTHROPIC_TIMEOUT
