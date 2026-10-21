"""OpenAI Chat Completions over plain httpx."""

import json
from typing import Any

import httpx
from pydantic import BaseModel

from gateway.base import (
    TIMEOUT_SECONDS,
    ChatTurn,
    LLMResult,
    Message,
    SchemaError,
    ToolCall,
    ToolSpec,
    Usage,
    parse_output,
    post_json,
    strict_schema,
)
from gateway.pricing import cost

MAX_COMPLETION_TOKENS = 8192


class OpenAIGateway:
    is_cloud = True

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str = "https://api.openai.com",
        client: httpx.Client | None = None,
    ) -> None:
        self.model = model
        self._api_key = api_key
        self._url = f"{base_url.rstrip('/')}/v1/chat/completions"
        self._client = client or httpx.Client(timeout=TIMEOUT_SECONDS)

    def structured[T: BaseModel](
        self,
        *,
        system: str,
        prompt: str,
        schema: type[T],
        purpose: str,
        cache_key: str | None = None,
    ) -> LLMResult[T]:
        payload = self._payload(
            [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
        )
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": schema.__name__,
                "strict": True,
                "schema": strict_schema(schema.model_json_schema()),
            },
        }
        body, latency_ms = post_json(
            self._client, self._url, provider="openai", headers=self._headers(), payload=payload
        )
        message = _message(body)
        refusal = message.get("refusal")
        if refusal:
            raise SchemaError("openai refused the request", raw=str(refusal))
        raw = message.get("content") or ""
        output = parse_output(schema, raw)
        return LLMResult[T](output=output, usage=self._usage(body, latency_ms))

    def chat(
        self,
        *,
        system: str,
        messages: list[Message],
        tools: list[ToolSpec],
        cache_key: str | None = None,
    ) -> ChatTurn:
        payload = self._payload([{"role": "system", "content": system}, *_to_wire(messages)])
        if tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.input_schema,
                    },
                }
                for t in tools
            ]
        body, latency_ms = post_json(
            self._client, self._url, provider="openai", headers=self._headers(), payload=payload
        )
        message = _message(body)
        calls: list[ToolCall] = []
        for call in message.get("tool_calls") or []:
            function = call.get("function", {})
            raw_args = function.get("arguments") or "{}"
            try:
                arguments = json.loads(raw_args)
            except ValueError as exc:
                raise SchemaError(
                    f"tool call {function.get('name')!r} has unparsable arguments", raw=raw_args
                ) from exc
            if not isinstance(arguments, dict):
                raise SchemaError(
                    f"tool call {function.get('name')!r} arguments are not an object",
                    raw=raw_args,
                )
            calls.append(ToolCall(id=call["id"], name=function["name"], arguments=arguments))
        text = message.get("content") or None
        return ChatTurn(text=text, tool_calls=calls, usage=self._usage(body, latency_ms))

    def _headers(self) -> dict[str, str]:
        return {"authorization": f"Bearer {self._api_key}", "content-type": "application/json"}

    def _payload(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_completion_tokens": MAX_COMPLETION_TOKENS,
        }
        # The gpt-5 family only accepts its default temperature; older models take 0.
        if not self.model.startswith("gpt-5"):
            payload["temperature"] = 0
        return payload

    def _usage(self, body: dict[str, Any], latency_ms: float) -> Usage:
        raw = body.get("usage") or {}
        tokens_in = int(raw.get("prompt_tokens", 0))
        tokens_out = int(raw.get("completion_tokens", 0))
        return Usage(
            model=self.model,
            input_tokens=tokens_in,
            output_tokens=tokens_out,
            cost_usd=cost(self.model, tokens_in, tokens_out),
            latency_ms=latency_ms,
        )


def _message(body: dict[str, Any]) -> dict[str, Any]:
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        raise SchemaError("openai reply has no choices", raw=json.dumps(body)[:500])
    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise SchemaError("openai reply has no message", raw=json.dumps(body)[:500])
    return message


def _to_wire(messages: list[Message]) -> list[dict[str, Any]]:
    wire: list[dict[str, Any]] = []
    for m in messages:
        if m.role == "tool":
            wire.append({"role": "tool", "tool_call_id": m.tool_call_id, "content": m.content})
        elif m.role == "assistant" and m.tool_calls:
            wire.append(
                {
                    "role": "assistant",
                    "content": m.content or None,
                    "tool_calls": [
                        {
                            "id": c.id,
                            "type": "function",
                            "function": {"name": c.name, "arguments": json.dumps(c.arguments)},
                        }
                        for c in m.tool_calls
                    ],
                }
            )
        else:
            wire.append({"role": m.role, "content": m.content})
    return wire
