"""Anthropic Messages API over plain httpx."""

import logging
from typing import Any

import httpx
from pydantic import BaseModel

from gateway.base import (
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

logger = logging.getLogger(__name__)

API_VERSION = "2023-06-01"
# Several current models think by default and thinking counts against max_tokens, so the
# cap leaves room for it. A reply that still hits the cap is reported, not parsed.
MAX_TOKENS = 16_000
# The shared 60 s is too short for a reply that may use all of MAX_TOKENS.
TIMEOUT_SECONDS = 180.0
# Keywords the structured-output format rejects. Pydantic still enforces them on the reply.
_UNSUPPORTED_KEYWORDS = frozenset(
    {
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "multipleOf",
        "minLength",
        "maxLength",
        "minItems",
        "maxItems",
    }
)


def thinking_config(model: str) -> dict[str, str] | None:
    """The `thinking` request field that is valid for this model, or None to omit it.

    Per the API documentation (checked 2026-10-20), thinking cannot be switched off the same
    way everywhere: Haiku takes `disabled`, Sonnet 5.5 rejects it and offers `between_tools`,
    and the Opus 5.5 and Fable models cannot turn it off at all, so for those the field is
    left out and the model's adaptive default applies. Thinking blocks are never sent back
    (Message has no place for them), which is why it is off wherever the API allows that.
    """
    if model.startswith("claude-haiku-"):
        return {"type": "disabled"}
    if model.startswith("claude-sonnet-5-5"):
        return {"type": "between_tools"}
    return None


class AnthropicGateway:
    is_cloud = True

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str = "https://api.anthropic.com",
        client: httpx.Client | None = None,
    ) -> None:
        self.model = model
        self._api_key = api_key
        self._url = f"{base_url.rstrip('/')}/v1/messages"
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
        # Structured outputs constrain the reply text to the schema. Per the API documentation
        # (checked 2026-10-20; no key has been used here), a forced tool_choice would do the
        # same job but is a 400 on Sonnet 5.5, Opus 5.5 and Fable 5.1, and current Claude
        # models accept only their default sampling parameters and answer 400 to
        # temperature=0, so no temperature is sent.
        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": MAX_TOKENS,
            "system": system,
            "messages": [{"role": "user", "content": prompt}],
            "output_config": {
                "format": {
                    "type": "json_schema",
                    "schema": _without_unsupported(strict_schema(schema.model_json_schema())),
                }
            },
        }
        thinking = thinking_config(self.model)
        if thinking is not None:
            payload["thinking"] = thinking
        body, latency_ms = self._post(payload)
        text = "".join(b.get("text", "") for b in _blocks(body) if b.get("type") == "text")
        stop_reason = body.get("stop_reason")
        if stop_reason == "max_tokens":
            raise SchemaError(
                f"anthropic stopped at max_tokens ({MAX_TOKENS}) before finishing the "
                f"{schema.__name__} output",
                raw=text,
            )
        if stop_reason == "refusal":
            raise SchemaError("anthropic declined the request", raw=text)
        output = parse_output(schema, text)
        return LLMResult[T](output=output, usage=self._usage(body, latency_ms))

    def chat(
        self,
        *,
        system: str,
        messages: list[Message],
        tools: list[ToolSpec],
        cache_key: str | None = None,
    ) -> ChatTurn:
        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": MAX_TOKENS,
            "system": system,
            "messages": _to_wire(messages),
        }
        # Thinking blocks would have to be echoed back with every tool result, and
        # the Message type has nowhere to keep them. The agent does not need them.
        thinking = thinking_config(self.model)
        if thinking is not None:
            payload["thinking"] = thinking
        if tools:
            payload["tools"] = [
                {"name": t.name, "description": t.description, "input_schema": t.input_schema}
                for t in tools
            ]
            payload["tool_choice"] = {"type": "auto"}
        body, latency_ms = self._post(payload)
        blocks = _blocks(body)
        text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
        calls = [
            ToolCall(id=b["id"], name=b["name"], arguments=b.get("input", {}))
            for b in blocks
            if b.get("type") == "tool_use"
        ]
        return ChatTurn(text=text or None, tool_calls=calls, usage=self._usage(body, latency_ms))

    def _post(self, payload: dict[str, Any]) -> tuple[dict[str, Any], float]:
        headers = {
            "x-api-key": self._api_key,
            "anthropic-version": API_VERSION,
            "content-type": "application/json",
        }
        return post_json(
            self._client, self._url, provider="anthropic", headers=headers, payload=payload
        )

    def _usage(self, body: dict[str, Any], latency_ms: float) -> Usage:
        raw = body.get("usage") or {}
        tokens_in = int(raw.get("input_tokens", 0))
        tokens_out = int(raw.get("output_tokens", 0))
        return Usage(
            model=self.model,
            input_tokens=tokens_in,
            output_tokens=tokens_out,
            cost_usd=cost(self.model, tokens_in, tokens_out),
            latency_ms=latency_ms,
        )


def _without_unsupported(node: Any) -> Any:
    if isinstance(node, dict):
        cleaned: dict[str, Any] = {}
        for key, value in node.items():
            if key in _UNSUPPORTED_KEYWORDS:
                continue
            if key in ("properties", "$defs") and isinstance(value, dict):
                # Keys here are field or model names, which may collide with keywords.
                cleaned[key] = {name: _without_unsupported(child) for name, child in value.items()}
            else:
                cleaned[key] = _without_unsupported(value)
        return cleaned
    if isinstance(node, list):
        return [_without_unsupported(item) for item in node]
    return node


def _blocks(body: dict[str, Any]) -> list[dict[str, Any]]:
    content = body.get("content")
    if not isinstance(content, list):
        return []
    return [b for b in content if isinstance(b, dict)]


def _to_wire(messages: list[Message]) -> list[dict[str, Any]]:
    """Map our messages onto Anthropic's block format.

    Tool results travel as blocks in a user turn, and the results answering one
    assistant turn must share a single user message.
    """
    wire: list[dict[str, Any]] = []
    for m in messages:
        if m.role == "tool":
            result = {"type": "tool_result", "tool_use_id": m.tool_call_id, "content": m.content}
            previous = wire[-1] if wire else None
            if (
                previous is not None
                and previous["role"] == "user"
                and isinstance(previous["content"], list)
                and previous["content"][-1].get("type") == "tool_result"
            ):
                previous["content"].append(result)
            else:
                wire.append({"role": "user", "content": [result]})
        elif m.role == "assistant" and m.tool_calls:
            blocks: list[dict[str, Any]] = []
            if m.content:
                blocks.append({"type": "text", "text": m.content})
            blocks.extend(
                {"type": "tool_use", "id": c.id, "name": c.name, "input": c.arguments}
                for c in m.tool_calls
            )
            wire.append({"role": "assistant", "content": blocks})
        else:
            wire.append({"role": m.role, "content": m.content})
    return wire
