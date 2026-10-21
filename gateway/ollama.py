"""Ollama /api/chat over plain httpx. Runs on the client's own hardware, so is_cloud is False."""

from typing import Any

import httpx
from pydantic import BaseModel

from gateway.base import (
    TIMEOUT_SECONDS,
    ChatTurn,
    LLMResult,
    Message,
    ToolCall,
    ToolSpec,
    Usage,
    parse_output,
    post_json,
)

# A prompt longer than Ollama's context window loses its beginning (the system prompt) without
# an error. The longest recorded agent turn is 2,890 input tokens (evals/cassettes) and each tool
# result adds more, so the window is set here rather than left to the server's default.
NUM_CTX = 8192


class OllamaGateway:
    is_cloud = False

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        timeout: float = TIMEOUT_SECONDS,
        client: httpx.Client | None = None,
    ) -> None:
        self.model = model
        self.timeout = timeout
        self._url = f"{base_url.rstrip('/')}/api/chat"
        self._client = client or httpx.Client(timeout=timeout)

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
        payload["format"] = schema.model_json_schema()
        body, latency_ms = self._post(payload)
        raw = str(_message(body).get("content") or "")
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
        body, latency_ms = self._post(payload)
        message = _message(body)
        calls: list[ToolCall] = []
        for index, call in enumerate(message.get("tool_calls") or []):
            function = call.get("function", {})
            arguments = function.get("arguments") or {}
            # Ollama does not always number its tool calls; the agent needs an id to answer.
            call_id = call.get("id") or f"call_{index}"
            calls.append(ToolCall(id=call_id, name=function["name"], arguments=arguments))
        text = message.get("content") or None
        return ChatTurn(text=text, tool_calls=calls, usage=self._usage(body, latency_ms))

    def _payload(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": 0, "num_ctx": NUM_CTX},
        }

    def _post(self, payload: dict[str, Any]) -> tuple[dict[str, Any], float]:
        return post_json(
            self._client,
            self._url,
            provider="ollama",
            headers={"content-type": "application/json"},
            payload=payload,
        )

    def _usage(self, body: dict[str, Any], latency_ms: float) -> Usage:
        return Usage(
            model=self.model,
            input_tokens=int(body.get("prompt_eval_count") or 0),
            output_tokens=int(body.get("eval_count") or 0),
            cost_usd=0.0,
            latency_ms=latency_ms,
        )


def _message(body: dict[str, Any]) -> dict[str, Any]:
    message = body.get("message")
    return message if isinstance(message, dict) else {}


def _to_wire(messages: list[Message]) -> list[dict[str, Any]]:
    # Ollama's tool messages name the tool rather than the call, so remember which
    # tool each call id belonged to.
    tool_names: dict[str, str] = {}
    wire: list[dict[str, Any]] = []
    for m in messages:
        if m.role == "tool":
            entry: dict[str, Any] = {"role": "tool", "content": m.content}
            name = tool_names.get(m.tool_call_id or "")
            if name:
                entry["tool_name"] = name
            wire.append(entry)
        elif m.role == "assistant" and m.tool_calls:
            for c in m.tool_calls:
                tool_names[c.id] = c.name
            wire.append(
                {
                    "role": "assistant",
                    "content": m.content,
                    "tool_calls": [
                        {"function": {"name": c.name, "arguments": c.arguments}}
                        for c in m.tool_calls
                    ],
                }
            )
        else:
            wire.append({"role": m.role, "content": m.content})
    return wire
