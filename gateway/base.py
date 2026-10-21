"""Types, errors and the HTTP plumbing shared by every LLM backend."""

import copy
import logging
import time
from typing import Any, Literal, Protocol

import httpx
from pydantic import BaseModel, Field, ValidationError

logger = logging.getLogger(__name__)

TIMEOUT_SECONDS = 60.0


class GatewayError(Exception):
    """The provider could not be reached or answered with an error status."""

    def __init__(self, message: str, *, provider: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.provider = provider
        self.status_code = status_code


class SchemaError(Exception):
    """The provider answered, but the output was not valid JSON for the requested schema."""

    def __init__(self, message: str, *, raw: str) -> None:
        super().__init__(message)
        self.raw = raw


class CassetteMissError(Exception):
    """Replay was asked for a response that was never recorded."""

    def __init__(self, key: str) -> None:
        super().__init__(f"no cassette recorded for key {key!r}")
        self.key = key


class ToolSpec(BaseModel):
    name: str
    description: str
    input_schema: dict[str, Any]


class ToolCall(BaseModel):
    id: str
    name: str
    arguments: dict[str, Any]


class Message(BaseModel):
    role: Literal["user", "assistant", "tool"]
    content: str
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_call_id: str | None = None


class Usage(BaseModel):
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: float


class LLMResult[T: BaseModel](BaseModel):
    output: T
    usage: Usage


class ChatTurn(BaseModel):
    text: str | None
    tool_calls: list[ToolCall]
    usage: Usage


class LLMGateway(Protocol):
    """The one interface the rest of the system talks to.

    `is_cloud` tells callers whether text must be redacted before it is sent.
    """

    @property
    def is_cloud(self) -> bool: ...

    @property
    def model(self) -> str: ...

    def structured[T: BaseModel](
        self,
        *,
        system: str,
        prompt: str,
        schema: type[T],
        purpose: str,
        cache_key: str | None = None,
    ) -> LLMResult[T]: ...

    def chat(
        self,
        *,
        system: str,
        messages: list[Message],
        tools: list[ToolSpec],
        cache_key: str | None = None,
    ) -> ChatTurn: ...


def post_json(
    client: httpx.Client,
    url: str,
    *,
    provider: str,
    headers: dict[str, str],
    payload: dict[str, Any],
) -> tuple[dict[str, Any], float]:
    """POST a JSON body and return (decoded body, elapsed milliseconds).

    Every failure mode a caller cannot act on differently (timeout, refused
    connection, error status, non-JSON body) becomes one GatewayError.
    """
    started = time.perf_counter()
    try:
        response = client.post(url, headers=headers, json=payload)
    except httpx.TimeoutException as exc:
        raise GatewayError(f"{provider} request timed out: {exc!r}", provider=provider) from exc
    except httpx.TransportError as exc:
        raise GatewayError(f"{provider} request failed: {exc!r}", provider=provider) from exc
    latency_ms = (time.perf_counter() - started) * 1000
    if not response.is_success:
        raise GatewayError(
            f"{provider} returned HTTP {response.status_code}: {response.text[:500]}",
            provider=provider,
            status_code=response.status_code,
        )
    try:
        body = response.json()
    except ValueError as exc:
        raise GatewayError(
            f"{provider} returned a non-JSON body: {response.text[:200]}",
            provider=provider,
            status_code=response.status_code,
        ) from exc
    if not isinstance(body, dict):
        raise GatewayError(f"{provider} returned an unexpected JSON body", provider=provider)
    return body, latency_ms


def parse_output[T: BaseModel](schema: type[T], raw: str) -> T:
    """Validate raw model text against the schema, keeping the text on failure."""
    try:
        return schema.model_validate_json(raw)
    except ValidationError as exc:
        raise SchemaError(
            f"output does not match {schema.__name__}: {exc.error_count()} validation error(s)",
            raw=raw,
        ) from exc


def strict_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Rewrite a Pydantic JSON schema to what strict structured-output modes accept.

    OpenAI strict mode and Anthropic's `output_config.format` both want every object
    closed (`additionalProperties: false`) with all properties listed as required;
    defaults are not allowed. Optional fields stay optional because Pydantic already
    types them as `anyOf [..., null]`.

    Free-form objects (`dict[...]` fields) cannot be closed, so they raise ValueError
    naming the field instead of being sent and rejected by the provider.
    """
    result = copy.deepcopy(schema)
    _close_objects(result, [])
    return result


def _close_objects(node: Any, path: list[str]) -> None:
    if isinstance(node, dict):
        node.pop("default", None)
        properties = node.get("properties")
        if node.get("type") == "object":
            if not isinstance(properties, dict):
                where = ".".join(path) or "<root>"
                raise ValueError(
                    f"field {where!r} is a free-form object (dict), which strict structured "
                    "output cannot express; use a list of key/value models instead"
                )
            node["additionalProperties"] = False
            node["required"] = list(properties)
        for key, value in node.items():
            if key == "properties" and isinstance(value, dict):
                for name, child in value.items():
                    _close_objects(child, [*path, name])
            elif key in ("$defs", "definitions") and isinstance(value, dict):
                for name, child in value.items():
                    _close_objects(child, [name])
            else:
                _close_objects(value, path)
    elif isinstance(node, list):
        for item in node:
            _close_objects(item, path)
