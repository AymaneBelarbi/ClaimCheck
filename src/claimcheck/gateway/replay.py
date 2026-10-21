"""Record-and-replay wrapper so CI and evals run without a model or a network.

One JSON file per request under the cassette directory. Failures from the inner gateway
(schema errors and transport errors alike) are recorded too and re-raised on replay, so a
caller that retries under a different key replays exactly the sequence that was recorded.
To retry a recorded failure against the real backend, delete its cassette.
A recorded response is served whenever its key is present, even if the request that
would be sent now differs from the one recorded; the drift is counted so an eval
report can show that its cassettes are stale.
"""

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ValidationError

from claimcheck.gateway.base import (
    CassetteMissError,
    ChatTurn,
    GatewayError,
    LLMGateway,
    LLMResult,
    Message,
    SchemaError,
    ToolSpec,
    Usage,
)

logger = logging.getLogger(__name__)

Kind = Literal["structured", "chat"]
_UNSAFE = re.compile(r"[^A-Za-z0-9._-]")


_PREFIX_CHARS = 80


def cassette_filename(key: str) -> str:
    """Map a cache key to a safe, collision-free file name.

    The readable part (`a:b:c.pdf` becomes `a__b__c.pdf`) is only for people browsing the
    directory; different keys can sanitise to the same text, so a short hash of the raw
    key makes the name unique.
    """
    prefix = _UNSAFE.sub("_", key.replace(":", "__")).lstrip(".")[:_PREFIX_CHARS] or "_"
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}-{digest}.json"


def _canonical_hash(request: dict[str, Any]) -> str:
    text = json.dumps(request, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class ReplayGateway:
    def __init__(self, inner: LLMGateway | None, directory: Path, record: bool) -> None:
        if record and inner is None:
            raise ValueError("recording cassettes needs an inner gateway to call")
        self._inner = inner
        self._dir = directory
        self._record = record
        self.drift_count = 0
        self.drifted_keys: list[str] = []
        self._recorded_model: str | None = None
        self._recorded_cloud: bool | None = None

    @property
    def model(self) -> str:
        if self._inner is not None:
            return self._inner.model
        self._load_recorded_identity()
        return self._recorded_model or "replay"

    @property
    def is_cloud(self) -> bool:
        if self._inner is not None:
            return self._inner.is_cloud
        self._load_recorded_identity()
        # With nothing recorded, assume cloud so callers redact rather than leak.
        return True if self._recorded_cloud is None else self._recorded_cloud

    def structured[T: BaseModel](
        self,
        *,
        system: str,
        prompt: str,
        schema: type[T],
        purpose: str,
        cache_key: str | None = None,
    ) -> LLMResult[T]:
        request = {
            "kind": "structured",
            "system": system,
            "prompt": prompt,
            "schema": schema.model_json_schema(),
        }
        key = cache_key or f"structured:{_canonical_hash(request)}"
        request_sha = _canonical_hash(request)
        stored = self._read(key, "structured", request_sha)
        if stored is not None:
            _raise_recorded_error(stored)
            try:
                output = schema.model_validate(stored["response"])
            except ValidationError as exc:
                raise SchemaError(
                    f"cassette {key!r} does not match {schema.__name__}",
                    raw=json.dumps(stored["response"]),
                ) from exc
            return LLMResult[T](output=output, usage=Usage.model_validate(stored["usage"]))
        inner = self._inner_for_miss(key)
        try:
            result = inner.structured(
                system=system, prompt=prompt, schema=schema, purpose=purpose, cache_key=cache_key
            )
        except (SchemaError, GatewayError) as exc:
            self._write_error(key, "structured", request_sha, exc)
            raise
        self._write(
            key, "structured", request_sha, result.output.model_dump(mode="json"), result.usage
        )
        return result

    def chat(
        self,
        *,
        system: str,
        messages: list[Message],
        tools: list[ToolSpec],
        cache_key: str | None = None,
    ) -> ChatTurn:
        request = {
            "kind": "chat",
            "system": system,
            "messages": [m.model_dump(mode="json") for m in messages],
            "tools": [t.model_dump(mode="json") for t in tools],
        }
        key = cache_key or f"chat:{_canonical_hash(request)}"
        request_sha = _canonical_hash(request)
        stored = self._read(key, "chat", request_sha)
        if stored is not None:
            _raise_recorded_error(stored)
            response = stored["response"]
            return ChatTurn(
                text=response["text"],
                tool_calls=response["tool_calls"],
                usage=Usage.model_validate(stored["usage"]),
            )
        inner = self._inner_for_miss(key)
        try:
            turn = inner.chat(system=system, messages=messages, tools=tools, cache_key=cache_key)
        except (SchemaError, GatewayError) as exc:
            self._write_error(key, "chat", request_sha, exc)
            raise
        response_json = {
            "text": turn.text,
            "tool_calls": [c.model_dump(mode="json") for c in turn.tool_calls],
        }
        self._write(key, "chat", request_sha, response_json, turn.usage)
        return turn

    def _inner_for_miss(self, key: str) -> LLMGateway:
        if not self._record or self._inner is None:
            raise CassetteMissError(key)
        return self._inner

    def _read(self, key: str, kind: Kind, request_sha: str) -> dict[str, Any] | None:
        path = self._dir / cassette_filename(key)
        if not path.exists():
            return None
        stored = _load(path)
        if stored.get("key") != key or stored.get("kind") != kind:
            # Another key sanitised to the same name, or the call type changed.
            return None
        if stored.get("request_sha256") != request_sha:
            self.drift_count += 1
            self.drifted_keys.append(key)
            logger.warning("cassette %r was recorded for a different request", key)
        self._recorded_model = stored.get("model", self._recorded_model)
        if isinstance(stored.get("is_cloud"), bool):
            self._recorded_cloud = stored["is_cloud"]
        return stored

    def _write(
        self, key: str, kind: Kind, request_sha: str, response: dict[str, Any], usage: Usage
    ) -> None:
        self._store(
            key,
            {
                "key": key,
                "request_sha256": request_sha,
                "kind": kind,
                "response": response,
                "usage": usage.model_dump(mode="json"),
            },
        )

    def _write_error(
        self, key: str, kind: Kind, request_sha: str, exc: SchemaError | GatewayError
    ) -> None:
        error: dict[str, Any]
        if isinstance(exc, SchemaError):
            error = {"type": "SchemaError", "message": str(exc), "raw": exc.raw}
        else:
            error = {
                "type": "GatewayError",
                "message": str(exc),
                "provider": exc.provider,
                "status_code": exc.status_code,
            }
        self._store(key, {"key": key, "request_sha256": request_sha, "kind": kind, "error": error})

    def _store(self, key: str, document: dict[str, Any]) -> None:
        assert self._inner is not None
        self._dir.mkdir(parents=True, exist_ok=True)
        path = self._dir / cassette_filename(key)
        if path.exists():
            existing = _load(path).get("key")
            if existing != key:
                raise GatewayError(
                    f"cassette {path.name} belongs to key {existing!r}, not {key!r}",
                    provider="replay",
                )
        document = {
            **document,
            "model": self._inner.model,
            "is_cloud": self._inner.is_cloud,
        }
        temp = path.with_suffix(".json.tmp")
        temp.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        temp.replace(path)

    def _load_recorded_identity(self) -> None:
        if self._recorded_model is not None or not self._dir.is_dir():
            return
        for path in sorted(self._dir.glob("*.json")):
            stored = _load(path)
            if "model" in stored:
                self._recorded_model = stored["model"]
                if isinstance(stored.get("is_cloud"), bool):
                    self._recorded_cloud = stored["is_cloud"]
                return


def _raise_recorded_error(stored: dict[str, Any]) -> None:
    error = stored.get("error")
    if not isinstance(error, dict):
        return
    message = str(error.get("message", "recorded error"))
    if error.get("type") == "GatewayError":
        status = error.get("status_code")
        raise GatewayError(
            message,
            provider=str(error.get("provider", "replay")),
            status_code=status if isinstance(status, int) else None,
        )
    raise SchemaError(message, raw=str(error.get("raw", "")))


def _load(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise GatewayError(f"cassette {path.name} is unreadable: {exc}", provider="replay") from exc
    if not isinstance(data, dict):
        raise GatewayError(f"cassette {path.name} is not a JSON object", provider="replay")
    return data
