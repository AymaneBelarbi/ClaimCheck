"""strict_schema must close every object, including those under $defs."""

from typing import Any

import pytest
from pydantic import BaseModel

from gateway.base import strict_schema


class LLMField[T](BaseModel):
    value: T | None = None
    confidence: float = 0.0
    source_text: str | None = None


class LineItem(BaseModel):
    description: str
    amount: float
    part_number: str | None = None


class Extraction(BaseModel):
    vendor: LLMField[str]
    items: LLMField[list[LineItem]]
    total: LLMField[float] | None = None
    notes: list[str] | None = None


def objects(node: Any) -> list[dict[str, Any]]:
    """Every dict in the schema that describes an object."""
    found: list[dict[str, Any]] = []
    if isinstance(node, dict):
        if node.get("type") == "object":
            found.append(node)
        for value in node.values():
            found.extend(objects(value))
    elif isinstance(node, list):
        for item in node:
            found.extend(objects(item))
    return found


def test_nested_generic_model_is_fully_closed() -> None:
    raw = Extraction.model_json_schema()
    schema = strict_schema(raw)
    found = objects(schema)
    # Extraction itself plus each LLMField[...] and LineItem under $defs.
    assert len(found) >= 5
    assert set(schema["$defs"]) >= {"LineItem"}
    for obj in found:
        assert obj["additionalProperties"] is False
        assert obj["required"] == list(obj["properties"])


def test_every_property_is_required_including_optional_ones() -> None:
    schema = strict_schema(Extraction.model_json_schema())
    assert schema["required"] == ["vendor", "items", "total", "notes"]
    line_item = schema["$defs"]["LineItem"]
    assert line_item["required"] == ["description", "amount", "part_number"]
    # Optional stays expressible as a nullable type, not as a missing key.
    assert {"type": "null"} in line_item["properties"]["part_number"]["anyOf"]


def test_defaults_are_removed_and_input_is_not_mutated() -> None:
    raw = Extraction.model_json_schema()
    before = repr(raw)
    schema = strict_schema(raw)
    assert repr(raw) == before
    for definition in schema["$defs"].values():
        for prop in definition["properties"].values():
            assert "default" not in prop


def test_dict_field_raises_naming_the_field() -> None:
    class WithMeta(BaseModel):
        name: str
        meta: dict[str, str]

    with pytest.raises(ValueError, match="'meta'"):
        strict_schema(WithMeta.model_json_schema())


def test_untyped_and_optional_dict_fields_raise_too() -> None:
    class Loose(BaseModel):
        extra: dict[str, Any] | None = None

    with pytest.raises(ValueError, match="'extra'"):
        strict_schema(Loose.model_json_schema())


def test_dict_inside_a_nested_model_names_the_model_and_field() -> None:
    class Inner(BaseModel):
        attrs: dict[str, int]

    class Outer(BaseModel):
        inner: Inner

    with pytest.raises(ValueError, match=r"'Inner\.attrs'"):
        strict_schema(Outer.model_json_schema())
