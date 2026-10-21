import os
from collections.abc import Iterator

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from core.settings import Settings, get_settings


@pytest.fixture(autouse=True)
def _isolated_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Start every test from a known environment.

    Variables from the developer's shell must not leak into tests. JWT_SECRET is optional in
    Settings, but api.auth refuses to run without it, so tests that sign or verify tokens
    need one.
    """
    for name in Settings.model_fields:
        monkeypatch.delenv(name.upper(), raising=False)
    monkeypatch.setenv(
        "JWT_SECRET", "test-secret-not-for-production-0123456789-0123456789-0123456789-abcdef"
    )
    get_settings.cache_clear()


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if os.environ.get("TEST_DATABASE_URL"):
        return
    skip = pytest.mark.skip(reason="TEST_DATABASE_URL not set (needs a Postgres with pgvector)")
    for item in items:
        if "db" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def _span_exporter() -> InMemorySpanExporter:
    # The global tracer provider can be set once per process, so the whole session shares one.
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    return exporter


@pytest.fixture
def spans(_span_exporter: InMemorySpanExporter) -> Iterator[InMemorySpanExporter]:
    """Finished spans of the current test."""
    _span_exporter.clear()
    yield _span_exporter
    _span_exporter.clear()
