"""Database fixtures for the retrieval tests: the shared migrated database from tests/db."""

from collections.abc import Iterator

import pytest
from sqlalchemy import create_engine
from sqlalchemy.engine import URL
from sqlalchemy.orm import Session

from tests.db.conftest import migrated_url  # noqa: F401  (re-exported fixture)


@pytest.fixture
def session(migrated_url: URL) -> Iterator[Session]:  # noqa: F811
    """A session on the migrated database; everything a test writes is rolled back."""
    engine = create_engine(migrated_url)
    with Session(engine) as s:
        yield s
        s.rollback()
    engine.dispose()
