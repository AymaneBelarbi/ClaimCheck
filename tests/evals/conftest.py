"""Evals that need a database reuse the throwaway-database factory from tests/db."""

from tests.db.conftest import make_database  # noqa: F401  (re-exported fixture)
