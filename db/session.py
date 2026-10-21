"""Engine and session helpers.

Row-level security (migration 0002) shows a transaction only the claims its `app.role` and
`app.user_id` settings allow, and nothing at all when they are unset. Every session opened here
carries an identity and sets it at the start of each of its transactions, including the one a
session begins after a rollback.
"""

from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager
from functools import lru_cache
from typing import Final, Literal

from sqlalchemy import Connection, Engine, create_engine, event, text
from sqlalchemy.orm import Session, SessionTransaction

from core.settings import get_settings
from core.types import Role

# The worker's identity: it processes every claim but is no person's token.
SYSTEM: Final = "system"
DbRole = Role | Literal["system"]

_SET_IDENTITY = text(
    "SELECT set_config('app.role', :role, true), set_config('app.user_id', :uid, true)"
)


@lru_cache
def engine_for(url: str) -> Engine:
    return create_engine(url, pool_pre_ping=True)


def identified_session(engine: Engine, role: DbRole, user_id: str) -> Session:
    """A session whose every transaction runs as `role` / `user_id`.

    set_config(..., true) is the bind-parameter form of SET LOCAL: the values vanish at
    commit or rollback, so a pooled connection never carries one caller's identity to the
    next. That is also why they are set again on each transaction rather than once.

    Objects keep their loaded values after a commit, so ids and the like read inside one
    transaction can be used after it without another query.
    """
    session = Session(engine, expire_on_commit=False)
    params = {"role": str(role), "uid": user_id}

    @event.listens_for(session, "after_begin")
    def set_identity(_: Session, transaction: SessionTransaction, conn: Connection) -> None:
        conn.execute(_SET_IDENTITY, params)

    return session


# Opens one transaction: committed when the block exits normally, rolled back otherwise.
SessionFactory = Callable[[], AbstractContextManager[Session]]


def session_factory(engine: Engine, role: DbRole, user_id: str) -> SessionFactory:
    """Transactions on `engine` as `role` / `user_id`, for code that must not hold one open
    throughout (the worker between pipeline phases, the agent's tools between calls)."""

    @contextmanager
    def transaction() -> Iterator[Session]:
        with identified_session(engine, role, user_id) as session, session.begin():
            yield session

    return transaction


@contextmanager
def session_scope(role: DbRole, user_id: str, url: str | None = None) -> Iterator[Session]:
    """One transaction as `role` / `user_id`; by default on the API's row-level-secured role."""
    engine = engine_for(url or get_settings().app_database_url)
    with session_factory(engine, role, user_id)() as session:
        yield session
