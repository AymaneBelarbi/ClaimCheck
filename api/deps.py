"""Dependencies shared by the routes: database, file storage, model gateway and embedder."""

from collections.abc import Iterator
from functools import lru_cache
from pathlib import Path
from typing import Annotated

from fastapi import Depends
from sqlalchemy import Engine
from sqlalchemy.orm import Session

from api.auth import Principal, current_principal
from core.settings import get_settings
from db.session import SessionFactory, engine_for, identified_session, session_factory
from gateway.base import LLMGateway
from gateway.factory import get_gateway
from rag.embedder import Embedder, FastEmbedEmbedder
from worker.storage import Storage


def api_engine() -> Engine:
    """The connection every API request uses: the claimcheck_app role, which row-level
    security applies to. The one place that picks the API's database role."""
    return engine_for(get_settings().app_database_url)


def get_engine() -> Engine:
    """`api_engine` as a dependency, so tests can point the app at their own database."""
    return api_engine()


CurrentPrincipal = Annotated[Principal, Depends(current_principal)]
ApiEngine = Annotated[Engine, Depends(get_engine)]


def get_session(principal: CurrentPrincipal, engine: ApiEngine) -> Iterator[Session]:
    """A session for one request, as the caller. Endpoints commit explicitly; anything else is
    rolled back. The caller's identity is set on every transaction the session begins, so it
    still holds after an endpoint rolls back and reads again."""
    with identified_session(engine, principal.role, principal.sub) as session:
        yield session


def get_storage() -> Storage:
    return Storage(Path(get_settings().storage_dir))


def get_session_factory(principal: CurrentPrincipal, engine: ApiEngine) -> SessionFactory:
    """Short transactions as the caller, for work (the agent's tools) that commits on its own
    rather than in the request's session."""
    return session_factory(engine, principal.role, principal.sub)


@lru_cache
def get_llm() -> LLMGateway:
    return get_gateway(get_settings())


@lru_cache
def get_embedder() -> Embedder:
    # Loaded on first use and kept: the model takes seconds to load.
    return FastEmbedEmbedder()
