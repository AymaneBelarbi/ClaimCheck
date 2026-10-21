from typing import Annotated

from fastapi import APIRouter, Depends, Response
from sqlalchemy import Engine, text
from sqlalchemy.exc import SQLAlchemyError

from claimcheck.api.deps import get_engine
from claimcheck.api.schemas import Health

router = APIRouter()


@router.get("/health")
def health(response: Response, engine: Annotated[Engine, Depends(get_engine)]) -> Health:
    # A plain connection: no caller and no table, so no identity is needed.
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except SQLAlchemyError:
        response.status_code = 503
        return Health(status="unavailable", db=False)
    return Health(status="ok", db=True)
