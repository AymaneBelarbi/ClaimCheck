"""The ClaimCheck HTTP API.

uvicorn claimcheck.api.main:app
"""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from claimcheck.api.auth import PLACEHOLDER_SECRETS, BearerPrecheckMiddleware, signing_secret
from claimcheck.api.limits import BodyLimitMiddleware
from claimcheck.api.routes import claims, health
from claimcheck.core.settings import get_settings
from claimcheck.core.tracing import setup_tracing

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    # Without a secret no request could be authenticated, so the API does not start.
    if signing_secret() in PLACEHOLDER_SECRETS:
        logger.warning(
            "JWT_SECRET is a published placeholder: anyone can sign adjuster tokens for this "
            "API. Fine for a local demo; set a random JWT_SECRET anywhere else (.env.example)."
        )
    setup_tracing("claimcheck-api", get_settings().otel_endpoint)
    yield


app = FastAPI(title="ClaimCheck", version="0.1.0", lifespan=lifespan)

# The last middleware added is the outermost. CORS wraps the others so that their early
# 401 and 413 answers still carry the headers the browser needs to show them.
app.add_middleware(BodyLimitMiddleware)
app.add_middleware(BearerPrecheckMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type", "Idempotency-Key"],
)

app.include_router(health.router)
app.include_router(claims.router)
