"""Upload limits, and the request-size cap that enforces them before a body is parsed.

Starlette writes each uploaded file to a temporary file while it parses the form, before any
endpoint code runs. Without a cap on the request as a whole, a client could fill the disk
with a body no endpoint ever gets to reject.
"""

from fastapi import HTTPException
from starlette.datastructures import Headers
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

MAX_FILES = 10
MAX_FILE_BYTES = 20 * 1024 * 1024
# Room for the form fields and multipart framing around the largest allowed packet.
MAX_REQUEST_BYTES = MAX_FILES * MAX_FILE_BYTES + 1024 * 1024

TOO_LARGE = "request body too large"


class BodyLimitMiddleware:
    """Answer 413 as soon as a request body is known or seen to exceed `max_bytes`."""

    def __init__(self, app: ASGIApp, max_bytes: int = MAX_REQUEST_BYTES) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        declared = Headers(scope=scope).get("content-length")
        if declared is not None and declared.isdigit() and int(declared) > self.max_bytes:
            await JSONResponse({"detail": TOO_LARGE}, status_code=413)(scope, receive, send)
            return

        received = 0

        async def counted() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    # FastAPI re-raises an HTTPException from body parsing as is.
                    raise HTTPException(status_code=413, detail=TOO_LARGE)
            return message

        await self.app(scope, counted, send)
