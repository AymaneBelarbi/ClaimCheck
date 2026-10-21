import asyncio
from collections.abc import Iterator

from fastapi import FastAPI, Request, UploadFile
from fastapi.testclient import TestClient
from starlette.types import Message

from api.limits import BodyLimitMiddleware


def _app(max_bytes: int) -> TestClient:
    app = FastAPI()
    app.add_middleware(BodyLimitMiddleware, max_bytes=max_bytes)

    @app.post("/echo")
    async def echo(request: Request) -> dict[str, int]:
        return {"size": len(await request.body())}

    @app.post("/upload")
    async def upload(file: UploadFile) -> dict[str, str | None]:
        return {"name": file.filename}

    return TestClient(app)


def test_declared_oversize_is_rejected_before_the_body_is_read() -> None:
    response = _app(100).post("/echo", content=b"x" * 101)
    assert response.status_code == 413


def test_body_at_the_limit_passes() -> None:
    response = _app(100).post("/echo", content=b"x" * 100)
    assert response.status_code == 200
    assert response.json() == {"size": 100}


def test_chunked_oversize_is_cut_off_while_streaming() -> None:
    def chunks() -> Iterator[bytes]:
        for _ in range(20):
            yield b"x" * 10

    response = _app(100).post("/echo", content=chunks())
    assert response.status_code == 413


def test_oversize_multipart_is_413_not_a_parse_error() -> None:
    def chunks() -> Iterator[bytes]:
        yield b'--b\r\nContent-Disposition: form-data; name="file"; filename="a.pdf"\r\n\r\n'
        for _ in range(50):
            yield b"x" * 10
        yield b"\r\n--b--\r\n"

    response = _app(100).post(
        "/upload", content=chunks(), headers={"Content-Type": "multipart/form-data; boundary=b"}
    )
    assert response.status_code == 413


def test_unauthenticated_upload_is_refused_without_reading_the_body() -> None:
    from api.main import app

    read: list[str] = []

    async def receive() -> Message:
        read.append("receive")
        return {"type": "http.request", "body": b"x" * 1024, "more_body": True}

    sent: list[Message] = []

    async def send(message: Message) -> None:
        sent.append(message)

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/claims",
        "raw_path": b"/claims",
        "query_string": b"",
        "headers": [
            (b"content-type", b"multipart/form-data; boundary=b"),
            (b"content-length", b"200000000"),
        ],
    }
    asyncio.run(app(scope, receive, send))

    assert read == []
    assert sent[0]["status"] == 401
    assert (b"www-authenticate", b"Bearer") in sent[0]["headers"]


def test_early_refusals_carry_cors_headers(adjuster: dict[str, str]) -> None:
    from api.main import app

    client = TestClient(app)
    origin = {"Origin": "http://localhost:5173"}

    unauthenticated = client.post("/claims", headers=origin, content=b"x")
    assert unauthenticated.status_code == 401
    assert unauthenticated.headers["access-control-allow-origin"] == "http://localhost:5173"

    too_big = client.post(
        "/claims", headers={**adjuster, **origin, "Content-Length": str(10**10)}, content=b"x"
    )
    assert too_big.status_code == 413
    assert too_big.headers["access-control-allow-origin"] == "http://localhost:5173"


def test_bad_token_on_a_write_is_refused_early() -> None:
    from api.main import app

    response = TestClient(app).post(
        "/claims", headers={"Authorization": "Bearer nope"}, content=b"x"
    )
    assert response.status_code == 401
    assert response.headers["WWW-Authenticate"] == "Bearer"
