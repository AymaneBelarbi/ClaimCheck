import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine

from api.deps import get_engine
from api.main import app

pytestmark = pytest.mark.db


def test_health_reports_database_up(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "db": True}


def test_health_is_503_when_the_database_is_down() -> None:
    # Nothing listens on port 1.
    dead = create_engine("postgresql+psycopg://x:x@127.0.0.1:1/x", pool_pre_ping=False)
    app.dependency_overrides[get_engine] = lambda: dead
    try:
        response = TestClient(app).get("/health")
    finally:
        app.dependency_overrides.clear()
        dead.dispose()
    assert response.status_code == 503
    assert response.json()["db"] is False


def test_cors_allows_only_the_dev_ui_origin(client: TestClient) -> None:
    preflight = {
        "Access-Control-Request-Method": "GET",
        "Access-Control-Request-Headers": "authorization",
    }
    ok = client.options("/claims", headers={"Origin": "http://localhost:5173", **preflight})
    assert ok.headers["access-control-allow-origin"] == "http://localhost:5173"

    other = client.options("/claims", headers={"Origin": "http://evil.example", **preflight})
    assert "access-control-allow-origin" not in other.headers
