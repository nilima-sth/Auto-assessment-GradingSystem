from backend.app.api.health import health


def test_health_endpoint() -> None:
    assert health() == {"status": "ok"}
