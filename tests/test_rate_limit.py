import asyncio

import pytest
from fastapi import HTTPException
from starlette.requests import Request

from backend.app.core.config import get_settings
from backend.app.core.rate_limit import expensive_endpoint_rate_limit, reset_rate_limits


def _request(path: str = "/api/grading/evaluate") -> Request:
    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": path,
            "headers": [],
            "client": ("127.0.0.1", 12345),
            "server": ("testserver", 80),
            "scheme": "http",
        }
    )


def test_expensive_endpoint_rate_limit_returns_429(monkeypatch) -> None:
    get_settings.cache_clear()
    reset_rate_limits()
    monkeypatch.setenv("RATE_LIMIT_REQUESTS", "1")
    monkeypatch.setenv("RATE_LIMIT_WINDOW_SECONDS", "60")

    asyncio.run(expensive_endpoint_rate_limit(_request()))
    with pytest.raises(HTTPException) as exc:
        asyncio.run(expensive_endpoint_rate_limit(_request()))

    assert exc.value.status_code == 429
    get_settings.cache_clear()
    reset_rate_limits()
