from __future__ import annotations

import time
from collections import defaultdict, deque
from threading import Lock

from fastapi import HTTPException, Request, status

from backend.app.core.config import get_settings

_requests: dict[str, deque[float]] = defaultdict(deque)
_lock = Lock()


def reset_rate_limits() -> None:
    with _lock:
        _requests.clear()


async def expensive_endpoint_rate_limit(request: Request) -> None:
    settings = get_settings()
    limit = max(1, settings.rate_limit_requests)
    window = max(1, settings.rate_limit_window_seconds)
    client = request.client.host if request.client else "unknown"
    key = f"{client}:{request.url.path}"
    now = time.monotonic()

    with _lock:
        hits = _requests[key]
        while hits and now - hits[0] >= window:
            hits.popleft()
        if len(hits) >= limit:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded. Try again later; limit is {limit} requests per {window} seconds.",
            )
        hits.append(now)
