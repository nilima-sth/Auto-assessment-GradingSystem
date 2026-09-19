from __future__ import annotations


class InjectedRetrievalTimeout(TimeoutError):
    """Controlled test-only retrieval failure."""


def failing_retrieval(**_kwargs):
    raise InjectedRetrievalTimeout("injected retrieval timeout")

