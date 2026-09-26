from __future__ import annotations


class InjectedRetrievalTimeout(TimeoutError):
    """Retrieval error used only by the evaluation test."""


def failing_retrieval(**_kwargs):
    raise InjectedRetrievalTimeout("injected retrieval timeout")
